"""Experiment 1: CLMM Event Reconstruction and LP Realized PnL.

Pipeline stages:
  raw events → zero-value filter → price attach → burn+ match →
  left-censor filter → FIFO position reconstruction → PnL computation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from data.generate_synthetic_data import generate_all_pool_data
from src.event_schema import POOL_REGISTRY
from src.event_processor import (
    standardize_events, sort_events_chronologically,
    filter_zero_value_events, filter_lps_with_min_mint_burn,
    build_event_stream,
)
from src.price_attacher import build_price_series, attach_price_to_events
from src.burn_plus_matcher import match_burn_collect, validate_burn_plus
from src.dynamic_balance import (
    filter_left_censored, reconstruct_positions, compute_position_capitals,
)
from src.pnl_calculator import (
    positions_to_dataframe, aggregate_lp_pool_pnl, compute_validation_counts,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

POOL_IDS = [
    'uniswap_v3_005',
    'aerodrome_ss_005',
    'pancakeswap_v3_001',
    'sushiswap_v3_001',
]

# Paper targets for comparison (from the actual dataset, synthetic will differ)
PAPER_AFTER_FILTER = 8_019_145
PAPER_AFTER_RECONSTRUCTION = 5_244_345


def _to_unix_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the timestamp column from pandas Timestamp to Unix integer seconds."""
    if 'timestamp' in df.columns:
        df = df.copy()
        df['timestamp'] = (
            pd.to_datetime(df['timestamp']).astype('int64') // 10 ** 9
        )
    return df


def _drop_price_col(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the synthetic 'price' column to prevent merge_asof column conflict."""
    if 'price' in df.columns:
        df = df.drop(columns=['price'])
    return df


def run_pool_pipeline(pool_id: str, events: dict) -> dict:
    """Run the full reconstruction pipeline for one pool.

    Args:
        pool_id: Pool identifier key in POOL_REGISTRY.
        events: Dict with keys 'mint', 'burn', 'collect', 'swap' (DataFrames).

    Returns:
        Dict with keys:
            positions         – list of closed position dicts (with capitals).
            raw_mint_df       – raw mint DataFrame (before any filtering).
            filtered_mint_df  – mint DataFrame after zero-value filter.
            burn_plus_stats   – result of validate_burn_plus().
    """
    meta = POOL_REGISTRY[pool_id]
    t0_dec = meta['token0_decimals']
    t1_dec = meta['token1_decimals']

    logger.info('[%s] Starting pool pipeline', pool_id)

    # ------------------------------------------------------------------
    # 1. Load and convert timestamps
    # ------------------------------------------------------------------
    mint_df = _to_unix_seconds(events['mint'].copy())
    burn_df = _to_unix_seconds(events['burn'].copy())
    collect_df = _to_unix_seconds(events['collect'].copy())
    swap_df = _to_unix_seconds(events['swap'].copy())

    raw_mint_df = mint_df.copy()

    # ------------------------------------------------------------------
    # 2. Standardize and sort liquidity events
    # ------------------------------------------------------------------
    mint_df = sort_events_chronologically(standardize_events(mint_df, pool_id))
    burn_df = sort_events_chronologically(standardize_events(burn_df, pool_id))
    collect_df = sort_events_chronologically(standardize_events(collect_df, pool_id))

    logger.info(
        '[%s] After standardize+sort: mint=%d  burn=%d  collect=%d',
        pool_id, len(mint_df), len(burn_df), len(collect_df),
    )

    # ------------------------------------------------------------------
    # 3. Filter zero-value events
    # ------------------------------------------------------------------
    mint_df = filter_zero_value_events(mint_df, 'mint')
    burn_df = filter_zero_value_events(burn_df, 'burn')
    collect_df = filter_zero_value_events(collect_df, 'collect')

    filtered_mint_df = mint_df.copy()
    logger.info(
        '[%s] After zero-value filter: mint=%d  burn=%d  collect=%d',
        pool_id, len(mint_df), len(burn_df), len(collect_df),
    )

    # ------------------------------------------------------------------
    # 4. Build price series from swap events
    #    The generator stores sqrtPriceX96; rename to match build_price_series.
    # ------------------------------------------------------------------
    if 'sqrtPriceX96' in swap_df.columns:
        swap_df = swap_df.rename(columns={'sqrtPriceX96': 'sqrt_price_x96'})

    try:
        price_series = build_price_series(swap_df, pool_id)
    except Exception as exc:
        logger.error('[%s] build_price_series failed: %s', pool_id, exc)
        return {
            'positions': [],
            'raw_mint_df': raw_mint_df,
            'filtered_mint_df': filtered_mint_df,
            'burn_plus_stats': {},
        }

    logger.info('[%s] Price series: %d observations', pool_id, len(price_series))

    # ------------------------------------------------------------------
    # 5. Attach prices to mint and burn events
    #    Drop pre-existing synthetic 'price' column to avoid merge conflict.
    # ------------------------------------------------------------------
    mint_df = attach_price_to_events(_drop_price_col(mint_df), price_series)
    burn_df = attach_price_to_events(_drop_price_col(burn_df), price_series)

    # ------------------------------------------------------------------
    # 6. Match burn + collect → burn_plus
    # ------------------------------------------------------------------
    burn_plus_df = match_burn_collect(burn_df, collect_df, pool_id)
    burn_plus_stats = validate_burn_plus(burn_plus_df)

    logger.info(
        '[%s] Burn+ match: %d/%d burns matched (%.1f%%)',
        pool_id,
        burn_plus_stats['matched'],
        burn_plus_stats['total_burns'],
        burn_plus_stats['matched_fraction'] * 100,
    )

    # ------------------------------------------------------------------
    # 7. Find eligible LPs (appear in both mints and burns)
    # ------------------------------------------------------------------
    eligible_lps = filter_lps_with_min_mint_burn(mint_df, burn_plus_df)
    logger.info('[%s] Eligible LPs: %d', pool_id, len(eligible_lps))

    # ------------------------------------------------------------------
    # 8. Build event streams per LP, reconstruct positions, compute capitals
    # ------------------------------------------------------------------
    all_positions = []

    for lp_addr in eligible_lps:
        try:
            stream = build_event_stream(mint_df, burn_plus_df, lp_addr, pool_id)
            if stream.empty:
                continue

            stream = filter_left_censored(stream)
            raw_positions = reconstruct_positions(stream)

            if raw_positions:
                raw_positions = compute_position_capitals(raw_positions, t0_dec, t1_dec)
                all_positions.extend(raw_positions)

        except Exception as exc:
            logger.warning('[%s] LP %s pipeline error: %s', pool_id, lp_addr[:10], exc)

    logger.info('[%s] Reconstructed %d closed positions', pool_id, len(all_positions))

    return {
        'positions': all_positions,
        'raw_mint_df': raw_mint_df,
        'filtered_mint_df': filtered_mint_df,
        'burn_plus_stats': burn_plus_stats,
    }


def make_plots(positions_df: pd.DataFrame, lp_pool_pnl: pd.DataFrame) -> None:
    """Create and save the three diagnostic plots for Experiment 1."""
    if positions_df.empty:
        logger.warning('positions_df is empty — skipping plots.')
        return

    # ------------------------------------------------------------------ (a)
    # Distribution of position-level PnL per pool
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, pool_id in zip(axes, POOL_IDS):
        subset = positions_df[positions_df['pool_id'] == pool_id]['realized_pnl'].dropna()
        if subset.empty:
            ax.set_title(f'{pool_id}\n(no data)')
            continue
        clipped = subset.clip(
            lower=subset.quantile(0.01), upper=subset.quantile(0.99)
        )
        ax.hist(clipped, bins=40, edgecolor='none', color='steelblue', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='PnL=0')
        ax.set_title(pool_id, fontsize=9)
        ax.set_xlabel('Realized PnL (USDC)', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle('Fig 1a — Position-Level Realized PnL Distribution by Pool', fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / 'exp1_position_pnl_distribution.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', out)

    # ------------------------------------------------------------------ (b)
    # LP-level total PnL distribution (all pools combined)
    # ------------------------------------------------------------------
    if lp_pool_pnl.empty:
        logger.warning('lp_pool_pnl is empty — skipping plot (b).')
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        pnl_vals = lp_pool_pnl['total_pnl'].dropna()
        clipped = pnl_vals.clip(
            lower=pnl_vals.quantile(0.02), upper=pnl_vals.quantile(0.98)
        )
        ax.hist(clipped, bins=50, edgecolor='none', color='teal', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.2, label='PnL=0')
        pct_pos = (pnl_vals > 0).mean() * 100
        ax.set_title(
            f'Fig 1b — LP-Pool Total PnL Distribution\n'
            f'(n={len(pnl_vals):,}  profitable={pct_pos:.1f}%)',
            fontsize=10,
        )
        ax.set_xlabel('Total Realized PnL (USDC)', fontsize=9)
        ax.set_ylabel('LP-Pool Count', fontsize=9)
        ax.legend()
        fig.tight_layout()
        out = RESULTS_DIR / 'exp1_lp_pool_pnl_distribution.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', out)

    # ------------------------------------------------------------------ (c)
    # Number of closed positions per LP (all pools)
    # ------------------------------------------------------------------
    pos_counts = (
        positions_df.groupby('lp_address')['realized_pnl']
        .count()
        .rename('n_positions')
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos_counts, bins=30, edgecolor='none', color='darkorange', alpha=0.8)
    ax.set_title(
        f'Fig 1c — Closed Positions per LP Wallet\n'
        f'(median={pos_counts.median():.0f}, mean={pos_counts.mean():.1f})',
        fontsize=10,
    )
    ax.set_xlabel('Number of Closed Positions', fontsize=9)
    ax.set_ylabel('Number of LP Wallets', fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / 'exp1_positions_per_lp.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', out)


def main() -> None:
    logger.info('=== Experiment 1: CLMM Event Reconstruction ===')

    # ------------------------------------------------------------------
    # Generate synthetic data
    # ------------------------------------------------------------------
    logger.info('Generating synthetic data for %d pools...', len(POOL_IDS))
    pool_data = generate_all_pool_data(n_lps=200, seed=42)

    # ------------------------------------------------------------------
    # Run pipeline for all pools; track counts at each stage
    # ------------------------------------------------------------------
    raw_mint_dfs: dict = {}
    filtered_mint_dfs: dict = {}
    used_position_dfs: dict = {}
    all_positions = []
    burn_plus_stats_all = {}

    for pool_id in POOL_IDS:
        print(f'\n--- Pool: {pool_id} ---')
        result = run_pool_pipeline(pool_id, pool_data[pool_id])

        raw_mint_dfs[pool_id] = result['raw_mint_df']
        filtered_mint_dfs[pool_id] = result['filtered_mint_df']
        burn_plus_stats_all[pool_id] = result['burn_plus_stats']
        all_positions.extend(result['positions'])

    # ------------------------------------------------------------------
    # Flatten positions to DataFrame and aggregate to LP-pool level
    # ------------------------------------------------------------------
    positions_df = positions_to_dataframe(all_positions)
    logger.info('Total reconstructed positions: %d', len(positions_df))

    if not positions_df.empty:
        lp_pool_pnl = aggregate_lp_pool_pnl(positions_df)
        for pool_id in POOL_IDS:
            used_position_dfs[pool_id] = (
                positions_df[positions_df['pool_id'] == pool_id]
                if 'pool_id' in positions_df.columns else pd.DataFrame()
            )
    else:
        lp_pool_pnl = pd.DataFrame(
            columns=['lp_address', 'pool_id', 'total_pnl', 'position_count',
                     'first_open_timestamp', 'last_close_timestamp']
        )
        used_position_dfs = {pid: pd.DataFrame() for pid in POOL_IDS}

    # ------------------------------------------------------------------
    # Validation counts at each pipeline stage
    # ------------------------------------------------------------------
    raw_counts = compute_validation_counts(raw_mint_dfs, 'raw')
    filtered_counts = compute_validation_counts(filtered_mint_dfs, 'filtered')
    used_counts = compute_validation_counts(used_position_dfs, 'used')

    validation_tables = {
        'raw': raw_counts,
        'filtered': filtered_counts,
        'used': used_counts,
        'burn_plus_stats': burn_plus_stats_all,
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if not positions_df.empty:
        positions_df.to_parquet(RESULTS_DIR / 'positions_reconstructed.parquet', index=False)
        logger.info('Saved positions_reconstructed.parquet (%d rows)', len(positions_df))
    else:
        logger.warning('positions_df is empty — parquet not saved')

    if not lp_pool_pnl.empty:
        lp_pool_pnl.to_parquet(RESULTS_DIR / 'lp_pool_pnl.parquet', index=False)
        logger.info('Saved lp_pool_pnl.parquet (%d rows)', len(lp_pool_pnl))

    with open(RESULTS_DIR / 'validation_tables.json', 'w') as fh:
        json.dump(validation_tables, fh, indent=2)
    logger.info('Saved validation_tables.json')

    # ------------------------------------------------------------------
    # Summary table: compare against paper targets
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('EXPERIMENT 1 — PIPELINE VALIDATION SUMMARY')
    print('=' * 70)
    print(f'{"Stage":<30} {"Synthetic":>12} {"Paper Target":>14}')
    print('-' * 70)
    print(
        f'{"Mint events (raw)":<30} '
        f'{raw_counts["total"]:>12,} '
        f'{"(no target)":>14}'
    )
    print(
        f'{"Mint events (after zero filter)":<30} '
        f'{filtered_counts["total"]:>12,} '
        f'{PAPER_AFTER_FILTER:>14,}'
    )
    print(
        f'{"Reconstructed positions":<30} '
        f'{used_counts["total"]:>12,} '
        f'{PAPER_AFTER_RECONSTRUCTION:>14,}'
    )
    print('-' * 70)
    print(
        '  NOTE: Synthetic data uses ~200 LPs per pool; '
        'paper uses the full Base-chain dataset.\n'
        '  Counts are expected to be far smaller with synthetic data.'
    )
    print()

    print(f'{"Pool":<30} {"Raw Mints":>10} {"Filt Mints":>12} {"Positions":>10}')
    print('-' * 65)
    for pid in POOL_IDS:
        print(
            f'{pid:<30} '
            f'{raw_counts["by_pool"].get(pid, 0):>10,} '
            f'{filtered_counts["by_pool"].get(pid, 0):>12,} '
            f'{used_counts["by_pool"].get(pid, 0):>10,}'
        )
    print()

    if not lp_pool_pnl.empty:
        pct_profit = (lp_pool_pnl['total_pnl'] > 0).mean() * 100
        print(f'LP-pool entities:    {len(lp_pool_pnl):,}')
        print(f'Profitable LP-pools: {(lp_pool_pnl["total_pnl"] > 0).sum():,}  ({pct_profit:.1f}%)')
        print(f'Mean PnL (USDC):     {lp_pool_pnl["total_pnl"].mean():.4f}')
        print(f'Median PnL (USDC):   {lp_pool_pnl["total_pnl"].median():.4f}')
    print()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        make_plots(positions_df, lp_pool_pnl)
        print('Plots saved to', RESULTS_DIR)
    except Exception as exc:
        logger.error('Plotting failed: %s', exc)

    logger.info('=== Experiment 1 complete ===')


if __name__ == '__main__':
    main()
