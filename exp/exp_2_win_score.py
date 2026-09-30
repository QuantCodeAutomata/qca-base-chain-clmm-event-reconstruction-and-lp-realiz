"""Experiment 2: Win-Score (Omega) Metric and LP Classification.

Computes the omega (Ω) win-score for every LP–pool entity, classifies LPs as
profitable / consistently successful, and aggregates results by pool.

Run:
    python exp/exp_2_win_score.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.event_schema import POOL_REGISTRY
from src.pnl_calculator import positions_to_dataframe, aggregate_lp_pool_pnl
from src.win_score import (
    compute_cumulative_pnl_path, compute_win_score,
    classify_lps, aggregate_success_by_pool,
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

# Paper Table 4 reference values for comparison
PAPER_TABLE4 = {
    'uniswap_v3_005':     {'n_lps': None, 'pct_profitable': 46.8, 'pct_consistent': 16.4},
    'aerodrome_ss_005':   {'n_lps': None, 'pct_profitable': 44.6, 'pct_consistent': 14.3},
    'pancakeswap_v3_001': {'n_lps': None, 'pct_profitable': 42.4, 'pct_consistent': 13.0},
    'sushiswap_v3_001':   {'n_lps': None, 'pct_profitable': 41.8, 'pct_consistent': 12.1},
}


def _load_or_regenerate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load positions and lp_pool_pnl from results/, or regenerate from synthetic data.

    Returns:
        Tuple (positions_df, lp_pool_pnl_df).
    """
    pos_path = RESULTS_DIR / 'positions_reconstructed.parquet'
    pnl_path = RESULTS_DIR / 'lp_pool_pnl.parquet'

    if pos_path.exists() and pnl_path.exists():
        logger.info('Loading from existing parquet files...')
        positions_df = pd.read_parquet(pos_path)
        lp_pool_pnl = pd.read_parquet(pnl_path)
        logger.info(
            'Loaded %d positions and %d LP-pool rows.',
            len(positions_df), len(lp_pool_pnl),
        )
        return positions_df, lp_pool_pnl

    logger.info('Parquet files not found — regenerating from synthetic data...')
    from exp.exp_1_reconstruction import run_pool_pipeline
    from data.generate_synthetic_data import generate_all_pool_data

    pool_data = generate_all_pool_data(n_lps=200, seed=42)
    all_positions = []
    for pool_id in POOL_IDS:
        result = run_pool_pipeline(pool_id, pool_data[pool_id])
        all_positions.extend(result['positions'])

    positions_df = positions_to_dataframe(all_positions)
    lp_pool_pnl = aggregate_lp_pool_pnl(positions_df)

    if not positions_df.empty:
        positions_df.to_parquet(pos_path, index=False)
        lp_pool_pnl.to_parquet(pnl_path, index=False)

    return positions_df, lp_pool_pnl


def make_plots(lp_scores_df: pd.DataFrame, pool_summary: pd.DataFrame) -> None:
    """Create and save the three diagnostic plots for Experiment 2."""
    if lp_scores_df.empty:
        logger.warning('lp_scores_df empty — skipping plots.')
        return

    # ------------------------------------------------------------------ (a)
    # Distribution of omega across all LP-pool entities
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    omega_vals = lp_scores_df['omega'].dropna()
    ax.hist(omega_vals, bins=40, edgecolor='none', color='mediumslateblue', alpha=0.85)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2, label='Ω=0.5 (neutral)')
    ax.set_title(
        f'Fig 2a — Omega (Ω) Win-Score Distribution\n'
        f'(n={len(omega_vals):,}  mean={omega_vals.mean():.3f}  median={omega_vals.median():.3f})',
        fontsize=10,
    )
    ax.set_xlabel('Omega (Ω)', fontsize=9)
    ax.set_ylabel('LP-Pool Count', fontsize=9)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / 'exp2_omega_distribution.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', out)

    # ------------------------------------------------------------------ (b)
    # Scatter: total PnL vs omega, coloured by pool
    # ------------------------------------------------------------------
    colours = {
        'uniswap_v3_005':     '#1f77b4',
        'aerodrome_ss_005':   '#ff7f0e',
        'pancakeswap_v3_001': '#2ca02c',
        'sushiswap_v3_001':   '#d62728',
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    for pool_id, grp in lp_scores_df.groupby('pool_id'):
        pnl = grp['total_pnl'].clip(
            lower=grp['total_pnl'].quantile(0.02),
            upper=grp['total_pnl'].quantile(0.98),
        )
        ax.scatter(
            grp['omega'], pnl,
            s=12, alpha=0.5, label=pool_id,
            color=colours.get(pool_id, 'grey'),
        )
    ax.axvline(0.5, color='red', linestyle='--', linewidth=0.8)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title('Fig 2b — Total PnL vs Omega (Ω) by Pool', fontsize=10)
    ax.set_xlabel('Omega (Ω)', fontsize=9)
    ax.set_ylabel('Total Realized PnL (USDC)', fontsize=9)
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    out = RESULTS_DIR / 'exp2_pnl_vs_omega_scatter.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', out)

    # ------------------------------------------------------------------ (c)
    # Bar chart: success rates by pool
    # ------------------------------------------------------------------
    if pool_summary.empty:
        logger.warning('pool_summary empty — skipping bar chart.')
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pool_summary))
    width = 0.35
    ax.bar(
        x - width / 2,
        pool_summary['profitable_fraction'] * 100,
        width, label='Profitable (PnL>0)', color='steelblue', alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        pool_summary['consistent_fraction'] * 100,
        width, label='Consistent (PnL>0 & Ω>0.5)', color='darkorange', alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [p.replace('_', '\n') for p in pool_summary['pool_id']],
        fontsize=8,
    )
    ax.set_ylabel('% of LP-Pool Entities', fontsize=9)
    ax.set_title('Fig 2c — Profitability and Consistency Rates by Pool', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out = RESULTS_DIR / 'exp2_success_rates_by_pool.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', out)


def main() -> None:
    logger.info('=== Experiment 2: Win-Score (Omega) Metric ===')

    # ------------------------------------------------------------------
    # Load or regenerate data
    # ------------------------------------------------------------------
    try:
        positions_df, lp_pool_pnl = _load_or_regenerate()
    except Exception as exc:
        logger.error('Failed to load/regenerate data: %s', exc)
        raise

    if positions_df.empty or lp_pool_pnl.empty:
        logger.error('No positions available — cannot compute win scores.')
        return

    # ------------------------------------------------------------------
    # Compute omega and classify LPs
    # ------------------------------------------------------------------
    logger.info('Computing omega win-scores for %d LP-pool entities...', len(lp_pool_pnl))
    try:
        lp_scores_df = classify_lps(lp_pool_pnl, positions_df)
    except Exception as exc:
        logger.error('classify_lps failed: %s', exc)
        raise

    logger.info(
        'Classification complete: %d profitable, %d consistent out of %d total.',
        lp_scores_df['profitable_flag'].sum(),
        lp_scores_df['consistent_flag'].sum(),
        len(lp_scores_df),
    )

    # ------------------------------------------------------------------
    # Pool-level aggregation
    # ------------------------------------------------------------------
    pool_summary = aggregate_success_by_pool(lp_scores_df)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    lp_scores_df.to_parquet(RESULTS_DIR / 'lp_win_score.parquet', index=False)
    logger.info('Saved lp_win_score.parquet (%d rows)', len(lp_scores_df))

    # ------------------------------------------------------------------
    # Print Table 4-style summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 80)
    print('EXPERIMENT 2 — TABLE 4 STYLE: LP SUCCESS RATES BY POOL')
    print('=' * 80)
    header = (
        f'{"Pool":<28} {"N LPs":>7} {"Profitable":>11} '
        f'{"Consistent":>11} {"% Profit":>9} {"% Consist":>10}'
    )
    print(header)
    print('-' * 80)

    totals = {
        'n_lps': 0, 'profitable': 0, 'consistent': 0,
    }
    for _, row in pool_summary.iterrows():
        pid = row['pool_id']
        print(
            f'{pid:<28} '
            f'{int(row["total_lps"]):>7,} '
            f'{int(row["profitable_lps"]):>11,} '
            f'{int(row["consistent_lps"]):>11,} '
            f'{row["profitable_fraction"]*100:>8.1f}% '
            f'{row["consistent_fraction"]*100:>9.1f}%'
        )
        totals['n_lps'] += int(row['total_lps'])
        totals['profitable'] += int(row['profitable_lps'])
        totals['consistent'] += int(row['consistent_lps'])

    print('-' * 80)
    tot_pct_p = totals['profitable'] / max(totals['n_lps'], 1) * 100
    tot_pct_c = totals['consistent'] / max(totals['n_lps'], 1) * 100
    print(
        f'{"TOTAL (all pools)":<28} '
        f'{totals["n_lps"]:>7,} '
        f'{totals["profitable"]:>11,} '
        f'{totals["consistent"]:>11,} '
        f'{tot_pct_p:>8.1f}% '
        f'{tot_pct_c:>9.1f}%'
    )

    print('\nPaper Table 4 reference (synthetic values will differ):')
    print(f'  {"Pool":<28} {"% Profit (paper)":>18} {"% Consist (paper)":>19}')
    print('  ' + '-' * 68)
    for pid, ref in PAPER_TABLE4.items():
        print(
            f'  {pid:<28} '
            f'{ref["pct_profitable"]:>17.1f}% '
            f'{ref["pct_consistent"]:>18.1f}%'
        )
    print()

    # ------------------------------------------------------------------
    # Omega summary statistics
    # ------------------------------------------------------------------
    omega = lp_scores_df['omega']
    print(f'Omega (Ω) statistics across all LP-pool entities (n={len(omega):,}):')
    print(f'  Mean:    {omega.mean():.4f}')
    print(f'  Median:  {omega.median():.4f}')
    print(f'  Std:     {omega.std():.4f}')
    print(f'  >0.5:    {(omega > 0.5).sum():,}  ({(omega > 0.5).mean()*100:.1f}%)')
    print(f'  =0.5:    {(omega == 0.5).sum():,}  ({(omega == 0.5).mean()*100:.1f}%)')
    print(f'  <0.5:    {(omega < 0.5).sum():,}  ({(omega < 0.5).mean()*100:.1f}%)')
    print()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        make_plots(lp_scores_df, pool_summary)
        print('Plots saved to', RESULTS_DIR)
    except Exception as exc:
        logger.error('Plotting failed: %s', exc)

    logger.info('=== Experiment 2 complete ===')


if __name__ == '__main__':
    main()
