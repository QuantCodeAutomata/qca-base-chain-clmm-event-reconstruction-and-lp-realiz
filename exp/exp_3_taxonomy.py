"""Experiment 3: Position Taxonomy — 15-Type Classification.

Classifies reconstructed CLMM positions into 15 types based on how the pool
price moved relative to each position's tick range between open and close.

Run:
    python exp/exp_3_taxonomy.py
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
from src.taxonomy import (
    compute_position_metrics, assign_types_to_positions,
    aggregate_by_type, width_bucket,
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

# All 4 pools share the same token decimals (POOL_REGISTRY token0=USDC, token1=WETH)
TOKEN0_DECIMALS = 6    # USDC
TOKEN1_DECIMALS = 18   # WETH

# Paper Table 7 reference: type share targets (approximate)
PAPER_TYPE_SHARES = {
    3:  {'txn_share': 0.002, 'description': 'Open below, close inside'},
    5:  {'txn_share': 0.458, 'description': 'Both inside, price up (dominant type)'},
    6:  {'txn_share': 0.450, 'description': 'Both inside, price down (dominant type)'},
    7:  {'txn_share': 0.004, 'description': 'Open inside, close above'},
}

# Type descriptions for table output
TYPE_DESCRIPTIONS = {
    1:  'Both below range, price down',
    2:  'Close below, open at/above a',
    3:  'Open below, close inside',
    4:  'Open below, close above (full traverse ↑)',
    5:  'Both inside, price up/flat',
    6:  'Both inside, price down/flat',
    7:  'Open inside, close above',
    8:  'Both above range, any direction',
    9:  'Start above, end inside',
    10: 'Full traversal downward',
    11: 'At/below lower — placeholder',
    12: 'Open at/below lower, close inside',
    13: 'Both at/above upper',
    14: 'Open inside, close at/above upper',
    15: 'Both strictly below, price went up',
    -1: 'Unclassified (fallback)',
}


def _load_or_regenerate() -> pd.DataFrame:
    """Load positions_reconstructed from results/, or regenerate if missing.

    Returns:
        positions_df with opening_capital and realized_pnl columns present.
    """
    pos_path = RESULTS_DIR / 'positions_reconstructed.parquet'
    if pos_path.exists():
        logger.info('Loading positions_reconstructed.parquet...')
        df = pd.read_parquet(pos_path)
        logger.info('Loaded %d positions.', len(df))
        return df

    logger.info('positions_reconstructed.parquet not found — regenerating...')
    from exp.exp_1_reconstruction import run_pool_pipeline
    from data.generate_synthetic_data import generate_all_pool_data

    pool_data = generate_all_pool_data(n_lps=200, seed=42)
    all_positions = []
    for pool_id in POOL_IDS:
        result = run_pool_pipeline(pool_id, pool_data[pool_id])
        all_positions.extend(result['positions'])

    positions_df = positions_to_dataframe(all_positions)
    if not positions_df.empty:
        positions_df.to_parquet(pos_path, index=False)
    return positions_df


def make_plots(
    typed_df: pd.DataFrame,
    type_summary: pd.DataFrame,
) -> None:
    """Create and save the three diagnostic plots for Experiment 3."""
    if typed_df.empty or type_summary.empty:
        logger.warning('Typed positions or summary empty — skipping plots.')
        return

    colours = plt.cm.tab20.colors  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ (a)
    # Bar chart: position type distribution by transaction count
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    valid_types = type_summary[type_summary['position_type'] != -1].copy()
    x = np.arange(len(valid_types))
    bars = ax.bar(
        x,
        valid_types['transaction_share'] * 100,
        color=[colours[i % len(colours)] for i in range(len(valid_types))],
        edgecolor='none', alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{int(t)}' for t in valid_types['position_type']], fontsize=8)
    ax.set_xlabel('Position Type', fontsize=9)
    ax.set_ylabel('Transaction Share (%)', fontsize=9)
    ax.set_title(
        'Fig 3a — Position Type Distribution by Transaction Count\n'
        '(Types 5 and 6 dominate in paper: ~45.8% and ~45.0%)',
        fontsize=9,
    )
    fig.tight_layout()
    out = RESULTS_DIR / 'exp3_type_distribution_txn.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', out)

    # ------------------------------------------------------------------ (b)
    # Bar chart: position type distribution by mint capital
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x,
        valid_types['mint_capital_share'] * 100,
        color=[colours[i % len(colours)] for i in range(len(valid_types))],
        edgecolor='none', alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{int(t)}' for t in valid_types['position_type']], fontsize=8)
    ax.set_xlabel('Position Type', fontsize=9)
    ax.set_ylabel('Mint Capital Share (%)', fontsize=9)
    ax.set_title('Fig 3b — Position Type Distribution by Mint Capital', fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / 'exp3_type_distribution_capital.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', out)

    # ------------------------------------------------------------------ (c)
    # Delta distribution for Type 5 positions
    # ------------------------------------------------------------------
    type5 = typed_df[typed_df['position_type'] == 5]
    if type5.empty:
        logger.warning('No Type 5 positions found — skipping delta plot.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    delta_start = type5['delta_start'].dropna()
    delta_end = type5['delta_end'].dropna()

    axes[0].hist(delta_start, bins=30, color='steelblue', edgecolor='none', alpha=0.8)
    axes[0].set_title('Type 5: δ_start (open price position in range)', fontsize=9)
    axes[0].set_xlabel('δ_start = (P_start − a) / (b − a)', fontsize=8)
    axes[0].set_ylabel('Count', fontsize=8)
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=1, label='centre')
    axes[0].legend(fontsize=7)

    axes[1].hist(delta_end, bins=30, color='darkorange', edgecolor='none', alpha=0.8)
    axes[1].set_title('Type 5: δ_end (close price position in range)', fontsize=9)
    axes[1].set_xlabel('δ_end = (P_end − a) / (b − a)', fontsize=8)
    axes[1].set_ylabel('Count', fontsize=8)
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=1, label='centre')
    axes[1].legend(fontsize=7)

    fig.suptitle(
        f'Fig 3c — Delta (δ) Distribution for Type 5 Positions (n={len(type5):,})',
        fontsize=10,
    )
    fig.tight_layout()
    out = RESULTS_DIR / 'exp3_type5_delta_distribution.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', out)


def main() -> None:
    logger.info('=== Experiment 3: Position Taxonomy (15 Types) ===')

    # ------------------------------------------------------------------
    # Load or regenerate positions
    # ------------------------------------------------------------------
    try:
        positions_df = _load_or_regenerate()
    except Exception as exc:
        logger.error('Failed to load/regenerate positions: %s', exc)
        raise

    if positions_df.empty:
        logger.error('No positions available — cannot compute taxonomy.')
        return

    # ------------------------------------------------------------------
    # Compute geometric and capital metrics
    # ------------------------------------------------------------------
    logger.info('Computing position metrics for %d positions...', len(positions_df))
    try:
        metrics_df = compute_position_metrics(
            positions_df,
            token0_decimals=TOKEN0_DECIMALS,
            token1_decimals=TOKEN1_DECIMALS,
        )
    except Exception as exc:
        logger.error('compute_position_metrics failed: %s', exc)
        raise

    # ------------------------------------------------------------------
    # Assign position types (1–15)
    # ------------------------------------------------------------------
    logger.info('Assigning position types...')
    try:
        typed_df = assign_types_to_positions(metrics_df)
    except Exception as exc:
        logger.error('assign_types_to_positions failed: %s', exc)
        raise

    type_counts = typed_df['position_type'].value_counts().sort_index()
    logger.info('Type distribution:\n%s', type_counts.to_string())

    # ------------------------------------------------------------------
    # Aggregate by type
    # ------------------------------------------------------------------
    type_summary = aggregate_by_type(typed_df)

    # ------------------------------------------------------------------
    # Width bucket analysis for Type 5
    # ------------------------------------------------------------------
    type5_df = typed_df[typed_df['position_type'] == 5].copy()
    if not type5_df.empty and 'width_usdc' in type5_df.columns:
        type5_df['width_bucket'] = type5_df['width_usdc'].apply(width_bucket)
        width_dist = (
            type5_df['width_bucket']
            .value_counts()
            .rename_axis('bucket')
            .reset_index(name='count')
        )
        width_dist['share'] = width_dist['count'] / width_dist['count'].sum()
        width_dist = width_dist.sort_values('bucket')
    else:
        width_dist = pd.DataFrame(columns=['bucket', 'count', 'share'])

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    typed_out = RESULTS_DIR / 'positions_typed.parquet'
    typed_df.to_parquet(typed_out, index=False)
    logger.info('Saved positions_typed.parquet (%d rows)', len(typed_df))

    summary_out = RESULTS_DIR / 'type_summary_table.csv'
    type_summary.to_csv(summary_out, index=False)
    logger.info('Saved type_summary_table.csv')

    # ------------------------------------------------------------------
    # Print Table 7-style summary
    # ------------------------------------------------------------------
    total_pos = len(typed_df)
    print('\n' + '=' * 85)
    print('EXPERIMENT 3 — TABLE 7 STYLE: POSITION TYPE SUMMARY')
    print('=' * 85)
    print(
        f'{"Type":>5} {"Count":>8} {"Txn Share":>10} {"Cap Share":>10}'
        f'  Description'
    )
    print('-' * 85)

    for _, row in type_summary.iterrows():
        t = int(row['position_type'])
        desc = TYPE_DESCRIPTIONS.get(t, '')
        print(
            f'{t:>5} '
            f'{int(row["count"]):>8,} '
            f'{row["transaction_share"]*100:>9.2f}% '
            f'{row["mint_capital_share"]*100:>9.2f}%'
            f'  {desc}'
        )

    print('-' * 85)
    print(f'{"TOTAL":>5} {total_pos:>8,}   100.00%   100.00%')
    print()

    # Highlight key types
    print('Key types detail (paper targets):')
    print(f'  {"Type":>5} {"Synthetic %":>12} {"Paper %":>12}  Notes')
    print('  ' + '-' * 60)
    for t_id in [3, 5, 6, 7]:
        row = type_summary[type_summary['position_type'] == t_id]
        synth_pct = row['transaction_share'].values[0] * 100 if len(row) > 0 else 0.0
        paper_ref = PAPER_TYPE_SHARES.get(t_id, {})
        paper_pct = paper_ref.get('txn_share', 0.0) * 100
        note = paper_ref.get('description', '')
        print(
            f'  {t_id:>5} {synth_pct:>11.2f}% {paper_pct:>11.2f}%  {note}'
        )
    print()

    # Width bucket summary for Type 5
    if not width_dist.empty:
        print(f'Type 5 width-bucket distribution (n={len(type5_df):,}):')
        print(f'  {"Bucket":<16} {"Count":>8} {"Share":>8}')
        print('  ' + '-' * 35)
        for _, wr in width_dist.iterrows():
            print(f'  {wr["bucket"]:<16} {int(wr["count"]):>8,} {wr["share"]*100:>7.1f}%')
        print()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        make_plots(typed_df, type_summary)
        print('Plots saved to', RESULTS_DIR)
    except Exception as exc:
        logger.error('Plotting failed: %s', exc)

    logger.info('=== Experiment 3 complete ===')


if __name__ == '__main__':
    main()
