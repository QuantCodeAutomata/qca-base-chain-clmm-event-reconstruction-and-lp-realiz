"""Experiment 4: Cross-Pool LP Identity Aggregation.

Aggregates per-LP-pool records to the wallet level to study multi-pool
participation, pool intersection patterns, and success rate differences
between mono-LP and multi-LP wallets.

Run:
    python exp/exp_4_cross_pool.py
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

from src.cross_pool import (
    normalize_address, aggregate_cross_pool,
    pool_participation_distribution, compute_pool_intersections,
    multi_lp_success_distribution,
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

# Paper reference targets
PAPER_TOTAL_WALLETS = 28_474        # approx
PAPER_MULTI_LP_COUNT = 2_923        # multi-LP wallets
PAPER_MULTI_LP_FRACTION = 0.103     # ~10.3% of unique wallets


def _load_or_regenerate() -> pd.DataFrame:
    """Load lp_win_score from results/, regenerating if necessary.

    Returns:
        lp_scores_df with columns: lp_address, pool_id, total_pnl,
        position_count, first_open_timestamp, last_close_timestamp,
        omega, profitable_flag, consistent_flag.
    """
    score_path = RESULTS_DIR / 'lp_win_score.parquet'
    if score_path.exists():
        logger.info('Loading lp_win_score.parquet...')
        df = pd.read_parquet(score_path)
        logger.info('Loaded %d LP-pool rows.', len(df))
        return df

    logger.info('lp_win_score.parquet not found — regenerating...')
    from exp.exp_1_reconstruction import run_pool_pipeline
    from exp.exp_2_win_score import _load_or_regenerate as load_exp2
    from data.generate_synthetic_data import generate_all_pool_data
    from src.pnl_calculator import positions_to_dataframe, aggregate_lp_pool_pnl
    from src.win_score import classify_lps

    # Try loading positions first
    pos_path = RESULTS_DIR / 'positions_reconstructed.parquet'
    pnl_path = RESULTS_DIR / 'lp_pool_pnl.parquet'

    if pos_path.exists() and pnl_path.exists():
        positions_df = pd.read_parquet(pos_path)
        lp_pool_pnl = pd.read_parquet(pnl_path)
    else:
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

    if lp_pool_pnl.empty or positions_df.empty:
        raise RuntimeError('No LP-pool data available for cross-pool analysis.')

    lp_scores_df = classify_lps(lp_pool_pnl, positions_df)
    lp_scores_df.to_parquet(score_path, index=False)
    return lp_scores_df


def _build_intersection_matrix(
    intersections_df: pd.DataFrame,
    pool_ids: list,
) -> pd.DataFrame:
    """Build a square pool × pool wallet-count matrix from intersection results.

    Fills the off-diagonal with the count of wallets that participated in
    both pools (pairwise subset of size 2).  The diagonal holds per-pool
    totals (subset of size 1).

    Args:
        intersections_df: Output of compute_pool_intersections().
        pool_ids: Ordered list of pool identifiers.

    Returns:
        DataFrame (index=pool_ids, columns=pool_ids) with integer counts.
    """
    n = len(pool_ids)
    matrix = pd.DataFrame(
        np.zeros((n, n), dtype=int),
        index=pool_ids,
        columns=pool_ids,
    )

    for _, row in intersections_df.iterrows():
        subset_repr = str(row['pool_subset'])
        # Check which pool_ids are mentioned in the subset repr
        matched = [p for p in pool_ids if p in subset_repr]
        count = int(row['wallet_count'])
        size = int(row['subset_size'])

        if size == 1 and len(matched) == 1:
            p = matched[0]
            matrix.at[p, p] = count
        elif size == 2 and len(matched) == 2:
            p1, p2 = matched[0], matched[1]
            matrix.at[p1, p2] = count
            matrix.at[p2, p1] = count

    return matrix


def make_plots(
    cross_pool_df: pd.DataFrame,
    participation_dist: pd.DataFrame,
    intersection_matrix: pd.DataFrame,
    lp_scores_df: pd.DataFrame,
) -> None:
    """Create and save the three diagnostic plots for Experiment 4."""
    # ------------------------------------------------------------------ (a)
    # Bar chart: wallet participation distribution (1, 2, 3, 4 pools)
    # ------------------------------------------------------------------
    if participation_dist.empty:
        logger.warning('participation_dist empty — skipping plot (a).')
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(
            participation_dist['num_pools'].astype(str),
            participation_dist['wallet_count'],
            color='steelblue', edgecolor='none', alpha=0.85,
        )
        ax.set_xlabel('Number of Pools Participated In', fontsize=9)
        ax.set_ylabel('Number of Wallets', fontsize=9)
        ax.set_title(
            'Fig 4a — Multi-LP Participation Distribution\n'
            '(Paper: ~10% of wallets participate in ≥2 pools)',
            fontsize=9,
        )
        total = participation_dist['wallet_count'].sum()
        for _, row in participation_dist.iterrows():
            cnt = int(row['wallet_count'])
            pct = cnt / total * 100
            ax.text(
                str(int(row['num_pools'])), cnt + total * 0.005,
                f'{pct:.1f}%', ha='center', fontsize=8,
            )
        fig.tight_layout()
        out = RESULTS_DIR / 'exp4_participation_distribution.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', out)

    # ------------------------------------------------------------------ (b)
    # Heatmap: pool intersection matrix
    # ------------------------------------------------------------------
    if not intersection_matrix.empty and intersection_matrix.values.sum() > 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        vals = intersection_matrix.values.astype(float)
        # Mask zeros for cleaner display
        masked = np.where(vals == 0, np.nan, vals)
        im = ax.imshow(masked, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, label='Wallet count')

        short_labels = [p.replace('_', '\n') for p in intersection_matrix.index]
        ax.set_xticks(range(len(short_labels)))
        ax.set_yticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = int(vals[i, j])
                if v > 0:
                    ax.text(j, i, str(v), ha='center', va='center',
                            fontsize=8, color='black')

        ax.set_title('Fig 4b — Pool Intersection Matrix\n(wallet count in row ∩ col)', fontsize=9)
        fig.tight_layout()
        out = RESULTS_DIR / 'exp4_pool_intersection_heatmap.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', out)

    # ------------------------------------------------------------------ (c)
    # Bar chart: success rates mono vs multi LP
    # ------------------------------------------------------------------
    if cross_pool_df.empty:
        logger.warning('cross_pool_df empty — skipping plot (c).')
        return

    mono = cross_pool_df[cross_pool_df['lp_type'] == 'mono_lp']
    multi = cross_pool_df[cross_pool_df['lp_type'] == 'multi_lp']

    # Re-derive profitable/consistent from lp_scores_df per wallet
    wallet_flags = (
        lp_scores_df
        .assign(wallet=lp_scores_df['lp_address'].apply(normalize_address))
        .groupby('wallet')
        .agg(
            any_profitable=('profitable_flag', 'any'),
            any_consistent=('consistent_flag', 'any'),
        )
        .reset_index()
    )

    mono_wallets = set(mono['wallet'])
    multi_wallets = set(multi['wallet'])

    def _rate(wallets_set, flag_col):
        sub = wallet_flags[wallet_flags['wallet'].isin(wallets_set)]
        if sub.empty:
            return 0.0
        return sub[flag_col].mean() * 100

    stats = {
        'Mono-LP': {
            'profitable': _rate(mono_wallets, 'any_profitable'),
            'consistent': _rate(mono_wallets, 'any_consistent'),
            'n': len(mono),
        },
        'Multi-LP': {
            'profitable': _rate(multi_wallets, 'any_profitable'),
            'consistent': _rate(multi_wallets, 'any_consistent'),
            'n': len(multi),
        },
    }

    categories = list(stats.keys())
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        x - width / 2,
        [stats[c]['profitable'] for c in categories],
        width, label='Any profitable pool', color='steelblue', alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        [stats[c]['consistent'] for c in categories],
        width, label='Any consistent pool', color='darkorange', alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{c}\n(n={stats[c]["n"]:,})' for c in categories],
        fontsize=9,
    )
    ax.set_ylabel('% of Wallets', fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title('Fig 4c — Success Rates: Mono-LP vs Multi-LP Wallets', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / 'exp4_mono_vs_multi_success.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', out)


def main() -> None:
    logger.info('=== Experiment 4: Cross-Pool LP Identity Aggregation ===')

    # ------------------------------------------------------------------
    # Load or regenerate data
    # ------------------------------------------------------------------
    try:
        lp_scores_df = _load_or_regenerate()
    except Exception as exc:
        logger.error('Failed to load/regenerate data: %s', exc)
        raise

    if lp_scores_df.empty:
        logger.error('No LP-pool data available.')
        return

    # ------------------------------------------------------------------
    # Aggregate to wallet level
    # ------------------------------------------------------------------
    logger.info('Aggregating %d LP-pool rows to wallet level...', len(lp_scores_df))
    cross_pool_df = aggregate_cross_pool(lp_scores_df)

    n_total = len(cross_pool_df)
    n_mono = (cross_pool_df['lp_type'] == 'mono_lp').sum()
    n_multi = (cross_pool_df['lp_type'] == 'multi_lp').sum()

    # ------------------------------------------------------------------
    # Participation and intersection distributions
    # ------------------------------------------------------------------
    participation_dist = pool_participation_distribution(cross_pool_df)
    intersections_df = compute_pool_intersections(lp_scores_df, POOL_IDS)
    multi_success_dist = multi_lp_success_distribution(cross_pool_df)
    intersection_matrix = _build_intersection_matrix(intersections_df, POOL_IDS)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    cross_pool_out = RESULTS_DIR / 'wallet_cross_pool_summary.parquet'
    cross_pool_df.to_parquet(cross_pool_out, index=False)
    logger.info('Saved wallet_cross_pool_summary.parquet (%d rows)', len(cross_pool_df))

    matrix_out = RESULTS_DIR / 'pool_intersection_matrix.csv'
    intersection_matrix.to_csv(matrix_out)
    logger.info('Saved pool_intersection_matrix.csv')

    # ------------------------------------------------------------------
    # Print top-level summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('EXPERIMENT 4 — CROSS-POOL LP IDENTITY SUMMARY')
    print('=' * 70)
    print(f'Total unique wallets:       {n_total:>8,}   (paper: ~{PAPER_TOTAL_WALLETS:,})')
    print(f'Mono-LP wallets (1 pool):   {n_mono:>8,}   ({n_mono/n_total*100:.1f}%)')
    print(f'Multi-LP wallets (≥2 pools):{n_multi:>8,}   ({n_multi/n_total*100:.1f}%)')
    print(
        f'Multi-LP fraction:          {n_multi/n_total*100:>7.1f}%   '
        f'(paper: ~{PAPER_MULTI_LP_FRACTION*100:.1f}%,  {PAPER_MULTI_LP_COUNT:,} wallets)'
    )
    print()

    # ------------------------------------------------------------------
    # Table 6-style: wallets by number of pools
    # ------------------------------------------------------------------
    print('Participation distribution:')
    print(f'  {"Pools":>6} {"Wallets":>9} {"Share":>8}')
    print('  ' + '-' * 27)
    for _, row in participation_dist.iterrows():
        np_ = int(row['num_pools'])
        cnt = int(row['wallet_count'])
        pct = cnt / n_total * 100
        print(f'  {np_:>6} {cnt:>9,} {pct:>7.1f}%')
    print()

    # Multi-LP consistent-pools distribution
    if not multi_success_dist.empty:
        print('Multi-LP wallets — consistently successful pool count:')
        print(f'  {"Consistent Pools":>18} {"Wallets":>9} {"Share":>8}')
        print('  ' + '-' * 38)
        n_multi_total = multi_success_dist['wallet_count'].sum()
        for _, row in multi_success_dist.iterrows():
            cp = int(row['consistent_pools'])
            cnt = int(row['wallet_count'])
            pct = cnt / max(n_multi_total, 1) * 100
            print(f'  {cp:>18} {cnt:>9,} {pct:>7.1f}%')
        print()

    # ------------------------------------------------------------------
    # Pool intersection summary (pairs)
    # ------------------------------------------------------------------
    pair_intersections = intersections_df[intersections_df['subset_size'] == 2].copy()
    if not pair_intersections.empty:
        print('Pool pairwise intersections (wallets in BOTH pools):')
        print(f'  {"Pool Pair":<50} {"Wallets":>9}')
        print('  ' + '-' * 62)
        for _, row in pair_intersections.iterrows():
            print(f'  {row["pool_subset"]:<50} {int(row["wallet_count"]):>9,}')
        print()

    # Intersection matrix
    print('Pool intersection matrix (wallets participating in both row and col pool):')
    short_ids = [p.replace('_', '-') for p in POOL_IDS]
    header_line = f'  {"":>22}' + ''.join(f'{s:>10}' for s in short_ids)
    print(header_line)
    print('  ' + '-' * (22 + 10 * len(POOL_IDS)))
    for pid in POOL_IDS:
        row_label = pid.replace('_', '-')
        vals = ''.join(f'{int(intersection_matrix.at[pid, c]):>10,}' for c in POOL_IDS)
        print(f'  {row_label:>22}{vals}')
    print()

    print(
        'NOTE: Synthetic data generates overlapping LP addresses across pools,\n'
        f'      so multi-LP counts will differ from the paper target of '
        f'{PAPER_MULTI_LP_COUNT:,} multi-LP wallets.\n'
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        make_plots(
            cross_pool_df, participation_dist,
            intersection_matrix, lp_scores_df,
        )
        print('Plots saved to', RESULTS_DIR)
    except Exception as exc:
        logger.error('Plotting failed: %s', exc)

    logger.info('=== Experiment 4 complete ===')


if __name__ == '__main__':
    main()
