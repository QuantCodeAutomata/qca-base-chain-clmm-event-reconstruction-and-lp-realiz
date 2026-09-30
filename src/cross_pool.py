"""Cross-pool LP identity aggregation.

CLMM liquidity providers often participate in multiple pools simultaneously.
This module aggregates the per-LP-pool records produced by
:mod:`src.win_score` to a *wallet* level so that behaviour across pools can
be studied.

Key questions answered:

* Which wallets are mono-LP (single pool) vs multi-LP (multiple pools)?
* How many pools does each multi-LP wallet participate in?
* What fraction of multi-LP wallets are consistently successful in at least
  one / more than one pool?
* Which pool pairs / combinations overlap most?
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_address(addr: str) -> str:
    """Return a normalised (lowercase, stripped) Ethereum address string.

    Args:
        addr: Raw address string (may contain leading/trailing whitespace
              and mixed-case hex characters).

    Returns:
        Lowercase, whitespace-stripped address string.
    """
    return str(addr).strip().lower()


def aggregate_cross_pool(lp_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LP–pool records to the wallet (address) level.

    Args:
        lp_scores_df: Output of :func:`~src.win_score.classify_lps` or
            equivalent DataFrame with columns ``lp_address``, ``pool_id``,
            ``profitable_flag``, ``consistent_flag``.

    Returns:
        Wallet-level DataFrame with columns:

        ``wallet`` (normalised address),
        ``num_pools`` (number of distinct pool_ids),
        ``lp_type`` (``'mono_lp'`` if num_pools == 1, else ``'multi_lp'``),
        ``profitable_pools`` (count of pools where profitable_flag is True),
        ``consistent_pools`` (count of pools where consistent_flag is True),
        ``pool_ids`` (sorted comma-separated string of participated pool_ids).

        Sorted by ``wallet`` ascending.
    """
    df = lp_scores_df.copy()
    df["wallet"] = df["lp_address"].apply(normalize_address)

    wallet_agg = (
        df.groupby("wallet")
        .agg(
            num_pools=("pool_id", "nunique"),
            profitable_pools=("profitable_flag", "sum"),
            consistent_pools=("consistent_flag", "sum"),
            pool_ids=("pool_id", lambda x: ",".join(sorted(x.unique()))),
        )
        .reset_index()
    )

    wallet_agg["lp_type"] = wallet_agg["num_pools"].apply(
        lambda n: "mono_lp" if n == 1 else "multi_lp"
    )

    # Re-order columns.
    wallet_agg = wallet_agg[
        ["wallet", "num_pools", "lp_type", "profitable_pools", "consistent_pools", "pool_ids"]
    ].sort_values("wallet").reset_index(drop=True)

    mono_count = (wallet_agg["lp_type"] == "mono_lp").sum()
    multi_count = (wallet_agg["lp_type"] == "multi_lp").sum()
    logger.info(
        "aggregate_cross_pool: %d wallets total (%d mono_lp, %d multi_lp).",
        len(wallet_agg),
        mono_count,
        multi_count,
    )
    return wallet_agg


def pool_participation_distribution(cross_pool_df: pd.DataFrame) -> pd.DataFrame:
    """Count wallets by number of pools participated in.

    Args:
        cross_pool_df: Output of :func:`aggregate_cross_pool`.  Must
            contain ``num_pools``.

    Returns:
        DataFrame with columns ``num_pools`` and ``wallet_count``, sorted
        by ``num_pools`` ascending.
    """
    dist = (
        cross_pool_df["num_pools"]
        .value_counts()
        .reset_index()
        .rename(columns={"num_pools": "wallet_count", "index": "num_pools"})
    )
    # pandas >= 2.0 value_counts returns column name as the series name
    if "num_pools" not in dist.columns:
        dist = (
            cross_pool_df["num_pools"]
            .value_counts()
            .rename_axis("num_pools")
            .reset_index(name="wallet_count")
        )
    dist = dist.sort_values("num_pools").reset_index(drop=True)
    return dist


def compute_pool_intersections(
    lp_scores_df: pd.DataFrame,
    pool_ids: List[str],
) -> pd.DataFrame:
    """Compute LP wallet counts for all non-empty pool intersections (subsets).

    For N pools this enumerates all 2^N − 1 non-empty subsets of
    ``pool_ids`` and counts how many wallets participated in *exactly* or
    *at least* those pools.  The function uses the "at least" (superset)
    semantics: a wallet is counted for subset S if it participated in every
    pool in S (and possibly more).

    Args:
        lp_scores_df: LP–pool level DataFrame with columns ``lp_address``
            and ``pool_id``.
        pool_ids: List of pool identifier strings to include.

    Returns:
        DataFrame with columns:

        ``pool_subset`` (frozenset repr, e.g. ``"{'pool_a', 'pool_b'}"``)
        ``subset_size`` (number of pools in the subset),
        ``wallet_count`` (wallets that participated in all pools in subset).

        Sorted by ``wallet_count`` descending.
    """
    df = lp_scores_df.copy()
    df["wallet"] = df["lp_address"].apply(normalize_address)

    # Build a binary participation matrix: wallet × pool_id.
    participation = (
        df[df["pool_id"].isin(pool_ids)]
        .groupby(["wallet", "pool_id"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
        .astype(bool)
    )

    # Ensure all requested pools are present as columns.
    for pid in pool_ids:
        if pid not in participation.columns:
            participation[pid] = False

    rows = []
    for size in range(1, len(pool_ids) + 1):
        for subset in combinations(pool_ids, size):
            mask = participation[list(subset)].all(axis=1)
            count = int(mask.sum())
            rows.append(
                {
                    "pool_subset": str(set(subset)),
                    "subset_size": size,
                    "wallet_count": count,
                }
            )

    result = (
        pd.DataFrame(rows)
        .sort_values("wallet_count", ascending=False)
        .reset_index(drop=True)
    )
    return result


def multi_lp_success_distribution(cross_pool_df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of consistent_pools count among multi-LP wallets.

    For wallets that participate in more than one pool, count how many of
    those pools are *consistently successful* (i.e. ``consistent_pools`` for
    each wallet).

    Args:
        cross_pool_df: Output of :func:`aggregate_cross_pool`.  Must contain
            ``lp_type`` and ``consistent_pools``.

    Returns:
        DataFrame with columns ``consistent_pools`` and ``wallet_count``,
        restricted to multi-LP wallets, sorted by ``consistent_pools``.
    """
    multi = cross_pool_df[cross_pool_df["lp_type"] == "multi_lp"].copy()

    if multi.empty:
        return pd.DataFrame(columns=["consistent_pools", "wallet_count"])

    dist = (
        multi["consistent_pools"]
        .value_counts()
        .rename_axis("consistent_pools")
        .reset_index(name="wallet_count")
        .sort_values("consistent_pools")
        .reset_index(drop=True)
    )
    return dist
