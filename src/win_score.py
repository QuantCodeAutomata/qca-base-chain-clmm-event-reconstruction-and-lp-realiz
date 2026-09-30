"""Win-score (omega, Ω) metric and LP classification.

The omega metric captures both the magnitude *and* temporal consistency of an
LP's cumulative PnL path.  An LP that is profitable throughout most of its
activity period scores near 1; an LP that is predominantly loss-making scores
near 0; an LP with a flat or zero PnL path scores 0.5.

Algorithm (exact reproduction of the paper's definition)
---------------------------------------------------------
Given the sorted sequence of (close_timestamp, cumulative_pnl) observations:

1. Split into positive path  ``PnL_cum_plus(t)  = max(cumulative_pnl, 0)``
   and negative path ``PnL_cum_minus(t) = min(cumulative_pnl, 0)``.
2. ``P_max_plus  = max(PnL_cum_plus)``
   ``P_min_minus = min(PnL_cum_minus)``
3. ``S = max(P_max_plus, −P_min_minus)``  (absolute scale normaliser).
4. If ``S == 0`` → return 0.5 (trivial path, no PnL movement).
5. ``C_plus(t)  = PnL_cum_plus  / S``  (normalised positive path, ∈ [0,1])
   ``C_minus(t) = −PnL_cum_minus / S`` (normalised negative path, ∈ [0,1])
6. Integrate step functions over time (timestamps in float seconds):
   ``A_plus  = Σ_i  C_plus(t_i)  * Δt_i``
   ``A_minus = Σ_i  C_minus(t_i) * Δt_i``
   where ``Δt_i = t_{i+1} − t_i`` for interior points and the last
   observation contributes zero width (step function with no final step).
7. ``Ω = A_plus / (A_plus + A_minus)``
   If ``A_plus + A_minus == 0`` → return 0.5.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_cumulative_pnl_path(
    positions_df: pd.DataFrame,
    lp_address: str,
    pool_id: str,
) -> pd.DataFrame:
    """Build the cumulative PnL time series for a given LP–pool pair.

    Args:
        positions_df: Flat positions DataFrame from
            :func:`~src.pnl_calculator.positions_to_dataframe`.  Must
            contain ``lp_address``, ``pool_id``, ``close_timestamp``, and
            ``realized_pnl`` columns.
        lp_address: Target LP wallet address (case-insensitive comparison).
        pool_id: Target pool identifier.

    Returns:
        DataFrame sorted by ``close_timestamp`` with columns:
        ``close_timestamp``, ``position_pnl``, ``cumulative_pnl``.
        Returns an empty DataFrame if no matching positions exist.
    """
    mask = (
        (positions_df["lp_address"].str.lower() == lp_address.lower())
        & (positions_df["pool_id"] == pool_id)
    )
    subset = (
        positions_df[mask][["close_timestamp", "realized_pnl"]]
        .rename(columns={"realized_pnl": "position_pnl"})
        .sort_values("close_timestamp")
        .reset_index(drop=True)
    )

    if subset.empty:
        return pd.DataFrame(
            columns=["close_timestamp", "position_pnl", "cumulative_pnl"]
        )

    subset["cumulative_pnl"] = subset["position_pnl"].cumsum()
    return subset


def compute_win_score(cum_pnl_path: pd.DataFrame) -> float:
    """Compute the omega (Ω) win-score from a cumulative PnL path.

    See module docstring for the full algorithm.

    Args:
        cum_pnl_path: DataFrame returned by
            :func:`compute_cumulative_pnl_path`.  Must contain
            ``close_timestamp`` (numeric seconds or any type convertible to
            float) and ``cumulative_pnl`` columns.

    Returns:
        Omega score in [0, 1].  Returns 0.5 when the path is trivially flat
        or when ``A_plus + A_minus == 0``.
    """
    if cum_pnl_path.empty or len(cum_pnl_path) < 2:
        return 0.5

    timestamps = cum_pnl_path["close_timestamp"].astype(float).to_numpy()
    cum_pnl = cum_pnl_path["cumulative_pnl"].to_numpy(dtype=float)

    # Step 1 — split into positive / negative paths.
    pnl_cum_plus = np.maximum(cum_pnl, 0.0)
    pnl_cum_minus = np.minimum(cum_pnl, 0.0)

    # Steps 2-3 — scale normaliser.
    p_max_plus = float(np.max(pnl_cum_plus))
    p_min_minus = float(np.min(pnl_cum_minus))
    s = max(p_max_plus, -p_min_minus)

    # Step 4 — trivial path guard.
    if s == 0.0:
        return 0.5

    # Step 5 — normalise.
    c_plus = pnl_cum_plus / s
    c_minus = -pnl_cum_minus / s  # non-negative

    # Step 6 — integrate step functions.
    # Δt_i = t_{i+1} − t_i for i = 0 … n-2; last interval has width 0.
    dt = np.diff(timestamps.astype(float))  # shape (n-1,)
    a_plus = float(np.dot(c_plus[:-1], dt))
    a_minus = float(np.dot(c_minus[:-1], dt))

    # Step 7 — omega.
    denom = a_plus + a_minus
    if denom == 0.0:
        return 0.5

    return a_plus / denom


def classify_lps(
    lp_pool_pnl_df: pd.DataFrame,
    positions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-LP-pool omega, profitable, and consistent flags.

    For each row in *lp_pool_pnl_df* this function:

    1. Builds the cumulative PnL path via :func:`compute_cumulative_pnl_path`.
    2. Computes ``omega`` via :func:`compute_win_score`.
    3. Sets ``profitable_flag = (total_pnl > 0)``.
    4. Sets ``consistent_flag = (total_pnl > 0) AND (omega > 0.5)``.

    Args:
        lp_pool_pnl_df: LP–pool level DataFrame from
            :func:`~src.pnl_calculator.aggregate_lp_pool_pnl`.  Must
            contain ``lp_address``, ``pool_id``, ``total_pnl``.
        positions_df: Position-level DataFrame from
            :func:`~src.pnl_calculator.positions_to_dataframe`.

    Returns:
        A copy of *lp_pool_pnl_df* with added columns:
        ``omega``, ``profitable_flag``, ``consistent_flag``.
    """
    result = lp_pool_pnl_df.copy()
    omegas = []
    profitables = []
    consistents = []

    for _, row in result.iterrows():
        path = compute_cumulative_pnl_path(
            positions_df,
            lp_address=row["lp_address"],
            pool_id=row["pool_id"],
        )
        omega = compute_win_score(path)
        profitable = bool(row["total_pnl"] > 0)
        consistent = profitable and (omega > 0.5)

        omegas.append(omega)
        profitables.append(profitable)
        consistents.append(consistent)

    result["omega"] = omegas
    result["profitable_flag"] = profitables
    result["consistent_flag"] = consistents

    logger.info(
        "classify_lps: %d LP–pool rows classified; %d profitable, %d consistent.",
        len(result),
        sum(profitables),
        sum(consistents),
    )
    return result


def aggregate_success_by_pool(lp_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LP success metrics to pool level (mirrors paper Table 4).

    Args:
        lp_scores_df: Output of :func:`classify_lps` or equivalent DataFrame
            with columns ``pool_id``, ``profitable_flag``, ``consistent_flag``.

    Returns:
        Pool-level DataFrame with columns:

        ``pool_id``,
        ``total_lps`` (unique LP–pool rows),
        ``profitable_lps`` (count where profitable_flag is True),
        ``consistent_lps`` (count where consistent_flag is True),
        ``profitable_fraction`` (profitable_lps / total_lps),
        ``consistent_fraction`` (consistent_lps / total_lps).

        Sorted by ``pool_id``.
    """
    agg = (
        lp_scores_df
        .groupby("pool_id", sort=True)
        .agg(
            total_lps=("lp_address", "count"),
            profitable_lps=("profitable_flag", "sum"),
            consistent_lps=("consistent_flag", "sum"),
        )
        .reset_index()
    )

    agg["profitable_fraction"] = agg["profitable_lps"] / agg["total_lps"]
    agg["consistent_fraction"] = agg["consistent_lps"] / agg["total_lps"]

    logger.info("aggregate_success_by_pool: %d pool rows.", len(agg))
    return agg
