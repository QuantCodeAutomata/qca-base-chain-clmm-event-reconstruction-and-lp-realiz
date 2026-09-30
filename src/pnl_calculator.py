"""Aggregate per-position PnL to LP–pool level summaries.

This module provides three utilities:

* :func:`positions_to_dataframe` — flatten the list of position dicts
  produced by :mod:`src.dynamic_balance` into a tidy DataFrame.
* :func:`aggregate_lp_pool_pnl` — group-by-sum to the LP–pool granularity.
* :func:`compute_validation_counts` — audit helper that counts events per
  pool at different pipeline stages.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def positions_to_dataframe(positions: List[dict]) -> pd.DataFrame:
    """Flatten a list of closed position dicts to a tidy DataFrame.

    Each dict produced by :func:`~src.dynamic_balance.reconstruct_positions`
    (and optionally enriched by
    :func:`~src.dynamic_balance.compute_position_capitals`) becomes one row.
    The ``fragments`` field (a list) is serialised to a string so that the
    DataFrame remains rectangular; callers that need per-fragment detail
    should work with the raw list directly.

    Args:
        positions: List of position dicts as returned by
            :func:`~src.dynamic_balance.compute_position_capitals`.

    Returns:
        DataFrame with one row per closed position and a column for every
        key present across all dicts.  The ``fragments`` column, if present,
        contains the JSON-like string representation of the list.  An empty
        DataFrame is returned when *positions* is empty.
    """
    if not positions:
        return pd.DataFrame()

    rows = []
    for pos in positions:
        row = {k: v for k, v in pos.items() if k != "fragments"}
        row["fragments"] = str(pos.get("fragments", []))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Normalise timestamp columns to numeric where possible.
    for col in ("open_timestamp", "close_timestamp"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("positions_to_dataframe: %d rows created.", len(df))
    return df


def aggregate_lp_pool_pnl(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Group positions by (lp_address, pool_id) and aggregate PnL metrics.

    Args:
        positions_df: DataFrame returned by :func:`positions_to_dataframe`.
            Must contain columns: ``lp_address``, ``pool_id``,
            ``realized_pnl``, ``open_timestamp``, ``close_timestamp``.

    Returns:
        LP–pool level DataFrame with columns:

        ``lp_address``, ``pool_id``,
        ``total_pnl`` (sum of realized_pnl),
        ``position_count`` (number of closed positions),
        ``first_open_timestamp`` (min open_timestamp),
        ``last_close_timestamp`` (max close_timestamp).

        Rows are sorted by ``lp_address``, then ``pool_id``.
    """
    if positions_df.empty:
        return pd.DataFrame(
            columns=[
                "lp_address",
                "pool_id",
                "total_pnl",
                "position_count",
                "first_open_timestamp",
                "last_close_timestamp",
            ]
        )

    agg = (
        positions_df
        .groupby(["lp_address", "pool_id"], sort=True)
        .agg(
            total_pnl=("realized_pnl", "sum"),
            position_count=("realized_pnl", "count"),
            first_open_timestamp=("open_timestamp", "min"),
            last_close_timestamp=("close_timestamp", "max"),
        )
        .reset_index()
    )

    logger.info(
        "aggregate_lp_pool_pnl: %d LP–pool rows aggregated from %d positions.",
        len(agg),
        len(positions_df),
    )
    return agg


def compute_validation_counts(
    events_dfs: Dict[str, pd.DataFrame],
    stage: str,
) -> dict:
    """Count events per pool and in total at a given pipeline stage.

    This utility is designed to be called at multiple checkpoints
    (``'raw'``, ``'filtered'``, ``'used'``) so that event attrition can be
    tracked end-to-end.

    Args:
        events_dfs: Mapping of ``pool_id → DataFrame`` for the events at
            this stage.  Each DataFrame is expected to have a ``pool_id``
            column (or the key is used as the identifier if the column is
            absent).
        stage: Label for the current pipeline stage.  Used only in the
            returned dict and log messages.  Typical values: ``'raw'``,
            ``'filtered'``, ``'used'``.

    Returns:
        Dict with structure::

            {
                "stage": <stage>,
                "total": <int>,          # sum across all pools
                "by_pool": {
                    "<pool_id>": <int>,  # row count for that pool
                    ...
                },
            }
    """
    if stage not in {"raw", "filtered", "used"}:
        logger.warning(
            "compute_validation_counts: unexpected stage '%s' (expected one of "
            "'raw', 'filtered', 'used').",
            stage,
        )

    by_pool: Dict[str, int] = {}
    for pool_id, df in events_dfs.items():
        by_pool[pool_id] = len(df) if df is not None else 0

    total = sum(by_pool.values())
    result = {
        "stage": stage,
        "total": total,
        "by_pool": by_pool,
    }
    logger.info(
        "compute_validation_counts [%s]: total=%d, by_pool=%s",
        stage,
        total,
        by_pool,
    )
    return result
