"""Match Burn records with Collect records to form Burn+ composite events.

In Uniswap V3-style CLMM pools a liquidity withdrawal always involves two
on-chain events:

1. **Burn** — removes liquidity and accounts for owed principal amounts.
2. **Collect** — transfers the owed tokens (principal + accrued fees) to
   the LP's wallet.

A ``Burn+`` pairs these two events so that both the principal (from Burn)
and the total collected tokens including fees (from Collect) are available
in a single record for PnL reconstruction.

Matching strategy
-----------------
Primary:   same ``transaction_hash``, same LP (``lp_address``), same tick
           range (``tick_lower``, ``tick_upper``).
Fallback:  if no same-transaction Collect is found, take the nearest
           subsequent Collect for the same LP and tick range (by
           ``ordering_key``).  This handles protocols that split the
           Burn and Collect into consecutive transactions.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.price_attacher import compute_ordering_key

logger = logging.getLogger(__name__)

# Columns in the final Burn+ DataFrame that originate from the Collect side.
_COLLECT_COLS = ["amount0_collected", "amount1_collected"]


def match_burn_collect(
    burn_df: pd.DataFrame,
    collect_df: pd.DataFrame,
    pool_id: str,
) -> pd.DataFrame:
    """Pair each Burn record with its corresponding Collect record.

    Matching proceeds in two passes:

    1. **Same-transaction match** — a Collect with the same
       ``transaction_hash``, ``lp_address``, ``tick_lower``, and
       ``tick_upper`` as the Burn.
    2. **Nearest-subsequent fallback** — for Burns still unmatched after
       pass 1, find the earliest Collect that shares ``lp_address``,
       ``tick_lower``, ``tick_upper``, and has an ``ordering_key`` ≥ the
       Burn's ``ordering_key``.

    Unmatched Burns (no Collect found in either pass) receive ``NaN`` in
    the ``amount0_collected`` and ``amount1_collected`` columns.

    Args:
        burn_df: DataFrame of Burn events.  Must contain ``lp_address``,
            ``tick_lower``, ``tick_upper``, ``transaction_hash``,
            ``block_number``, ``transaction_index``, and ``log_index``.
        collect_df: DataFrame of Collect events.  Must contain the same
            identity columns plus ``amount0`` and ``amount1`` (which become
            ``amount0_collected`` / ``amount1_collected`` in the output).
        pool_id: Pool identifier string (used only for log messages).

    Returns:
        A copy of *burn_df* with two additional columns:
        ``amount0_collected`` and ``amount1_collected``.  Rows are in the
        same order as *burn_df*.

    Notes:
        When multiple Collect events match the same Burn in the fallback
        pass the one with the smallest ``ordering_key`` (earliest) is
        selected.  Each Collect is only consumed once (first-come, first-
        served across Burns sorted by ordering_key).
    """
    burns = burn_df.copy()
    burns["ordering_key"] = compute_ordering_key(burns)

    collects = collect_df.copy()
    collects["ordering_key"] = compute_ordering_key(collects)

    # Rename collect amount columns to avoid collision.
    collects = collects.rename(
        columns={"amount0": "amount0_collected", "amount1": "amount1_collected"}
    )

    burns["amount0_collected"] = np.nan
    burns["amount1_collected"] = np.nan

    _MATCH_KEY = ["lp_address", "tick_lower", "tick_upper", "transaction_hash"]
    _RANGE_KEY = ["lp_address", "tick_lower", "tick_upper"]

    # ------------------------------------------------------------------
    # Pass 1: same-transaction match
    # ------------------------------------------------------------------
    collect_by_tx = (
        collects
        .drop_duplicates(subset=_MATCH_KEY, keep="first")
        .set_index(_MATCH_KEY)[["amount0_collected", "amount1_collected"]]
    )

    burn_keys = burns[_MATCH_KEY].copy()
    burn_keys["lp_address"] = burn_keys["lp_address"].str.lower()
    collect_by_tx.index = pd.MultiIndex.from_tuples(
        [
            (str(a).lower(), tl, tu, tx)
            for a, tl, tu, tx in collect_by_tx.index
        ],
        names=_MATCH_KEY,
    )

    merged_tx = burns[_MATCH_KEY].copy()
    merged_tx["lp_address"] = merged_tx["lp_address"].str.lower()
    matched_mask = pd.Series(False, index=burns.index)

    for idx, row in burns.iterrows():
        key = (
            str(row["lp_address"]).lower(),
            row["tick_lower"],
            row["tick_upper"],
            row["transaction_hash"],
        )
        if key in collect_by_tx.index:
            burns.at[idx, "amount0_collected"] = collect_by_tx.at[key, "amount0_collected"]
            burns.at[idx, "amount1_collected"] = collect_by_tx.at[key, "amount1_collected"]
            matched_mask.at[idx] = True

    pass1_matched = matched_mask.sum()
    logger.info(
        "match_burn_collect [%s] pass-1 (same-tx): %d / %d burns matched.",
        pool_id,
        pass1_matched,
        len(burns),
    )

    # ------------------------------------------------------------------
    # Pass 2: nearest-subsequent fallback for unmatched burns
    # ------------------------------------------------------------------
    unmatched_idx = burns.index[~matched_mask].tolist()

    if unmatched_idx:
        # Build a lookup structure: for each (lp, tick_lower, tick_upper),
        # a list of (ordering_key, amount0_collected, amount1_collected)
        # sorted ascending by ordering_key.  We track which collects have
        # already been consumed.
        collect_pool: Dict = {}
        for _, c_row in collects.sort_values("ordering_key").iterrows():
            rkey = (
                str(c_row["lp_address"]).lower(),
                int(c_row["tick_lower"]),
                int(c_row["tick_upper"]),
            )
            collect_pool.setdefault(rkey, []).append(
                {
                    "ordering_key": int(c_row["ordering_key"]),
                    "amount0_collected": c_row["amount0_collected"],
                    "amount1_collected": c_row["amount1_collected"],
                    "consumed": False,
                }
            )

        # Process unmatched burns in chronological order.
        unmatched_burns_sorted = (
            burns.loc[unmatched_idx]
            .sort_values("ordering_key")
        )

        for idx, b_row in unmatched_burns_sorted.iterrows():
            rkey = (
                str(b_row["lp_address"]).lower(),
                int(b_row["tick_lower"]),
                int(b_row["tick_upper"]),
            )
            candidates = collect_pool.get(rkey, [])
            b_key = int(b_row["ordering_key"])

            for candidate in candidates:
                if not candidate["consumed"] and candidate["ordering_key"] >= b_key:
                    burns.at[idx, "amount0_collected"] = candidate["amount0_collected"]
                    burns.at[idx, "amount1_collected"] = candidate["amount1_collected"]
                    candidate["consumed"] = True
                    break

    pass2_matched = (
        burns.loc[unmatched_idx, "amount0_collected"].notna().sum()
        if unmatched_idx
        else 0
    )
    total_matched = pass1_matched + pass2_matched
    total_unmatched = len(burns) - total_matched

    logger.info(
        "match_burn_collect [%s] pass-2 (nearest-subsequent): %d additional matches.",
        pool_id,
        pass2_matched,
    )
    logger.info(
        "match_burn_collect [%s] final: %d matched, %d unmatched out of %d total burns.",
        pool_id,
        total_matched,
        total_unmatched,
        len(burns),
    )

    return burns


def validate_burn_plus(burn_plus_df: pd.DataFrame) -> dict:
    """Summarise the matching quality of a Burn+ DataFrame.

    Args:
        burn_plus_df: DataFrame returned by :func:`match_burn_collect`.

    Returns:
        Dict with keys:

        ``total_burns``
            Total number of Burn records.
        ``matched``
            Burns that have non-NaN ``amount0_collected``.
        ``unmatched``
            Burns with NaN ``amount0_collected``.
        ``matched_fraction``
            ``matched / total_burns`` (0.0 if total_burns == 0).
    """
    total = len(burn_plus_df)
    matched = int(burn_plus_df["amount0_collected"].notna().sum())
    unmatched = total - matched
    matched_fraction = matched / total if total > 0 else 0.0

    result = {
        "total_burns": total,
        "matched": matched,
        "unmatched": unmatched,
        "matched_fraction": matched_fraction,
    }
    logger.info("validate_burn_plus: %s", result)
    return result
