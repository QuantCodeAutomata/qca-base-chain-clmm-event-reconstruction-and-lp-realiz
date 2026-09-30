"""Attach pool-level price observations to liquidity events.

For each Mint or Burn+ event we need to know the prevailing pool price at the
moment it was executed.  The pool price is extracted from the Swap event stream
and then joined to liquidity events using a backward-looking (last-observation-
carried-forward) merge based on a scalar ordering key derived from
(block_number, transaction_index, log_index).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.event_schema import POOL_REGISTRY
from src import tick_math

logger = logging.getLogger(__name__)


def compute_ordering_key(df: pd.DataFrame) -> pd.Series:
    """Compute a scalar total-ordering key from the three position columns.

    The key encodes the canonical Ethereum event order:

        key = block_number * 10^12 + transaction_index * 10^6 + log_index

    The multipliers are chosen so that block_number dominates, then
    transaction_index, then log_index, with enough headroom for realistic
    values of each field (log_index < 10^6, transaction_index < 10^6).

    Args:
        df: DataFrame that must contain ``block_number``,
            ``transaction_index``, and ``log_index`` columns.

    Returns:
        Integer Series of ordering keys with dtype ``int64``.
    """
    key = (
        df["block_number"].astype(np.int64) * np.int64(10 ** 12)
        + df["transaction_index"].astype(np.int64) * np.int64(10 ** 6)
        + df["log_index"].astype(np.int64)
    )
    return key.astype(np.int64)


def build_price_series(swap_df: pd.DataFrame, pool_id: str) -> pd.DataFrame:
    """Extract a chronological price series from Swap events.

    Price is derived from ``sqrt_price_x96`` if that column is present and
    non-zero; otherwise it falls back to the tick-based approximation via
    :func:`~src.tick_math.tick_to_price`.

    Args:
        swap_df: DataFrame of Swap events for the given pool, expected to
            contain at minimum ``block_number``, ``transaction_index``,
            ``log_index``, and either ``sqrt_price_x96`` or ``tick``.
        pool_id: Key into :data:`~src.event_schema.POOL_REGISTRY` used to
            look up token decimal metadata.

    Returns:
        DataFrame with columns
        ``(block_number, transaction_index, log_index, price, pool_id,
        ordering_key)`` sorted by ``ordering_key`` ascending.

    Raises:
        KeyError: If *pool_id* is not in :data:`~src.event_schema.POOL_REGISTRY`.
        ValueError: If *swap_df* contains neither ``sqrt_price_x96`` nor
            ``tick`` columns.
    """
    if pool_id not in POOL_REGISTRY:
        raise KeyError(f"pool_id '{pool_id}' not found in POOL_REGISTRY.")

    meta = POOL_REGISTRY[pool_id]
    t0_dec: int = meta["token0_decimals"]
    t1_dec: int = meta["token1_decimals"]

    has_sqrt = "sqrt_price_x96" in swap_df.columns
    has_tick = "tick" in swap_df.columns

    if not has_sqrt and not has_tick:
        raise ValueError(
            "swap_df must contain 'sqrt_price_x96' or 'tick' to derive price."
        )

    prices_df = swap_df[
        ["block_number", "transaction_index", "log_index"]
    ].copy()

    if has_sqrt:
        valid_sqrt = (
            swap_df["sqrt_price_x96"].notna()
            & (swap_df["sqrt_price_x96"] > 0)
        )
        prices = np.where(
            valid_sqrt,
            swap_df["sqrt_price_x96"].apply(
                lambda v: tick_math.sqrt_price_x96_to_price(int(v), t0_dec, t1_dec)
                if pd.notna(v) and v > 0
                else np.nan
            ),
            np.nan,
        )
        prices_df["price"] = prices.astype(float)

        # Fall back to tick-based price for any rows where sqrt was unusable.
        if has_tick:
            fallback_mask = prices_df["price"].isna()
            if fallback_mask.any():
                prices_df.loc[fallback_mask, "price"] = (
                    swap_df.loc[fallback_mask, "tick"].apply(
                        lambda t: tick_math.tick_to_price(int(t), t0_dec, t1_dec)
                    )
                )
    else:
        prices_df["price"] = swap_df["tick"].apply(
            lambda t: tick_math.tick_to_price(int(t), t0_dec, t1_dec)
        )

    prices_df["pool_id"] = pool_id
    prices_df["ordering_key"] = compute_ordering_key(prices_df)
    prices_df = prices_df.dropna(subset=["price"])
    prices_df = prices_df.sort_values("ordering_key").reset_index(drop=True)

    logger.info(
        "build_price_series [%s]: %d price observations extracted from %d swap rows.",
        pool_id,
        len(prices_df),
        len(swap_df),
    )
    return prices_df


def attach_price_to_events(
    events_df: pd.DataFrame,
    price_series_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the last-observed pool price to each liquidity event.

    For each row in *events_df*, the function finds the most recent price
    observation from *price_series_df* whose ``ordering_key`` is ≤ that of
    the event.  This implements a "last observation carried forward" (LOCF)
    join using :func:`pandas.merge_asof`.

    If an event precedes *all* price observations (i.e. no prior price
    exists) the event's ``price`` column will be ``NaN``.  Callers should
    decide whether to drop or impute such rows.

    Args:
        events_df: DataFrame of liquidity events (Mint / Burn+) with
            ``block_number``, ``transaction_index``, and ``log_index``
            columns.
        price_series_df: Price series returned by :func:`build_price_series`,
            containing an ``ordering_key`` and a ``price`` column.

    Returns:
        A copy of *events_df* with an additional ``price`` column.  Rows
        are returned in the same order as the input after the merge.
    """
    events = events_df.copy()
    events["ordering_key"] = compute_ordering_key(events)
    original_order = events.index.copy()

    events_sorted = events.sort_values("ordering_key").reset_index(drop=True)

    price_ref = price_series_df[["ordering_key", "price"]].sort_values(
        "ordering_key"
    )

    merged = pd.merge_asof(
        events_sorted,
        price_ref,
        on="ordering_key",
        direction="backward",
    )

    # Restore original ordering.
    merged = merged.sort_values("ordering_key").reset_index(drop=True)

    nan_count = merged["price"].isna().sum()
    if nan_count > 0:
        logger.warning(
            "attach_price_to_events: %d events have no prior price observation "
            "(they precede the first swap in the price series).",
            nan_count,
        )

    return merged
