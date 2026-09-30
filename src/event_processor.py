"""Event standardization, filtering, and ordering utilities.

This module normalises raw DataFrames loaded from on-chain data exports into a
consistent shape, removes economically trivial zero-value rows, and assembles
per-LP event streams ready for position reconstruction.
"""

from __future__ import annotations

import logging
from typing import Set

import pandas as pd

from src.event_schema import POOL_REGISTRY

logger = logging.getLogger(__name__)

# Columns that must be present in every liquidity-event DataFrame.
_REQUIRED_LIQUIDITY_COLS = [
    "block_number",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "timestamp",
    "lp_address",
    "tick_lower",
    "tick_upper",
    "liquidity",
    "amount0",
    "amount1",
]

# Columns required for swap DataFrames.
_REQUIRED_SWAP_COLS = [
    "block_number",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "timestamp",
    "sqrt_price_x96",
    "tick",
]


def standardize_events(df: pd.DataFrame, pool_id: str) -> pd.DataFrame:
    """Attach pool metadata and ensure required columns exist.

    Adds a ``pool_id`` column to *df* and verifies that the columns expected
    by downstream stages are present.  Missing columns are added with a
    ``NaN`` / 0 fill so that the rest of the pipeline can proceed; a warning
    is emitted for each missing column so that data-quality issues surface
    early.

    Args:
        df: Raw event DataFrame (Mint, Burn, Collect, or Swap).
        pool_id: Key into :data:`~src.event_schema.POOL_REGISTRY`.

    Returns:
        A copy of *df* with a ``pool_id`` column and all required columns
        present.

    Raises:
        KeyError: If *pool_id* is not found in :data:`~src.event_schema.POOL_REGISTRY`.
    """
    if pool_id not in POOL_REGISTRY:
        raise KeyError(f"pool_id '{pool_id}' is not in POOL_REGISTRY.")

    out = df.copy()
    out["pool_id"] = pool_id

    required = _REQUIRED_LIQUIDITY_COLS if "liquidity" in df.columns else _REQUIRED_SWAP_COLS
    for col in required:
        if col not in out.columns:
            logger.warning(
                "standardize_events: column '%s' missing from %s DataFrame — filling with 0.",
                col,
                pool_id,
            )
            out[col] = 0

    return out


def sort_events_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Sort events by their canonical on-chain ordering.

    The canonical order is (block_number ASC, transaction_index ASC,
    log_index ASC), which matches the order in which events would be
    processed by an Ethereum node replaying the chain.

    Args:
        df: DataFrame containing ``block_number``, ``transaction_index``,
            and ``log_index`` columns.

    Returns:
        A new DataFrame sorted chronologically with the index reset.
    """
    return (
        df.sort_values(
            by=["block_number", "transaction_index", "log_index"],
            ascending=True,
        )
        .reset_index(drop=True)
    )


def filter_zero_value_events(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Remove economically trivial zero-value rows.

    A row is considered zero-value — and therefore discarded — when *all*
    three of the following hold simultaneously:

        ``liquidity == 0 AND amount0 == 0 AND amount1 == 0``

    This conservative triple-zero rule avoids accidentally discarding events
    where, for example, a narrow out-of-range position moves zero tokens but
    still records non-zero liquidity.  It applies to Mint, Burn, and Collect
    events; calling this function on a Swap DataFrame is harmless but a
    warning is emitted because Swap rows never carry a ``liquidity`` column
    in the mint/burn sense.

    Args:
        df: DataFrame of events (Mint, Burn, or Collect).
        event_type: One of ``'mint'``, ``'burn'``, or ``'collect'``;
            used only for logging.

    Returns:
        Filtered DataFrame with zero-value rows removed and index reset.
    """
    if event_type.lower() not in {"mint", "burn", "collect"}:
        logger.warning(
            "filter_zero_value_events: unexpected event_type '%s'; applying filter anyway.",
            event_type,
        )

    before = len(df)
    mask = (
        (df["liquidity"].fillna(0) == 0)
        & (df["amount0"].fillna(0) == 0)
        & (df["amount1"].fillna(0) == 0)
    )
    out = df[~mask].reset_index(drop=True)
    removed = before - len(out)
    logger.info(
        "filter_zero_value_events [%s]: removed %d / %d zero-value rows.",
        event_type,
        removed,
        before,
    )
    return out


def filter_lps_with_min_mint_burn(
    mint_df: pd.DataFrame,
    burn_df: pd.DataFrame,
) -> Set[str]:
    """Return the set of LP addresses that appear in both mints and burns.

    Only LPs with at least one Mint *and* at least one Burn are eligible for
    full PnL reconstruction.  LPs that only ever minted (no observed closure)
    are excluded because their positions are right-open and their realised PnL
    cannot be computed.

    Args:
        mint_df: DataFrame of Mint events containing an ``lp_address`` column.
        burn_df: DataFrame of Burn (or Burn+) events containing an
            ``lp_address`` column.

    Returns:
        Set of lowercase LP address strings present in both DataFrames.
    """
    minters: Set[str] = set(mint_df["lp_address"].str.lower().unique())
    burners: Set[str] = set(burn_df["lp_address"].str.lower().unique())
    eligible = minters & burners
    logger.info(
        "filter_lps_with_min_mint_burn: %d minters, %d burners → %d eligible LPs.",
        len(minters),
        len(burners),
        len(eligible),
    )
    return eligible


def build_event_stream(
    mint_df: pd.DataFrame,
    burn_plus_df: pd.DataFrame,
    lp_addr: str,
    pool_id: str,
) -> pd.DataFrame:
    """Build an ordered event stream for a single LP–pool combination.

    Selects all Mint and Burn+ rows for the given LP address and pool,
    concatenates them, sorts chronologically, and adds a signed
    ``signed_liquidity`` column:

    * **Positive** for Mint events (liquidity added).
    * **Negative** for Burn+ events (liquidity removed).

    Args:
        mint_df: Standardised Mint DataFrame (all LPs / pools).
        burn_plus_df: Standardised Burn+ DataFrame (all LPs / pools).
        lp_addr: Target LP wallet address (case-insensitive).
        pool_id: Target pool identifier.

    Returns:
        DataFrame of combined Mint + Burn+ events for the LP–pool pair,
        sorted chronologically, with an additional ``signed_liquidity``
        column and an ``event_type`` column (``'mint'`` or ``'burn_plus'``).
    """
    addr_lower = lp_addr.lower()

    mint_mask = (
        (mint_df["lp_address"].str.lower() == addr_lower)
        & (mint_df["pool_id"] == pool_id)
    )
    burn_mask = (
        (burn_plus_df["lp_address"].str.lower() == addr_lower)
        & (burn_plus_df["pool_id"] == pool_id)
    )

    mints = mint_df[mint_mask].copy()
    burns = burn_plus_df[burn_mask].copy()

    mints["event_type"] = "mint"
    mints["signed_liquidity"] = mints["liquidity"].abs()

    burns["event_type"] = "burn_plus"
    burns["signed_liquidity"] = -burns["liquidity"].abs()

    stream = pd.concat([mints, burns], ignore_index=True)
    stream = sort_events_chronologically(stream)
    return stream
