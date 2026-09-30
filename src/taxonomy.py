"""15-type CLMM position taxonomy and geometric metrics.

Position types characterise how the pool price moved relative to a position's
tick range between opening and closing.  The 15-type taxonomy is provisional
— exact boundary conditions should be cross-referenced with Figure 3 of the
accompanying paper.

Classification rule: types are tested in ascending order (1 → 15); the first
matching type is assigned.  This avoids ambiguity at boundary conditions.

Notation used throughout
------------------------
* ``a`` — lower range boundary (USDC/WETH price at tick_lower).
* ``b`` — upper range boundary (USDC/WETH price at tick_upper).
* ``P_start`` — pool price at position open.
* ``P_end`` — capital-weighted average pool price over closing fragments.
"""

from __future__ import annotations

import ast
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src import tick_math

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tick → price helper
# ---------------------------------------------------------------------------


def tick_to_price_safe(
    tick: int,
    token0_decimals: int = 18,
    token1_decimals: int = 6,
) -> float:
    """Wrapper around :func:`src.tick_math.tick_to_price` with default decimals.

    Default decimals (token0=18, token1=6) correspond to a WETH/USDC pool
    where WETH is token0 and USDC is token1, yielding price in USDC per WETH.

    Args:
        tick: Uniswap V3 tick index.
        token0_decimals: Decimal places for token0 (default 18 for WETH).
        token1_decimals: Decimal places for token1 (default 6 for USDC).

    Returns:
        Human-readable price (token1 per token0).
    """
    return tick_math.tick_to_price(tick, token0_decimals, token1_decimals)


# ---------------------------------------------------------------------------
# Geometric metrics
# ---------------------------------------------------------------------------


def _parse_fragments(raw) -> list:
    """Parse fragment data whether stored as a list or as a Python repr string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def compute_position_metrics(
    positions_df: pd.DataFrame,
    token0_decimals: int = 18,
    token1_decimals: int = 6,
) -> pd.DataFrame:
    """Compute geometric and capital metrics for every position.

    All raw on-chain token amounts (``mint_amount0``, ``mint_amount1``,
    fragment amounts) are scaled to human-readable units using the supplied
    decimal parameters before computing capital.

    Metrics added as new columns:

    +---------------------+---------------------------------------------------+
    | Column              | Description                                       |
    +=====================+===================================================+
    | ``a``               | Price at tick_lower (USDC/WETH)                   |
    +---------------------+---------------------------------------------------+
    | ``b``               | Price at tick_upper (USDC/WETH)                   |
    +---------------------+---------------------------------------------------+
    | ``P_start``         | Pool price at open                                |
    +---------------------+---------------------------------------------------+
    | ``P_end``           | Capital-weighted avg price over close fragments   |
    +---------------------+---------------------------------------------------+
    | ``delta_start``     | (P_start − a) / (b − a) clipped to [0, 1]        |
    +---------------------+---------------------------------------------------+
    | ``delta_end``       | (P_end − a) / (b − a) clipped to [0, 1]          |
    +---------------------+---------------------------------------------------+
    | ``delta``           | delta_end − delta_start                           |
    +---------------------+---------------------------------------------------+
    | ``m_a``             | P_start/a − 1                                     |
    +---------------------+---------------------------------------------------+
    | ``m_b``             | P_start/b − 1                                     |
    +---------------------+---------------------------------------------------+
    | ``d_a``             | P_end/a − 1                                       |
    +---------------------+---------------------------------------------------+
    | ``d_b``             | P_end/b − 1                                       |
    +---------------------+---------------------------------------------------+
    | ``rel_a``           | d_a − m_a                                         |
    +---------------------+---------------------------------------------------+
    | ``rel_b``           | d_b − m_b                                         |
    +---------------------+---------------------------------------------------+
    | ``R_cap``           | Q/W − 1  (capital return)                         |
    +---------------------+---------------------------------------------------+
    | ``R_price``         | P_end/P_start − 1  (price return)                 |
    +---------------------+---------------------------------------------------+
    | ``duration_sec``    | close_timestamp − open_timestamp in seconds       |
    +---------------------+---------------------------------------------------+
    | ``width_usdc``      | b − a  (range width in USDC)                      |
    +---------------------+---------------------------------------------------+

    Args:
        positions_df: DataFrame from
            :func:`~src.pnl_calculator.positions_to_dataframe` augmented with
            capital fields from
            :func:`~src.dynamic_balance.compute_position_capitals`.  The
            ``fragments`` column may be a Python repr string or a list.
        token0_decimals: ERC-20 decimal places for token0.
        token1_decimals: ERC-20 decimal places for token1.

    Returns:
        A copy of *positions_df* with all metrics columns appended.
    """
    df = positions_df.copy()
    scale0 = 10 ** token0_decimals
    scale1 = 10 ** token1_decimals

    a_vals, b_vals = [], []
    p_start_vals, p_end_vals = [], []
    delta_start_vals, delta_end_vals, delta_vals = [], [], []
    m_a_vals, m_b_vals, d_a_vals, d_b_vals = [], [], [], []
    rel_a_vals, rel_b_vals = [], []
    r_cap_vals, r_price_vals = [], []
    duration_vals, width_vals = [], []

    for _, row in df.iterrows():
        a = tick_to_price_safe(
            int(row["tick_lower"]), token0_decimals, token1_decimals
        )
        b = tick_to_price_safe(
            int(row["tick_upper"]), token0_decimals, token1_decimals
        )
        a_vals.append(a)
        b_vals.append(b)

        p_start = float(row.get("open_price") or 0.0)
        p_start_vals.append(p_start)

        # Capital-weighted P_end from fragments.
        fragments = _parse_fragments(row.get("fragments", []))
        p_end = _compute_p_end(fragments, scale0, scale1, p_start)
        p_end_vals.append(p_end)

        # Width (b - a); guard against degenerate (zero-width) ranges.
        width = b - a if b != a else 1e-10
        width_vals.append(b - a)

        # Delta metrics.
        ds = float(np.clip((p_start - a) / width, 0.0, 1.0))
        de = float(np.clip((p_end - a) / width, 0.0, 1.0))
        delta_start_vals.append(ds)
        delta_end_vals.append(de)
        delta_vals.append(de - ds)

        # Margin metrics (guard div-by-zero on degenerate ticks).
        m_a = (p_start / a - 1.0) if a != 0 else float("nan")
        m_b = (p_start / b - 1.0) if b != 0 else float("nan")
        d_a = (p_end / a - 1.0) if a != 0 else float("nan")
        d_b = (p_end / b - 1.0) if b != 0 else float("nan")
        m_a_vals.append(m_a)
        m_b_vals.append(m_b)
        d_a_vals.append(d_a)
        d_b_vals.append(d_b)
        rel_a_vals.append(d_a - m_a)
        rel_b_vals.append(d_b - m_b)

        # Capital return.
        w = float(row.get("opening_capital") or 0.0)
        q = float(row.get("closing_capital") or 0.0)
        r_cap = (q / w - 1.0) if w != 0 else float("nan")
        r_cap_vals.append(r_cap)

        # Price return.
        r_price = (p_end / p_start - 1.0) if p_start != 0 else float("nan")
        r_price_vals.append(r_price)

        # Duration.
        open_ts = float(row.get("open_timestamp") or 0.0)
        close_ts = float(row.get("close_timestamp") or 0.0)
        duration_vals.append(close_ts - open_ts)

    df["a"] = a_vals
    df["b"] = b_vals
    df["P_start"] = p_start_vals
    df["P_end"] = p_end_vals
    df["delta_start"] = delta_start_vals
    df["delta_end"] = delta_end_vals
    df["delta"] = delta_vals
    df["m_a"] = m_a_vals
    df["m_b"] = m_b_vals
    df["d_a"] = d_a_vals
    df["d_b"] = d_b_vals
    df["rel_a"] = rel_a_vals
    df["rel_b"] = rel_b_vals
    df["R_cap"] = r_cap_vals
    df["R_price"] = r_price_vals
    df["duration_sec"] = duration_vals
    df["width_usdc"] = width_vals

    return df


def _compute_p_end(
    fragments: list,
    scale0: float,
    scale1: float,
    fallback: float,
) -> float:
    """Compute capital-weighted average close price from fragment records.

    Each fragment contributes weight = amount1/scale1 + (amount0/scale0) * price.
    If total capital weight is zero (e.g. all-NaN fragments), returns *fallback*.
    """
    total_weight = 0.0
    weighted_price = 0.0

    for frag in fragments:
        a0 = float(frag.get("amount0") or 0.0) / scale0
        a1 = float(frag.get("amount1") or 0.0) / scale1
        p = float(frag.get("price") or 0.0)
        cap = a1 + a0 * p
        total_weight += cap
        weighted_price += cap * p

    if total_weight == 0.0:
        return fallback
    return weighted_price / total_weight


# ---------------------------------------------------------------------------
# 15-type classification
# ---------------------------------------------------------------------------


def classify_position_type(
    P_start: float,
    P_end: float,
    a: float,
    b: float,
) -> int:
    """Assign one of 15 position types based on price vs range boundaries.

    NOTE: These type definitions are *provisional*.  The exact boundaries,
    especially for edge cases where P_start or P_end equals ``a`` or ``b``,
    should be cross-referenced with Figure 3 of the accompanying paper.
    Types are evaluated in ascending order; the first match is returned.

    Types 11 and 15 share the "both at or below lower" domain but are kept
    disjoint by excluding from Type 11 the sub-case covered by Type 15
    (``P_start < a AND P_end < a AND P_end > P_start``).  Type 15 therefore
    captures exactly the case where price is *strictly* below the range AND
    moved upward during the position's life.

    Type overview (a = tick_lower price, b = tick_upper price):

    ====  ============================================================
    Type  Condition (first match wins)
    ====  ============================================================
     1    P_end <= P_start < a              (both strictly below, down)
     2    P_end < a <= P_start              (close below, open at/above a)
     3    P_start < a < P_end <= b          (open below, close inside)
     4    P_start < a AND P_end > b         (open below, close above)
     5    a <= P_start <= P_end <= b        (both inside, price up/flat)
     6    a <= P_end <= P_start <= b        (both inside, price down/flat)
     7    a <= P_start <= b < P_end         (open inside, close above)
     8    P_start > b AND P_end >= b        (both above, any direction)
     9    P_start > b AND a <= P_end < b    (start above, end inside)
    10    P_start > b AND P_end < a         (full traversal downward)
    11    P_start <= a AND P_end <= a, excluding Type-15 sub-case
                                            (at/below lower — not strictly-below up)
    12    P_start <= a AND a <= P_end <= b  (open at/below, close inside)
    13    P_start >= b AND P_end >= b       (both at/above upper)
    14    a <= P_start <= b AND P_end >= b  (open inside, close at/above)
    15    P_start < a AND P_end < a         (both strictly below, price went up)
    ====  ============================================================

    Args:
        P_start: Pool price at position open.
        P_end: Pool price at (or capital-weighted average over) close.
        a: Price at tick_lower boundary.
        b: Price at tick_upper boundary (must satisfy b >= a).

    Returns:
        Integer in [1, 15].  Returns -1 if no type matches (should not
        occur for valid inputs; indicates a data issue).
    """
    # Type 1: both strictly below range, price moved down (P_end <= P_start < a)
    if P_end <= P_start and P_start < a:
        return 1

    # Type 2: close below lower, open at or above lower — but NOT above upper
    # (P_start > b cases are handled by Types 8-10).
    if P_end < a and a <= P_start <= b:
        return 2

    # Type 3: open below range, close inside (P_start < a < P_end <= b)
    if P_start < a and P_end > a and P_end <= b:
        return 3

    # Type 4: open below range, close above range (P_start < a AND P_end > b)
    if P_start < a and P_end > b:
        return 4

    # Type 5: both strictly inside range, price up or flat.
    # Use strict lower and upper bounds so that:
    #   P_start == a → falls through to Type 12 (open at/below lower, close inside)
    #   P_end   == b → falls through to Type 14 (open inside, close at/above upper)
    if a < P_start <= P_end < b:
        return 5

    # Type 6: both inside (with boundary), price down or flat.
    if a <= P_end <= P_start <= b and P_end <= P_start:
        return 6

    # Type 7: open inside range, close strictly above.
    # Use strict P_start < b so that P_start == b falls through to Type 13
    # (both at or above upper boundary).
    if a <= P_start < b and P_end > b:
        return 7

    # Type 8: both above range (P_start > b AND P_end >= b)
    if P_start > b and P_end >= b:
        return 8

    # Type 9: start above, end inside (P_start > b AND a <= P_end < b)
    if P_start > b and a <= P_end < b:
        return 9

    # Type 10: start above, end below — full traversal downward
    if P_start > b and P_end < a:
        return 10

    # Type 11: both at or below lower boundary, but NOT the "strictly below AND price
    # went up" sub-case (which belongs to Type 15).  This keeps Types 11 and 15
    # mutually exclusive despite their overlapping literal descriptions.
    #
    # NOTE on reachability: with the current first-match ordering, every sub-case
    # that satisfies (P_start <= a AND P_end <= a) is consumed by an earlier type
    # (Type 1 for went-down, Type 2 for P_start in [a,b], Type 6 for P_start=P_end=a,
    # Type 12 for P_end=a going up) before reaching this check.  Type 11 is therefore
    # practically unreachable and is retained only as a placeholder pending Figure 3
    # clarification from the paper.
    _type15_subcase = P_start < a and P_end < a and P_end > P_start
    if P_start <= a and P_end <= a and not _type15_subcase:
        return 11

    # Type 12: open at/below lower, close inside (P_start <= a AND a <= P_end <= b)
    if P_start <= a and a <= P_end <= b:
        return 12

    # Type 13: both at or above upper boundary (P_start >= b AND P_end >= b)
    if P_start >= b and P_end >= b:
        return 13

    # Type 14: open inside, close at/above upper (a <= P_start <= b AND P_end >= b)
    if a <= P_start <= b and P_end >= b:
        return 14

    # Type 15: both strictly below range, price moved up (P_start < a AND P_end < a)
    if P_start < a and P_end < a:
        return 15

    # Fallback — indicates an unexpected combination of inputs.
    logger.warning(
        "classify_position_type: no type matched "
        "(P_start=%.4f, P_end=%.4f, a=%.4f, b=%.4f); returning -1.",
        P_start,
        P_end,
        a,
        b,
    )
    return -1


# ---------------------------------------------------------------------------
# Apply classification and aggregate
# ---------------------------------------------------------------------------


def assign_types_to_positions(positions_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`classify_position_type` row-wise to add a ``position_type`` column.

    Args:
        positions_metrics_df: Output of :func:`compute_position_metrics`
            containing ``P_start``, ``P_end``, ``a``, ``b`` columns.

    Returns:
        A copy with an integer ``position_type`` column added.
    """
    df = positions_metrics_df.copy()
    df["position_type"] = df.apply(
        lambda row: classify_position_type(
            float(row["P_start"]),
            float(row["P_end"]),
            float(row["a"]),
            float(row["b"]),
        ),
        axis=1,
    )
    type_counts = df["position_type"].value_counts().sort_index().to_dict()
    logger.info("assign_types_to_positions: type distribution: %s", type_counts)
    return df


def aggregate_by_type(typed_positions_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise positions by type: count, transaction share, mint capital share.

    Args:
        typed_positions_df: Output of :func:`assign_types_to_positions`
            containing ``position_type``, ``opening_capital`` columns.

    Returns:
        DataFrame with columns:

        ``position_type``, ``count``,
        ``transaction_share`` (count / total positions),
        ``mint_capital_share`` (sum(opening_capital) / total opening_capital).

        Sorted by ``position_type`` ascending.
    """
    total_positions = len(typed_positions_df)
    total_capital = typed_positions_df["opening_capital"].sum()

    agg = (
        typed_positions_df
        .groupby("position_type")
        .agg(
            count=("position_type", "count"),
            mint_capital=("opening_capital", "sum"),
        )
        .reset_index()
        .sort_values("position_type")
    )

    agg["transaction_share"] = (
        agg["count"] / total_positions if total_positions > 0 else 0.0
    )
    agg["mint_capital_share"] = (
        agg["mint_capital"] / total_capital if total_capital != 0 else 0.0
    )

    return agg[["position_type", "count", "transaction_share", "mint_capital_share"]]


# ---------------------------------------------------------------------------
# Width bucket helper
# ---------------------------------------------------------------------------


def width_bucket(width_usdc: float) -> str:
    """Assign a position's range width to a labelled bucket.

    Buckets are defined in USDC (same units as ``width_usdc`` from
    :func:`compute_position_metrics`):

    * ``'<10'``        — widths below 10 USDC.
    * ``'[10,100)'``   — widths in [10, 100).
    * ``'[100,1000)'`` — widths in [100, 1 000).
    * ``'[1000,10000)'`` — widths in [1 000, 10 000).
    * ``'>=10000'``    — widths at or above 10 000 USDC.

    Args:
        width_usdc: Range width in USDC (``b − a``).

    Returns:
        Bucket label string.
    """
    if width_usdc < 10:
        return "<10"
    if width_usdc < 100:
        return "[10,100)"
    if width_usdc < 1000:
        return "[100,1000)"
    if width_usdc < 10000:
        return "[1000,10000)"
    return ">=10000"
