"""FIFO position reconstruction with left-censored filtering and overshoot splitting.

Pipeline overview
-----------------
1. :func:`filter_left_censored` — remove Burn+ events that imply the position
   was opened before the observation window (left censoring).
2. :func:`reconstruct_positions` — consume the filtered event stream with a
   FIFO queue per ``(lp_address, pool_id, tick_lower, tick_upper)`` group to
   emit fully closed position records.
3. :func:`compute_position_capitals` — add monetary capital metrics (W, Q,
   realized PnL) to each closed position.

Capital convention
------------------
All raw token amounts (``mint_amount0``, ``mint_amount1``, fragment amounts)
are converted to human-readable units by dividing by ``10 ** decimals``.
Prices are assumed to be already in human-readable units (token1 per token0)
as produced by :mod:`src.price_attacher`.  Capital is expressed in token1
units:

    W = amount1_M + amount0_M * P_M
    Q = Σ_s  (amount1_s + amount0_s * P_s)
    PnL = Q − W
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def filter_left_censored(stream_df: pd.DataFrame) -> pd.DataFrame:
    """Mark Burn+ events that predate any observed Mint as left-censored.

    For each ``(lp_address, tick_lower, tick_upper)`` group events are visited
    in chronological order.  A running balance tracks the cumulative
    net-minted liquidity observed so far.  A Burn+ event is flagged as
    ``left_censored`` when its absolute liquidity exceeds the running balance,
    because the excess could only have been minted before the observation
    window began.  Left-censored burns are retained in the returned DataFrame
    (with ``left_censored=True``) so that callers can audit them; downstream
    functions must ignore them.

    Running balance update rules:

    * **Mint** → ``running_balance += liquidity``; ``left_censored = False``.
    * **Burn+, not censored** → ``running_balance -= abs(liquidity)``;
      ``left_censored = False``.
    * **Burn+, censored** → ``running_balance`` unchanged; ``left_censored = True``.

    Args:
        stream_df: Combined and chronologically sorted event stream as returned
            by :func:`~src.event_processor.build_event_stream`.  Must contain
            ``lp_address``, ``tick_lower``, ``tick_upper``, ``event_type``
            (``'mint'`` or ``'burn_plus'``), and ``signed_liquidity`` columns.

    Returns:
        A copy of *stream_df* with a boolean ``left_censored`` column added.
        Mint rows always have ``left_censored=False``.
    """
    out = stream_df.copy()
    out["left_censored"] = False

    group_keys = ["lp_address", "tick_lower", "tick_upper"]
    for _, group in out.groupby(group_keys, sort=False):
        running_balance = 0
        for idx, row in group.iterrows():
            if row["event_type"] == "mint":
                running_balance += abs(row["signed_liquidity"])
                out.at[idx, "left_censored"] = False
            else:  # burn_plus
                burn_liq = abs(row["signed_liquidity"])
                if burn_liq > running_balance:
                    out.at[idx, "left_censored"] = True
                    logger.debug(
                        "filter_left_censored: censored burn idx=%d "
                        "(burn_liq=%d > running_balance=%d).",
                        idx,
                        burn_liq,
                        running_balance,
                    )
                else:
                    running_balance -= burn_liq
                    out.at[idx, "left_censored"] = False

    n_censored = out["left_censored"].sum()
    logger.info(
        "filter_left_censored: %d / %d events marked as left_censored.",
        n_censored,
        len(out),
    )
    return out


def reconstruct_positions(stream_df: pd.DataFrame) -> List[dict]:
    """Reconstruct fully closed CLMM positions using a FIFO queue.

    Each ``(lp_address, pool_id, tick_lower, tick_upper)`` group is processed
    independently.  Left-censored rows (``left_censored == True``) are
    excluded before processing.

    FIFO mechanics
    ~~~~~~~~~~~~~~
    An internal queue holds open positions ordered by ``(block_number,
    transaction_index, log_index)``.  Each queue entry carries:

    * The original Mint event metadata (open coordinates, liquidity).
    * ``remaining_liquidity`` — the portion not yet consumed by any Burn+.
    * ``fragments`` — list of partial-close records accumulated so far.

    When a Burn+ event arrives with ``burn_liq = abs(signed_liquidity)``:

    1. Starting from the oldest open position, assign ``consume = min(remaining_liq,
       carry_liq)``.
    2. The fraction ``consume / carry_liq`` of the Burn+'s collected amounts is
       allocated to the fragment.
    3. If the position is fully consumed (``remaining_liq == 0``) it is
       **closed** and appended to the results list.
    4. Any un-consumed ``carry_liq`` moves to the next oldest open position.

    Args:
        stream_df: Output of :func:`filter_left_censored` — must contain
            ``left_censored`` column.

    Returns:
        List of position dicts, each with keys:

        ``lp_address``, ``pool_id``, ``tick_lower``, ``tick_upper``,
        ``open_block``, ``open_tx_index``, ``open_log_index``,
        ``open_timestamp``, ``close_timestamp``,
        ``mint_liquidity``, ``mint_amount0``, ``mint_amount1``,
        ``open_price``,
        ``fragments`` (list of dicts: ``liquidity``, ``amount0``,
        ``amount1``, ``price``, ``timestamp``),
        ``total_close_amount0``, ``total_close_amount1``.
    """
    active = stream_df[~stream_df["left_censored"]].copy()

    group_keys = ["lp_address", "pool_id", "tick_lower", "tick_upper"]
    closed_positions: List[dict] = []

    for group_id, group in active.groupby(group_keys, sort=False):
        lp_addr, pool_id, tick_lower, tick_upper = group_id
        group_sorted = group.sort_values(
            ["block_number", "transaction_index", "log_index"]
        )

        # Queue of open position dicts.
        open_queue: deque = deque()

        for _, row in group_sorted.iterrows():
            if row["event_type"] == "mint":
                open_queue.append(
                    {
                        "lp_address": lp_addr,
                        "pool_id": pool_id,
                        "tick_lower": tick_lower,
                        "tick_upper": tick_upper,
                        "open_block": int(row["block_number"]),
                        "open_tx_index": int(row["transaction_index"]),
                        "open_log_index": int(row["log_index"]),
                        "open_timestamp": int(row.get("timestamp", 0)),
                        "mint_liquidity": int(row["liquidity"]),
                        "mint_amount0": int(row.get("amount0", 0)),
                        "mint_amount1": int(row.get("amount1", 0)),
                        "open_price": float(row.get("price", float("nan"))),
                        "remaining_liquidity": int(row["liquidity"]),
                        "fragments": [],
                        "close_timestamp": None,
                    }
                )
            else:  # burn_plus
                carry_liq = float(abs(row["signed_liquidity"]))
                carry_a0 = float(row.get("amount0_collected", 0) or 0)
                carry_a1 = float(row.get("amount1_collected", 0) or 0)
                burn_price = float(row.get("price", float("nan")))
                burn_ts = int(row.get("timestamp", 0))

                while carry_liq > 0 and open_queue:
                    pos = open_queue[0]
                    remaining = float(pos["remaining_liquidity"])

                    consume = min(remaining, carry_liq)
                    # Proportional share of collected amounts for this consume.
                    frac = consume / carry_liq if carry_liq > 0 else 0.0
                    frag_a0 = carry_a0 * frac
                    frag_a1 = carry_a1 * frac

                    pos["fragments"].append(
                        {
                            "liquidity": consume,
                            "amount0": frag_a0,
                            "amount1": frag_a1,
                            "price": burn_price,
                            "timestamp": burn_ts,
                        }
                    )

                    pos["remaining_liquidity"] = remaining - consume
                    carry_liq -= consume
                    carry_a0 -= frag_a0
                    carry_a1 -= frag_a1

                    if pos["remaining_liquidity"] <= 0:
                        pos["close_timestamp"] = burn_ts
                        pos["total_close_amount0"] = sum(
                            f["amount0"] for f in pos["fragments"]
                        )
                        pos["total_close_amount1"] = sum(
                            f["amount1"] for f in pos["fragments"]
                        )
                        # Remove internal tracking keys before storing.
                        pos.pop("remaining_liquidity", None)
                        closed_positions.append(pos)
                        open_queue.popleft()

    logger.info(
        "reconstruct_positions: %d closed positions reconstructed.",
        len(closed_positions),
    )
    return closed_positions


def compute_position_capitals(
    positions: List[dict],
    token0_decimals: int,
    token1_decimals: int,
) -> List[dict]:
    """Add capital metrics (W, Q, realized PnL) to each closed position.

    Token amounts stored in position dicts are expected to be in raw on-chain
    integer units.  This function converts them to human-readable units using
    the supplied decimal parameters before computing capital values.

    Capital formulae (all values in token1 units after decimal adjustment):

        W = amount1_M / 10^d1 + (amount0_M / 10^d0) * P_M

        Q = Σ_s  [ amount1_s / 10^d1 + (amount0_s / 10^d0) * P_s ]

        PnL = Q − W

    where ``P`` is the price expressed as token1 per token0 (human-readable),
    ``M`` denotes the opening Mint event, and ``s`` indexes the closing
    fragments.

    Args:
        positions: List of position dicts as returned by
            :func:`reconstruct_positions`.
        token0_decimals: ERC-20 decimal places for token0.
        token1_decimals: ERC-20 decimal places for token1.

    Returns:
        The same list with each dict augmented by:
        ``opening_capital`` (W), ``closing_capital`` (Q), ``realized_pnl``.
    """
    scale0 = 10 ** token0_decimals
    scale1 = 10 ** token1_decimals

    for pos in positions:
        amount0_m = pos["mint_amount0"] / scale0
        amount1_m = pos["mint_amount1"] / scale1
        price_m = pos.get("open_price") or 0.0

        opening_capital = amount1_m + amount0_m * price_m

        closing_capital = 0.0
        for frag in pos.get("fragments", []):
            frag_a0 = frag["amount0"] / scale0
            frag_a1 = frag["amount1"] / scale1
            frag_p = frag.get("price") or 0.0
            closing_capital += frag_a1 + frag_a0 * frag_p

        pos["opening_capital"] = opening_capital
        pos["closing_capital"] = closing_capital
        pos["realized_pnl"] = closing_capital - opening_capital

    return positions
