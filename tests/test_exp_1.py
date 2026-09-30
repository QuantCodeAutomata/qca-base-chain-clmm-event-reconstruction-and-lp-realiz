"""Tests for Experiment 1: CLMM event reconstruction and LP PnL pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
import numpy as np
import pandas as pd
from typing import List

from src.tick_math import tick_to_price, sqrt_price_x96_to_price, price_to_tick
from src.event_processor import (
    standardize_events,
    sort_events_chronologically,
    filter_zero_value_events,
    filter_lps_with_min_mint_burn,
    build_event_stream,
)
from src.price_attacher import build_price_series, attach_price_to_events
from src.burn_plus_matcher import match_burn_collect, validate_burn_plus
from src.dynamic_balance import (
    filter_left_censored,
    reconstruct_positions,
    compute_position_capitals,
)
from src.pnl_calculator import positions_to_dataframe, aggregate_lp_pool_pnl


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

TOKEN0_DEC = 18   # WETH
TOKEN1_DEC = 6    # USDC
POOL_ID = "uniswap_v3_005"   # registered pool, used for registry-aware tests


# ---------------------------------------------------------------------------
# tick_math tests
# ---------------------------------------------------------------------------


def test_tick_math_roundtrip():
    """tick_to_price and price_to_tick are approximate inverses."""
    prices = [100.0, 1000.0, 2500.0, 5000.0]
    for original_price in prices:
        tick = price_to_tick(original_price, TOKEN0_DEC, TOKEN1_DEC)
        recovered = tick_to_price(tick, TOKEN0_DEC, TOKEN1_DEC)
        rel_error = abs(recovered - original_price) / original_price
        assert rel_error < 0.001, (
            f"Roundtrip failed for price={original_price}: "
            f"recovered={recovered:.6f}, rel_error={rel_error:.6e}"
        )


def test_sqrt_price_x96_conversion():
    """sqrt_price_x96_to_price converts a known sqrtPriceX96 within 1% of expected."""
    expected_price = 2500.0
    # price = (sqrt_price_x96 / 2^96)^2 * 10^TOKEN0_DEC / 10^TOKEN1_DEC
    # => sqrt_raw = sqrt(price * 10^TOKEN1_DEC / 10^TOKEN0_DEC)
    price_raw = expected_price * (10 ** TOKEN1_DEC) / (10 ** TOKEN0_DEC)
    sqrt_raw = math.sqrt(price_raw)
    sqrt_price_x96 = int(sqrt_raw * (2 ** 96))

    computed = sqrt_price_x96_to_price(sqrt_price_x96, TOKEN0_DEC, TOKEN1_DEC)
    rel_error = abs(computed - expected_price) / expected_price
    assert rel_error < 0.01, (
        f"sqrt_price_x96_to_price mismatch: expected≈{expected_price}, "
        f"got {computed:.4f}, rel_error={rel_error:.4e}"
    )


# ---------------------------------------------------------------------------
# event_processor tests
# ---------------------------------------------------------------------------


def _make_mint_df(rows):
    """Build a minimal mint DataFrame from a list of dicts."""
    defaults = dict(
        block_number=1, transaction_index=0, log_index=0,
        transaction_hash="0xabc", timestamp=1000,
        lp_address="0xLP1", tick_lower=-100, tick_upper=100,
        liquidity=500, amount0=100, amount1=200,
    )
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


def test_filter_zero_value_events_removes_zeros():
    """Rows where liquidity=0, amount0=0, amount1=0 are dropped."""
    df = _make_mint_df([
        {"liquidity": 0, "amount0": 0, "amount1": 0},   # zero-value → remove
        {"liquidity": 500, "amount0": 0, "amount1": 0},  # non-zero liquidity → keep
    ])
    filtered = filter_zero_value_events(df, "mint")
    assert len(filtered) == 1, (
        f"Expected 1 row after filtering zeros, got {len(filtered)}"
    )


def test_filter_zero_value_events_keeps_nonzero():
    """All non-zero rows are retained intact."""
    df = _make_mint_df([
        {"liquidity": 100, "amount0": 5, "amount1": 10},
        {"liquidity": 200, "amount0": 0, "amount1": 20},
        {"liquidity": 0, "amount0": 1, "amount1": 0},    # amount0 nonzero → keep
    ])
    filtered = filter_zero_value_events(df, "mint")
    assert len(filtered) == 3, (
        f"All 3 non-zero-on-all-three rows should be retained; got {len(filtered)}"
    )


def test_sort_events_chronologically():
    """Events are sorted by (block_number, transaction_index, log_index) ascending."""
    df = pd.DataFrame([
        {"block_number": 200, "transaction_index": 1, "log_index": 0, "val": "C"},
        {"block_number": 100, "transaction_index": 0, "log_index": 5, "val": "B"},
        {"block_number": 100, "transaction_index": 0, "log_index": 2, "val": "A"},
        {"block_number": 200, "transaction_index": 2, "log_index": 0, "val": "D"},
    ])
    sorted_df = sort_events_chronologically(df)
    vals = sorted_df["val"].tolist()
    assert vals == ["A", "B", "C", "D"], (
        f"Chronological sort incorrect: got {vals}"
    )


def test_filter_lps_with_min_mint_burn():
    """LPs without both mint and burn records are excluded; those with both are kept."""
    mint_df = pd.DataFrame({"lp_address": ["0xLP1", "0xLP2", "0xLP3"]})
    burn_df = pd.DataFrame({"lp_address": ["0xLP2", "0xLP3", "0xLP4"]})
    eligible = filter_lps_with_min_mint_burn(mint_df, burn_df)

    assert "0xlp1" not in eligible, "0xLP1 has no burns so must not be eligible"
    assert "0xlp4" not in eligible, "0xLP4 has no mints so must not be eligible"
    assert "0xlp2" in eligible, "0xLP2 appears in both → should be eligible"
    assert "0xlp3" in eligible, "0xLP3 appears in both → should be eligible"
    assert len(eligible) == 2, f"Expected 2 eligible LPs, got {len(eligible)}"


# ---------------------------------------------------------------------------
# price_attacher tests
# ---------------------------------------------------------------------------


def test_build_price_series():
    """build_price_series returns a sorted DataFrame with correct pool_id column."""
    # Use tick-based price (avoids needing exact sqrtPriceX96 values)
    # uniswap_v3_005: token0=USDC(6), token1=WETH(18)
    swap_df = pd.DataFrame([
        {"block_number": 300, "transaction_index": 0, "log_index": 0, "tick": 198000},
        {"block_number": 100, "transaction_index": 0, "log_index": 0, "tick": 197000},
        {"block_number": 200, "transaction_index": 0, "log_index": 0, "tick": 197500},
    ])
    price_series = build_price_series(swap_df, POOL_ID)

    assert "price" in price_series.columns, "Output must have a 'price' column"
    assert "ordering_key" in price_series.columns, "Output must have an 'ordering_key' column"
    assert (price_series["pool_id"] == POOL_ID).all(), "pool_id column must match the input pool_id"
    # Verify sorted by ordering_key
    keys = price_series["ordering_key"].tolist()
    assert keys == sorted(keys), f"Price series must be sorted by ordering_key; got {keys}"
    assert len(price_series) == 3, f"Expected 3 price observations, got {len(price_series)}"


def test_attach_price_to_events():
    """Each event gets the price from the most recent prior swap (LOCF)."""
    # Swaps at blocks 50, 150, 250 with prices 1000, 2000, 3000 (arbitrary units)
    price_series = pd.DataFrame([
        {"block_number": 50,  "transaction_index": 0, "log_index": 0,
         "price": 1000.0, "pool_id": POOL_ID},
        {"block_number": 150, "transaction_index": 0, "log_index": 0,
         "price": 2000.0, "pool_id": POOL_ID},
        {"block_number": 250, "transaction_index": 0, "log_index": 0,
         "price": 3000.0, "pool_id": POOL_ID},
    ])
    from src.price_attacher import compute_ordering_key
    price_series["ordering_key"] = compute_ordering_key(price_series)

    # Events at blocks 100, 200, 300 — each should get the prior swap price
    events_df = pd.DataFrame([
        {"block_number": 100, "transaction_index": 0, "log_index": 0, "lp_address": "0xLP"},
        {"block_number": 200, "transaction_index": 0, "log_index": 0, "lp_address": "0xLP"},
        {"block_number": 300, "transaction_index": 0, "log_index": 0, "lp_address": "0xLP"},
    ])
    result = attach_price_to_events(events_df, price_series)

    prices = result.sort_values("block_number")["price"].tolist()
    assert prices[0] == pytest.approx(1000.0), (
        f"Event at block 100 should get prior price 1000, got {prices[0]}"
    )
    assert prices[1] == pytest.approx(2000.0), (
        f"Event at block 200 should get prior price 2000, got {prices[1]}"
    )
    assert prices[2] == pytest.approx(3000.0), (
        f"Event at block 300 should get prior price 3000, got {prices[2]}"
    )


# ---------------------------------------------------------------------------
# burn_plus_matcher tests
# ---------------------------------------------------------------------------


def _burn_row(**kwargs):
    base = dict(
        block_number=10, transaction_index=0, log_index=1,
        transaction_hash="0xtx1", lp_address="0xLP1",
        tick_lower=-100, tick_upper=100,
        liquidity=1000, amount0=100, amount1=200,
        timestamp=2000,
    )
    return {**base, **kwargs}


def _collect_row(**kwargs):
    base = dict(
        block_number=10, transaction_index=0, log_index=2,
        transaction_hash="0xtx1", lp_address="0xLP1",
        tick_lower=-100, tick_upper=100,
        amount0=110, amount1=220,
        timestamp=2000,
    )
    return {**base, **kwargs}


def test_match_burn_collect_same_tx():
    """Burn and Collect in the same transaction are matched via same-tx pass."""
    burn_df = pd.DataFrame([_burn_row()])
    collect_df = pd.DataFrame([_collect_row()])

    result = match_burn_collect(burn_df, collect_df, POOL_ID)

    assert result["amount0_collected"].notna().all(), (
        "Burn matched in same-tx pass should have non-NaN amount0_collected"
    )
    assert result["amount0_collected"].iloc[0] == pytest.approx(110.0), (
        f"amount0_collected should be 110 (from collect), got {result['amount0_collected'].iloc[0]}"
    )
    assert result["amount1_collected"].iloc[0] == pytest.approx(220.0), (
        f"amount1_collected should be 220 (from collect), got {result['amount1_collected'].iloc[0]}"
    )


def test_match_burn_collect_fallback():
    """Burn and Collect in different transactions are matched via fallback pass."""
    burn_df = pd.DataFrame([_burn_row(transaction_hash="0xtx_burn")])
    # Collect is in a later block/tx with a different hash
    collect_df = pd.DataFrame([_collect_row(
        block_number=11, transaction_index=0, log_index=0,
        transaction_hash="0xtx_collect",
    )])

    result = match_burn_collect(burn_df, collect_df, POOL_ID)

    assert result["amount0_collected"].notna().all(), (
        "Fallback nearest-subsequent match should populate amount0_collected"
    )
    assert result["amount0_collected"].iloc[0] == pytest.approx(110.0), (
        f"Fallback amount0_collected should be 110, got {result['amount0_collected'].iloc[0]}"
    )


def test_validate_burn_plus():
    """validate_burn_plus returns correct matched/unmatched counts."""
    burn_plus_df = pd.DataFrame({
        "amount0_collected": [100.0, 200.0, np.nan, np.nan],
        "amount1_collected": [10.0, 20.0, np.nan, np.nan],
    })
    stats = validate_burn_plus(burn_plus_df)

    assert stats["total_burns"] == 4, f"Expected total_burns=4, got {stats['total_burns']}"
    assert stats["matched"] == 2, f"Expected matched=2, got {stats['matched']}"
    assert stats["unmatched"] == 2, f"Expected unmatched=2, got {stats['unmatched']}"
    assert stats["matched_fraction"] == pytest.approx(0.5), (
        f"Expected matched_fraction=0.5, got {stats['matched_fraction']}"
    )


# ---------------------------------------------------------------------------
# dynamic_balance tests
# ---------------------------------------------------------------------------


def _make_stream(events):
    """Build a minimal stream DataFrame with all required columns."""
    defaults = dict(
        lp_address="0xLP1", pool_id="test_pool",
        tick_lower=-100, tick_upper=100,
        block_number=1, transaction_index=0, log_index=0,
        timestamp=1000, liquidity=1000,
        amount0=0, amount1=0,
        amount0_collected=0.0, amount1_collected=0.0,
        price=2500.0, signed_liquidity=1000,
        event_type="mint", left_censored=False,
    )
    return pd.DataFrame([{**defaults, **e} for e in events])


def test_filter_left_censored_removes_overshoots():
    """A burn+ with liquidity exceeding cumulative minted is flagged left_censored."""
    stream = _make_stream([
        {"block_number": 1, "event_type": "mint",     "liquidity": 1000, "signed_liquidity":  1000},
        {"block_number": 2, "event_type": "burn_plus", "liquidity": 1500, "signed_liquidity": -1500},
    ])
    result = filter_left_censored(stream)

    mint_row = result[result["event_type"] == "mint"].iloc[0]
    burn_row = result[result["event_type"] == "burn_plus"].iloc[0]

    assert not mint_row["left_censored"], "Mint rows are never left-censored"
    assert burn_row["left_censored"], (
        "Burn+ with liquidity 1500 > cumulative 1000 must be flagged left_censored"
    )


def test_reconstruct_positions_simple():
    """One mint followed by a matching burn+ produces exactly one closed position."""
    stream = _make_stream([
        {
            "block_number": 1, "event_type": "mint",
            "liquidity": 1000, "signed_liquidity": 1000,
            "amount0": 500, "amount1": 1_000_000, "timestamp": 1000,
        },
        {
            "block_number": 2, "event_type": "burn_plus",
            "liquidity": 1000, "signed_liquidity": -1000,
            "amount0_collected": 510.0, "amount1_collected": 1_050_000.0,
            "timestamp": 2000,
        },
    ])
    positions = reconstruct_positions(stream)

    assert len(positions) == 1, (
        f"Exactly 1 closed position expected, got {len(positions)}"
    )
    pos = positions[0]
    assert pos["mint_liquidity"] == 1000, (
        f"Closed position mint_liquidity should be 1000, got {pos['mint_liquidity']}"
    )
    assert pos["total_close_amount1"] == pytest.approx(1_050_000.0), (
        f"total_close_amount1 should be 1_050_000, got {pos['total_close_amount1']}"
    )


def test_reconstruct_positions_overshoot_split():
    """Burn+ liquidity > mint liquidity: closes 1000, 500 carries forward (unused)."""
    stream = _make_stream([
        {
            "block_number": 1, "event_type": "mint",
            "liquidity": 1000, "signed_liquidity": 1000,
            "amount0": 0, "amount1": 0, "timestamp": 1000,
        },
        {
            "block_number": 2, "event_type": "burn_plus",
            "liquidity": 1500, "signed_liquidity": -1500,
            "amount0_collected": 1500.0, "amount1_collected": 3000.0,
            "timestamp": 2000,
        },
    ])
    # Bypass filter_left_censored — test reconstruct_positions FIFO mechanics directly
    positions = reconstruct_positions(stream)

    assert len(positions) == 1, (
        f"The 1000-unit mint position should be closed; got {len(positions)} positions"
    )
    pos = positions[0]
    # frac = 1000/1500 → proportional amounts
    expected_a0 = 1500.0 * (1000.0 / 1500.0)
    expected_a1 = 3000.0 * (1000.0 / 1500.0)
    assert pos["total_close_amount0"] == pytest.approx(expected_a0, rel=1e-6), (
        f"total_close_amount0 should be {expected_a0:.2f} (1000/1500 share), "
        f"got {pos['total_close_amount0']}"
    )
    assert pos["total_close_amount1"] == pytest.approx(expected_a1, rel=1e-6), (
        f"total_close_amount1 should be {expected_a1:.2f} (1000/1500 share), "
        f"got {pos['total_close_amount1']}"
    )


def test_compute_position_capitals():
    """W, Q, and realized_pnl are computed correctly from raw amounts and prices."""
    # token0=WETH (18 dec), token1=USDC (6 dec)
    # mint: 1 WETH raw = 10^18, 2000 USDC raw = 2000*10^6
    # open_price = 2000 USDC/WETH
    # W = 2000/10^6 + (10^18/10^18) * 2000 = 2000 + 2000 = 4000 USDC
    #
    # fragment: 1 WETH raw, 2100 USDC raw, price=2100
    # Q = 2100/10^6 + (10^18/10^18) * 2100 = 2100 + 2100 = 4200 USDC
    # PnL = 4200 - 4000 = 200 USDC

    mint_amount0 = int(1 * 10 ** TOKEN0_DEC)   # 1 WETH in raw units
    mint_amount1 = int(2000 * 10 ** TOKEN1_DEC)  # 2000 USDC in raw units
    frag_amount0 = int(1 * 10 ** TOKEN0_DEC)
    frag_amount1 = int(2100 * 10 ** TOKEN1_DEC)

    positions = [{
        "lp_address": "0xLP1",
        "pool_id": "test_pool",
        "tick_lower": -100,
        "tick_upper": 100,
        "open_block": 1,
        "open_tx_index": 0,
        "open_log_index": 0,
        "open_timestamp": 1000,
        "close_timestamp": 2000,
        "mint_liquidity": 1000,
        "mint_amount0": mint_amount0,
        "mint_amount1": mint_amount1,
        "open_price": 2000.0,
        "fragments": [
            {"liquidity": 1000, "amount0": frag_amount0, "amount1": frag_amount1,
             "price": 2100.0, "timestamp": 2000},
        ],
        "total_close_amount0": float(frag_amount0),
        "total_close_amount1": float(frag_amount1),
    }]

    enriched = compute_position_capitals(positions, TOKEN0_DEC, TOKEN1_DEC)
    pos = enriched[0]

    assert pos["opening_capital"] == pytest.approx(4000.0, rel=1e-9), (
        f"W should be 4000 USDC, got {pos['opening_capital']}"
    )
    assert pos["closing_capital"] == pytest.approx(4200.0, rel=1e-9), (
        f"Q should be 4200 USDC, got {pos['closing_capital']}"
    )
    assert pos["realized_pnl"] == pytest.approx(200.0, rel=1e-9), (
        f"PnL should be 200 USDC (= Q - W), got {pos['realized_pnl']}"
    )


# ---------------------------------------------------------------------------
# pnl_calculator tests
# ---------------------------------------------------------------------------


def test_positions_to_dataframe_empty():
    """positions_to_dataframe([]) returns an empty DataFrame."""
    result = positions_to_dataframe([])
    assert isinstance(result, pd.DataFrame), "Result must be a DataFrame"
    assert result.empty, "DataFrame from empty list must be empty"


def test_aggregate_lp_pool_pnl():
    """aggregate_lp_pool_pnl sums realized_pnl correctly per LP-pool group."""
    positions_df = pd.DataFrame([
        # LP1 in pool_A: two positions with PnL +100 and -30 → total = +70
        {"lp_address": "0xLP1", "pool_id": "pool_A",
         "realized_pnl": 100.0, "open_timestamp": 1000, "close_timestamp": 2000},
        {"lp_address": "0xLP1", "pool_id": "pool_A",
         "realized_pnl": -30.0, "open_timestamp": 1500, "close_timestamp": 2500},
        # LP2 in pool_A: one position with PnL -50
        {"lp_address": "0xLP2", "pool_id": "pool_A",
         "realized_pnl": -50.0, "open_timestamp": 1200, "close_timestamp": 2200},
        # LP1 in pool_B: one position with PnL +200
        {"lp_address": "0xLP1", "pool_id": "pool_B",
         "realized_pnl": 200.0, "open_timestamp": 3000, "close_timestamp": 4000},
    ])

    agg = aggregate_lp_pool_pnl(positions_df)

    lp1_pool_a = agg[(agg["lp_address"] == "0xLP1") & (agg["pool_id"] == "pool_A")]
    lp2_pool_a = agg[(agg["lp_address"] == "0xLP2") & (agg["pool_id"] == "pool_A")]
    lp1_pool_b = agg[(agg["lp_address"] == "0xLP1") & (agg["pool_id"] == "pool_B")]

    assert len(lp1_pool_a) == 1, "LP1/pool_A should have exactly one aggregate row"
    assert lp1_pool_a["total_pnl"].iloc[0] == pytest.approx(70.0), (
        f"LP1/pool_A total_pnl should be 70, got {lp1_pool_a['total_pnl'].iloc[0]}"
    )
    assert lp1_pool_a["position_count"].iloc[0] == 2, (
        "LP1/pool_A should have 2 positions"
    )
    assert lp2_pool_a["total_pnl"].iloc[0] == pytest.approx(-50.0), (
        f"LP2/pool_A total_pnl should be -50, got {lp2_pool_a['total_pnl'].iloc[0]}"
    )
    assert lp1_pool_b["total_pnl"].iloc[0] == pytest.approx(200.0), (
        f"LP1/pool_B total_pnl should be 200, got {lp1_pool_b['total_pnl'].iloc[0]}"
    )
    assert len(agg) == 3, f"Expected 3 LP-pool aggregate rows, got {len(agg)}"
