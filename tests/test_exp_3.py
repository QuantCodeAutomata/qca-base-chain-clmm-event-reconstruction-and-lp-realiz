"""Tests for Experiment 3: CLMM position taxonomy (15-type classification)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from typing import List

from src.taxonomy import (
    classify_position_type,
    compute_position_metrics,
    assign_types_to_positions,
    aggregate_by_type,
    width_bucket,
)
from src.tick_math import price_to_tick, tick_to_price


# ---------------------------------------------------------------------------
# Constants for boundary prices used in classification tests
# ---------------------------------------------------------------------------

A = 2500.0   # tick-range lower boundary (USDC/WETH)
B = 3000.0   # tick-range upper boundary (USDC/WETH)


# ---------------------------------------------------------------------------
# classify_position_type — per-type tests
# ---------------------------------------------------------------------------


def test_classify_type_5_both_inside():
    """Both prices strictly inside the range with price going up → Type 5."""
    t = classify_position_type(P_start=2600.0, P_end=2700.0, a=A, b=B)
    assert t == 5, (
        f"P_start=2600 and P_end=2700 both strictly inside [2500,3000] with P_end>P_start "
        f"should give Type 5, got {t}"
    )


def test_classify_type_3_open_below_close_inside():
    """Open below range, close inside → Type 3."""
    t = classify_position_type(P_start=2300.0, P_end=2700.0, a=A, b=B)
    assert t == 3, (
        f"P_start=2300 < a=2500, P_end=2700 inside [2500,3000] → Type 3, got {t}"
    )


def test_classify_type_7_open_inside_close_above():
    """Open inside range (strictly below upper), close strictly above → Type 7."""
    t = classify_position_type(P_start=2700.0, P_end=3200.0, a=A, b=B)
    assert t == 7, (
        f"P_start=2700 inside [2500,3000), P_end=3200 > 3000 → Type 7, got {t}"
    )


def test_classify_type_boundary_all_below():
    """Both prices strictly below the range with price falling → Type 1 (or 11)."""
    t = classify_position_type(P_start=2000.0, P_end=1800.0, a=A, b=B)
    assert t in (1, 11), (
        f"Both P_start=2000 and P_end=1800 below a=2500 with price falling "
        f"should give Type 1 or 11, got {t}"
    )


def test_classify_type_boundary_all_above():
    """Both prices strictly above the range → Type 8 (or 13)."""
    t = classify_position_type(P_start=3500.0, P_end=3600.0, a=A, b=B)
    assert t in (8, 13), (
        f"Both P_start=3500 and P_end=3600 above b=3000 "
        f"should give Type 8 or 13, got {t}"
    )


# ---------------------------------------------------------------------------
# classify_position_type — exhaustive 15-type reachability test
# ---------------------------------------------------------------------------


def test_all_15_types_reachable():
    """Every type 1-15 is produced by a distinct (P_start, P_end) combination."""
    # a=2500, b=3000; boundary conditions chosen to hit each type exactly
    type_inputs = [
        # (P_start, P_end, expected_type)
        (2000.0, 1800.0, 1),   # both below range, price went down
        (2600.0, 2400.0, 2),   # open inside [a,b], close below a
        (2300.0, 2700.0, 3),   # open below a, close inside (strictly)
        (2300.0, 3200.0, 4),   # open below a, close above b
        (2600.0, 2700.0, 5),   # both strictly inside, price up
        (2700.0, 2600.0, 6),   # both inside, price down
        (2700.0, 3200.0, 7),   # open inside (P_start < b), close strictly above
        (3200.0, 3100.0, 8),   # both above b, any direction
        (3200.0, 2700.0, 9),   # start above b, end inside
        (3200.0, 2300.0, 10),  # full traversal downward
        (2300.0, 2500.0, 11),  # P_end == a exactly → at-lower, not going up strictly
        (2500.0, 2600.0, 12),  # P_start == a exactly, close inside
        (3000.0, 3200.0, 13),  # P_start == b exactly, both at/above upper
        (2700.0, 3000.0, 14),  # P_end == b exactly, open inside close at upper
        (2300.0, 2400.0, 15),  # both strictly below a, price went up
    ]

    produced_types = set()
    for p_start, p_end, expected in type_inputs:
        actual = classify_position_type(p_start, p_end, A, B)
        produced_types.add(actual)
        assert actual == expected, (
            f"P_start={p_start}, P_end={p_end}, a={A}, b={B}: "
            f"expected Type {expected}, got {actual}"
        )

    assert len(produced_types) == 15, (
        f"Expected 15 distinct types; produced only {len(produced_types)}: {sorted(produced_types)}"
    )
    assert produced_types == set(range(1, 16)), (
        f"Missing types: {set(range(1, 16)) - produced_types}"
    )


# ---------------------------------------------------------------------------
# aggregate_by_type
# ---------------------------------------------------------------------------


def test_type_shares_sum_to_100():
    """transaction_share across all types sums to 1.0 (100%)."""
    typed_df = pd.DataFrame([
        {"position_type": 5, "opening_capital": 1000.0},
        {"position_type": 5, "opening_capital": 2000.0},
        {"position_type": 6, "opening_capital":  500.0},
        {"position_type": 3, "opening_capital":  750.0},
        {"position_type": 7, "opening_capital": 1200.0},
    ])
    agg = aggregate_by_type(typed_df)
    total_share = agg["transaction_share"].sum()
    assert total_share == pytest.approx(1.0, abs=1e-9), (
        f"transaction_share columns should sum to 1.0 (100%), got {total_share}"
    )


# ---------------------------------------------------------------------------
# width_bucket
# ---------------------------------------------------------------------------


def test_width_bucket_boundaries():
    """width_bucket returns the correct label for each boundary value."""
    cases = [
        (0.5,    "<10"),
        (9.99,   "<10"),
        (10.0,   "[10,100)"),
        (99.0,   "[10,100)"),
        (100.0,  "[100,1000)"),
        (999.0,  "[100,1000)"),
        (1000.0, "[1000,10000)"),
        (9999.0, "[1000,10000)"),
        (10000.0, ">=10000"),
    ]
    for width, expected_label in cases:
        label = width_bucket(width)
        assert label == expected_label, (
            f"width_bucket({width}) should return '{expected_label}', got '{label}'"
        )


# ---------------------------------------------------------------------------
# compute_position_metrics
# ---------------------------------------------------------------------------


def _build_metrics_df(open_price, frag_price, frag_amount1_raw,
                      opening_capital, closing_capital,
                      token0_dec=18, token1_dec=6):
    """
    Build a single-row positions DataFrame suitable for compute_position_metrics.

    Uses price_to_tick to derive tick_lower / tick_upper for boundaries
    a ≈ 2500, b ≈ 3000.  The fragment has only amount1 (no WETH) so that
    P_end = frag_price exactly (capital-weighted average collapses to frag_price).
    """
    tick_lower = price_to_tick(A, token0_dec, token1_dec)
    tick_upper = price_to_tick(B, token0_dec, token1_dec)

    fragments = [
        {
            "amount0": 0,
            "amount1": frag_amount1_raw,
            "price": frag_price,
        }
    ]

    return pd.DataFrame([{
        "tick_lower":       tick_lower,
        "tick_upper":       tick_upper,
        "open_price":       open_price,
        "fragments":        fragments,
        "opening_capital":  opening_capital,
        "closing_capital":  closing_capital,
        "open_timestamp":   0,
        "close_timestamp":  3600,
    }])


def test_compute_position_metrics_delta_in_01():
    """delta_start and delta_end are clipped to [0, 1] when price is outside the range."""
    # P_start = 1000 << a ≈ 2500  → (P_start - a)/(b-a) < 0 → clipped to 0
    # P_end   = 4000 >> b ≈ 3000  → (P_end   - a)/(b-a) > 1 → clipped to 1
    #   With only amount1 in the fragment, P_end = frag_price = 4000.
    token0_dec, token1_dec = 18, 6
    df = _build_metrics_df(
        open_price=1000.0,
        frag_price=4000.0,
        frag_amount1_raw=int(1 * 10 ** token1_dec),   # 1 USDC raw
        opening_capital=1000.0,
        closing_capital=1200.0,
        token0_dec=token0_dec,
        token1_dec=token1_dec,
    )

    result = compute_position_metrics(df, token0_dec, token1_dec)

    ds = result["delta_start"].iloc[0]
    de = result["delta_end"].iloc[0]

    assert ds == pytest.approx(0.0), (
        f"P_start=1000 << a≈2500 → delta_start must be clipped to 0, got {ds}"
    )
    assert de == pytest.approx(1.0), (
        f"P_end=4000 >> b≈3000 → delta_end must be clipped to 1, got {de}"
    )


def test_compute_position_metrics_r_cap():
    """R_cap = closing_capital / opening_capital - 1."""
    token0_dec, token1_dec = 18, 6
    W, Q = 1000.0, 1200.0
    expected_r_cap = Q / W - 1.0   # = 0.2

    df = _build_metrics_df(
        open_price=2700.0,          # inside range, no clipping
        frag_price=2700.0,          # P_end = 2700 inside range
        frag_amount1_raw=int(1 * 10 ** token1_dec),
        opening_capital=W,
        closing_capital=Q,
        token0_dec=token0_dec,
        token1_dec=token1_dec,
    )

    result = compute_position_metrics(df, token0_dec, token1_dec)
    r_cap = result["R_cap"].iloc[0]

    assert r_cap == pytest.approx(expected_r_cap, rel=1e-9), (
        f"R_cap should be Q/W - 1 = {expected_r_cap:.4f}, got {r_cap}"
    )
