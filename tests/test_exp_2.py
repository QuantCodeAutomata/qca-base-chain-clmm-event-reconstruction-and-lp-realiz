"""Tests for Experiment 2: win-score (omega) metric and LP classification."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from typing import List

from src.win_score import (
    compute_cumulative_pnl_path,
    compute_win_score,
    classify_lps,
    aggregate_success_by_pool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cum_path(timestamps, cum_pnl_values):
    """Build a cumulative-PnL path DataFrame directly from parallel lists."""
    return pd.DataFrame({
        "close_timestamp": timestamps,
        "cumulative_pnl": cum_pnl_values,
        # position_pnl column not needed by compute_win_score
    })


def _positions_and_scores(lp_pool_records):
    """
    Build (positions_df, lp_pool_pnl_df) from a list of dicts.

    Each dict must contain:
      lp_address, pool_id, total_pnl,
      and a list 'pos_pnls' of (close_timestamp, realized_pnl) pairs.
    """
    pos_rows = []
    agg_rows = []
    for rec in lp_pool_records:
        agg_rows.append({
            "lp_address": rec["lp_address"],
            "pool_id": rec["pool_id"],
            "total_pnl": rec["total_pnl"],
        })
        for ts, pnl in rec["pos_pnls"]:
            pos_rows.append({
                "lp_address": rec["lp_address"],
                "pool_id": rec["pool_id"],
                "close_timestamp": ts,
                "realized_pnl": pnl,
            })
    return pd.DataFrame(pos_rows), pd.DataFrame(agg_rows)


# ---------------------------------------------------------------------------
# compute_win_score tests
# ---------------------------------------------------------------------------


def test_win_score_always_positive():
    """Monotonically increasing positive path yields omega == 1.0."""
    # timestamps uniform dt=1; cumulative grows: all values positive
    # dt = [1,1,1]; c_plus = [0.25, 0.5, 0.75, 1.0]; c_minus = 0
    # a_plus = 0.25+0.5+0.75 = 1.5; a_minus = 0 → omega = 1.0
    path = _cum_path([1, 2, 3, 4], [100, 200, 300, 400])
    omega = compute_win_score(path)
    assert omega == pytest.approx(1.0, abs=1e-12), (
        f"Always-positive path must yield omega=1.0, got {omega}"
    )


def test_win_score_always_negative():
    """Monotonically decreasing negative path yields omega == 0.0."""
    # c_plus = 0, c_minus > 0 throughout → a_plus = 0 → omega = 0.0
    path = _cum_path([1, 2, 3, 4], [-100, -200, -300, -400])
    omega = compute_win_score(path)
    assert omega == pytest.approx(0.0, abs=1e-12), (
        f"Always-negative path must yield omega=0.0, got {omega}"
    )


def test_win_score_zero_path():
    """Path stuck at zero returns omega == 0.5 (S == 0 guard)."""
    path = _cum_path([1, 2, 3, 4], [0, 0, 0, 0])
    omega = compute_win_score(path)
    assert omega == pytest.approx(0.5), (
        f"All-zero path must return omega=0.5, got {omega}"
    )


def test_win_score_symmetric():
    """Path spending equal time above and below zero → omega ≈ 0.5."""
    # timestamps [0,1,2], cum_pnl [100, -100, 0]
    # c_plus = [1, 0, 0], c_minus = [0, 1, 0]; dt = [1, 1]
    # a_plus = 1, a_minus = 1 → omega = 0.5
    path = _cum_path([0, 1, 2], [100, -100, 0])
    omega = compute_win_score(path)
    assert omega == pytest.approx(0.5, abs=1e-12), (
        f"Symmetric path should yield omega=0.5, got {omega}"
    )


def test_win_score_single_position():
    """Single-row path (< 2 points) returns 0.5 without error."""
    path = _cum_path([1000], [50.0])
    omega = compute_win_score(path)
    assert omega == pytest.approx(0.5), (
        f"Single-point path must return 0.5, got {omega}"
    )


def test_omega_range():
    """omega is always in [0, 1] for diverse path shapes."""
    shapes = [
        ([0, 1, 2, 3], [10, 20, 30, 40]),           # always positive
        ([0, 1, 2, 3], [-5, -10, -15, -20]),         # always negative
        ([0, 1, 2, 3], [0, 0, 0, 0]),                # flat zero
        ([0, 1, 2, 3], [10, -10, 10, -10]),          # alternating
        ([0, 1, 2, 3], [1, -1, 2, -2]),              # alternating growing
        ([0, 1], [100, -100]),                        # two points, equal
        ([0, 1, 2], [100, -100, 0]),                  # symmetric, three points
    ]
    for timestamps, cum_pnl in shapes:
        path = _cum_path(timestamps, cum_pnl)
        omega = compute_win_score(path)
        assert 0.0 <= omega <= 1.0, (
            f"omega={omega} is outside [0,1] for path cum_pnl={cum_pnl}"
        )


# ---------------------------------------------------------------------------
# classify_lps tests
# ---------------------------------------------------------------------------


def _base_lp_dataset():
    """
    Three LPs:
      LP1 — 2 positions, both profitable (cumulative stays positive) → consistent
      LP2 — 1 position, profitable but single-point → profitable but NOT consistent
      LP3 — 1 position, losing → neither profitable nor consistent
    """
    records = [
        {
            "lp_address": "0xLP1", "pool_id": "pool_A", "total_pnl": 150.0,
            "pos_pnls": [(1000, 100.0), (2000, 50.0)],
        },
        {
            "lp_address": "0xLP2", "pool_id": "pool_A", "total_pnl": 50.0,
            "pos_pnls": [(1000, 50.0)],
        },
        {
            "lp_address": "0xLP3", "pool_id": "pool_A", "total_pnl": -50.0,
            "pos_pnls": [(1000, -50.0)],
        },
    ]
    return _positions_and_scores(records)


def test_classify_lps_profitable_flag():
    """profitable_flag is True iff total_pnl > 0."""
    positions_df, lp_pool_pnl_df = _base_lp_dataset()
    result = classify_lps(lp_pool_pnl_df, positions_df)

    lp1 = result[result["lp_address"] == "0xLP1"].iloc[0]
    lp2 = result[result["lp_address"] == "0xLP2"].iloc[0]
    lp3 = result[result["lp_address"] == "0xLP3"].iloc[0]

    assert bool(lp1["profitable_flag"]), (
        "LP1 has total_pnl=150 > 0 so profitable_flag must be True"
    )
    assert bool(lp2["profitable_flag"]), (
        "LP2 has total_pnl=50 > 0 so profitable_flag must be True"
    )
    assert not lp3["profitable_flag"], (
        "LP3 has total_pnl=-50 < 0 so profitable_flag must be False"
    )


def test_classify_lps_consistent_flag():
    """consistent_flag is True only when total_pnl > 0 AND omega > 0.5."""
    positions_df, lp_pool_pnl_df = _base_lp_dataset()
    result = classify_lps(lp_pool_pnl_df, positions_df)

    lp1 = result[result["lp_address"] == "0xLP1"].iloc[0]
    lp2 = result[result["lp_address"] == "0xLP2"].iloc[0]
    lp3 = result[result["lp_address"] == "0xLP3"].iloc[0]

    # LP1: 2 positions with cumulative path [100, 150] — always positive → omega=1.0 > 0.5
    assert bool(lp1["consistent_flag"]), (
        "LP1 has positive total_pnl and always-positive path (omega=1.0) → must be consistent"
    )
    # LP2: only 1 position → omega=0.5, NOT strictly > 0.5
    assert not lp2["consistent_flag"], (
        "LP2 has total_pnl>0 but only 1 position so omega=0.5 (not >0.5) → not consistent"
    )
    # LP3: total_pnl < 0 → cannot be consistent
    assert not lp3["consistent_flag"], (
        "LP3 has total_pnl<0 → must not be consistent"
    )


def test_classify_lps_consistent_implies_profitable():
    """Every LP with consistent_flag=True also has profitable_flag=True."""
    positions_df, lp_pool_pnl_df = _base_lp_dataset()
    result = classify_lps(lp_pool_pnl_df, positions_df)

    consistent_rows = result[result["consistent_flag"] == True]
    for _, row in consistent_rows.iterrows():
        assert bool(row["profitable_flag"]), (
            f"LP {row['lp_address']} is consistent but NOT profitable — impossible by definition"
        )


# ---------------------------------------------------------------------------
# aggregate_success_by_pool tests
# ---------------------------------------------------------------------------


def test_aggregate_success_by_pool_fractions():
    """Fractions are in [0,1] and equal profitable_lps/total_lps."""
    lp_scores_df = pd.DataFrame([
        {"lp_address": "0xLP1", "pool_id": "pool_A",
         "profitable_flag": True,  "consistent_flag": True},
        {"lp_address": "0xLP2", "pool_id": "pool_A",
         "profitable_flag": True,  "consistent_flag": False},
        {"lp_address": "0xLP3", "pool_id": "pool_A",
         "profitable_flag": False, "consistent_flag": False},
        {"lp_address": "0xLP4", "pool_id": "pool_B",
         "profitable_flag": True,  "consistent_flag": True},
        {"lp_address": "0xLP5", "pool_id": "pool_B",
         "profitable_flag": False, "consistent_flag": False},
    ])
    result = aggregate_success_by_pool(lp_scores_df)

    for _, row in result.iterrows():
        assert 0.0 <= row["profitable_fraction"] <= 1.0, (
            f"profitable_fraction={row['profitable_fraction']} out of [0,1] for pool {row['pool_id']}"
        )
        assert 0.0 <= row["consistent_fraction"] <= 1.0, (
            f"consistent_fraction={row['consistent_fraction']} out of [0,1] for pool {row['pool_id']}"
        )

    pool_a = result[result["pool_id"] == "pool_A"].iloc[0]
    assert pool_a["total_lps"] == 3, "pool_A has 3 LP rows"
    assert pool_a["profitable_lps"] == 2, "pool_A has 2 profitable LPs"
    assert pool_a["consistent_lps"] == 1, "pool_A has 1 consistent LP"
    assert pool_a["profitable_fraction"] == pytest.approx(2 / 3), (
        "pool_A profitable_fraction should be 2/3"
    )

    pool_b = result[result["pool_id"] == "pool_B"].iloc[0]
    assert pool_b["total_lps"] == 2, "pool_B has 2 LP rows"
    assert pool_b["profitable_fraction"] == pytest.approx(0.5), (
        "pool_B profitable_fraction should be 0.5"
    )
    assert pool_b["consistent_fraction"] == pytest.approx(0.5), (
        "pool_B consistent_fraction should be 0.5"
    )
