"""Tests for Experiment 4: cross-pool LP identity aggregation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from typing import List

from src.cross_pool import (
    normalize_address,
    aggregate_cross_pool,
    pool_participation_distribution,
    compute_pool_intersections,
    multi_lp_success_distribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lp_scores(*rows):
    """Build an lp_scores DataFrame from keyword-argument dicts."""
    return pd.DataFrame(list(rows))


def _lp_row(lp_address, pool_id, profitable=False, consistent=False):
    return {
        "lp_address": lp_address,
        "pool_id": pool_id,
        "profitable_flag": profitable,
        "consistent_flag": consistent,
    }


# ---------------------------------------------------------------------------
# normalize_address
# ---------------------------------------------------------------------------


def test_normalize_address():
    """normalize_address lower-cases and strips whitespace."""
    assert normalize_address("0xABCD") == "0xabcd", (
        "normalize_address should lowercase '0xABCD' to '0xabcd'"
    )
    assert normalize_address("  0xDeAdBeEf  ") == "0xdeadbeef", (
        "normalize_address should strip whitespace and lowercase"
    )
    assert normalize_address("0x0000") == "0x0000", (
        "Already lowercase address should remain unchanged"
    )


# ---------------------------------------------------------------------------
# aggregate_cross_pool — mono LP
# ---------------------------------------------------------------------------


def test_aggregate_cross_pool_mono():
    """LPs each appearing in exactly one pool are classified as mono_lp."""
    lp_scores = _lp_scores(
        _lp_row("0xLP1", "pool_A"),
        _lp_row("0xLP2", "pool_A"),
        _lp_row("0xLP3", "pool_B"),
    )
    result = aggregate_cross_pool(lp_scores)

    assert (result["lp_type"] == "mono_lp").all(), (
        "All LPs appear in only one pool — every row must be 'mono_lp'"
    )
    assert (result["num_pools"] == 1).all(), (
        "Each mono LP participates in exactly 1 pool"
    )
    assert len(result) == 3, f"Expected 3 wallet rows, got {len(result)}"


# ---------------------------------------------------------------------------
# aggregate_cross_pool — multi LP
# ---------------------------------------------------------------------------


def test_aggregate_cross_pool_multi():
    """An LP appearing in two pools is classified as multi_lp with num_pools==2."""
    lp_scores = _lp_scores(
        _lp_row("0xLP1", "pool_A", profitable=True,  consistent=True),
        _lp_row("0xLP1", "pool_B", profitable=False, consistent=False),
        _lp_row("0xLP2", "pool_A", profitable=True,  consistent=False),
    )
    result = aggregate_cross_pool(lp_scores)

    lp1_row = result[result["wallet"] == "0xlp1"].iloc[0]
    lp2_row = result[result["wallet"] == "0xlp2"].iloc[0]

    assert lp1_row["lp_type"] == "multi_lp", (
        "0xLP1 appears in pool_A and pool_B → must be classified as multi_lp"
    )
    assert lp1_row["num_pools"] == 2, (
        f"0xLP1 is in 2 pools; num_pools should be 2, got {lp1_row['num_pools']}"
    )
    assert lp2_row["lp_type"] == "mono_lp", (
        "0xLP2 appears in only pool_A → must be classified as mono_lp"
    )


# ---------------------------------------------------------------------------
# pool_participation_distribution
# ---------------------------------------------------------------------------


def test_pool_participation_distribution():
    """Distribution counts wallets grouped by number of pools participated in."""
    lp_scores = _lp_scores(
        _lp_row("0xLP1", "pool_A"),
        _lp_row("0xLP1", "pool_B"),   # LP1 in 2 pools → multi
        _lp_row("0xLP2", "pool_A"),   # mono
        _lp_row("0xLP3", "pool_B"),   # mono
    )
    cross_pool_df = aggregate_cross_pool(lp_scores)
    dist = pool_participation_distribution(cross_pool_df)

    # Expected: num_pools=1 → 2 wallets (LP2, LP3), num_pools=2 → 1 wallet (LP1)
    row_1 = dist[dist["num_pools"] == 1]
    row_2 = dist[dist["num_pools"] == 2]

    assert len(row_1) == 1, "Should have exactly one distribution row for num_pools=1"
    assert row_1["wallet_count"].iloc[0] == 2, (
        f"2 wallets participate in exactly 1 pool; got {row_1['wallet_count'].iloc[0]}"
    )
    assert len(row_2) == 1, "Should have exactly one distribution row for num_pools=2"
    assert row_2["wallet_count"].iloc[0] == 1, (
        f"1 wallet participates in 2 pools; got {row_2['wallet_count'].iloc[0]}"
    )
    assert dist["num_pools"].is_monotonic_increasing, (
        "Distribution must be sorted by num_pools ascending"
    )


# ---------------------------------------------------------------------------
# mono + multi sum equals total unique wallets
# ---------------------------------------------------------------------------


def test_mono_multi_sum_equals_total():
    """Count of mono_lps + count of multi_lps equals total unique wallets."""
    lp_scores = _lp_scores(
        _lp_row("0xLP1", "pool_A"),
        _lp_row("0xLP1", "pool_B"),   # multi
        _lp_row("0xLP2", "pool_A"),   # mono
        _lp_row("0xLP3", "pool_B"),   # mono
        _lp_row("0xLP4", "pool_A"),   # mono
    )
    result = aggregate_cross_pool(lp_scores)

    total = len(result)
    mono_count  = (result["lp_type"] == "mono_lp").sum()
    multi_count = (result["lp_type"] == "multi_lp").sum()

    assert mono_count + multi_count == total, (
        f"mono ({mono_count}) + multi ({multi_count}) must equal total wallets ({total})"
    )
    assert total == 4, f"4 unique wallet addresses expected; got {total}"
    assert mono_count == 3, f"LP2, LP3, LP4 are mono → expected 3 mono_lps, got {mono_count}"
    assert multi_count == 1, f"LP1 is multi → expected 1 multi_lp, got {multi_count}"


# ---------------------------------------------------------------------------
# compute_pool_intersections
# ---------------------------------------------------------------------------


def test_compute_pool_intersections():
    """Wallet counts for single-pool and two-pool subsets are correct."""
    # LP1 in pool_A + pool_B; LP2 in pool_A; LP3 in pool_B
    lp_scores = _lp_scores(
        _lp_row("0xLP1", "pool_A"),
        _lp_row("0xLP1", "pool_B"),
        _lp_row("0xLP2", "pool_A"),
        _lp_row("0xLP3", "pool_B"),
    )
    pool_ids = ["pool_A", "pool_B"]
    result = compute_pool_intersections(lp_scores, pool_ids)

    # subset {pool_A}: LP1 + LP2 = 2 wallets
    row_a = result[result["pool_subset"].str.contains("pool_A") & (result["subset_size"] == 1)]
    assert len(row_a) == 1, "Should have exactly one size-1 subset for pool_A"
    assert row_a["wallet_count"].iloc[0] == 2, (
        f"pool_A subset should have 2 wallets, got {row_a['wallet_count'].iloc[0]}"
    )

    # subset {pool_B}: LP1 + LP3 = 2 wallets
    row_b = result[result["pool_subset"].str.contains("pool_B") & (result["subset_size"] == 1)]
    assert len(row_b) == 1, "Should have exactly one size-1 subset for pool_B"
    assert row_b["wallet_count"].iloc[0] == 2, (
        f"pool_B subset should have 2 wallets, got {row_b['wallet_count'].iloc[0]}"
    )

    # subset {pool_A, pool_B}: only LP1 = 1 wallet
    row_ab = result[result["subset_size"] == 2]
    assert len(row_ab) == 1, "Should have exactly one size-2 subset"
    assert row_ab["wallet_count"].iloc[0] == 1, (
        f"Intersection of pool_A ∩ pool_B should be 1 wallet (LP1), "
        f"got {row_ab['wallet_count'].iloc[0]}"
    )


# ---------------------------------------------------------------------------
# multi_lp_success_distribution
# ---------------------------------------------------------------------------


def test_multi_lp_success_distribution():
    """Distribution of consistent_pools among multi-LP wallets is correct."""
    # Create cross_pool_df directly: 4 multi-LP wallets
    cross_pool_df = pd.DataFrame([
        {"wallet": "0xa", "lp_type": "multi_lp", "num_pools": 2,
         "consistent_pools": 0, "profitable_pools": 0},
        {"wallet": "0xb", "lp_type": "multi_lp", "num_pools": 2,
         "consistent_pools": 0, "profitable_pools": 1},
        {"wallet": "0xc", "lp_type": "multi_lp", "num_pools": 3,
         "consistent_pools": 1, "profitable_pools": 2},
        {"wallet": "0xd", "lp_type": "multi_lp", "num_pools": 3,
         "consistent_pools": 2, "profitable_pools": 3},
        {"wallet": "0xe", "lp_type": "mono_lp",  "num_pools": 1,
         "consistent_pools": 1, "profitable_pools": 1},  # excluded (mono)
    ])
    dist = multi_lp_success_distribution(cross_pool_df)

    # Only multi-LP wallets: 0xa, 0xb, 0xc, 0xd
    # consistent_pools distribution: {0: 2 wallets, 1: 1 wallet, 2: 1 wallet}
    assert len(dist) == 3, (
        f"Expected 3 distinct consistent_pools values (0,1,2), got {len(dist)}"
    )
    row_0 = dist[dist["consistent_pools"] == 0]
    assert row_0["wallet_count"].iloc[0] == 2, (
        "2 multi-LP wallets have consistent_pools=0"
    )
    row_1 = dist[dist["consistent_pools"] == 1]
    assert row_1["wallet_count"].iloc[0] == 1, (
        "1 multi-LP wallet has consistent_pools=1"
    )
    row_2 = dist[dist["consistent_pools"] == 2]
    assert row_2["wallet_count"].iloc[0] == 1, (
        "1 multi-LP wallet has consistent_pools=2"
    )
    # Sorted ascending
    assert dist["consistent_pools"].is_monotonic_increasing, (
        "Distribution must be sorted by consistent_pools ascending"
    )


# ---------------------------------------------------------------------------
# Invariant tests: consistent_pools <= num_pools and profitable_pools <= num_pools
# ---------------------------------------------------------------------------


def test_consistent_pools_leq_num_pools():
    """For every wallet, consistent_pools <= num_pools."""
    lp_scores = _lp_scores(
        _lp_row("0xLP1", "pool_A", profitable=True,  consistent=True),
        _lp_row("0xLP1", "pool_B", profitable=True,  consistent=False),
        _lp_row("0xLP2", "pool_A", profitable=True,  consistent=True),
        _lp_row("0xLP3", "pool_A", profitable=False, consistent=False),
        _lp_row("0xLP3", "pool_B", profitable=False, consistent=False),
        _lp_row("0xLP3", "pool_C", profitable=True,  consistent=True),
    )
    result = aggregate_cross_pool(lp_scores)

    for _, row in result.iterrows():
        assert row["consistent_pools"] <= row["num_pools"], (
            f"Wallet {row['wallet']}: consistent_pools={row['consistent_pools']} "
            f"> num_pools={row['num_pools']} — impossible"
        )


def test_profitable_pools_leq_num_pools():
    """For every wallet, profitable_pools <= num_pools."""
    lp_scores = _lp_scores(
        _lp_row("0xLP1", "pool_A", profitable=True,  consistent=True),
        _lp_row("0xLP1", "pool_B", profitable=False, consistent=False),
        _lp_row("0xLP2", "pool_A", profitable=True,  consistent=True),
        _lp_row("0xLP2", "pool_B", profitable=True,  consistent=True),
        _lp_row("0xLP3", "pool_A", profitable=False, consistent=False),
    )
    result = aggregate_cross_pool(lp_scores)

    for _, row in result.iterrows():
        assert row["profitable_pools"] <= row["num_pools"], (
            f"Wallet {row['wallet']}: profitable_pools={row['profitable_pools']} "
            f"> num_pools={row['num_pools']} — impossible"
        )
