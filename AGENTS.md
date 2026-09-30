# AGENTS.md — Repository Notes for AI Assistants

## Project Overview
This repo implements a CLMM LP realized PnL reconstruction pipeline for four Base-chain WETH/USDC pools.

## Key Concepts
- **CLMM**: Concentrated Liquidity Market Maker (Uniswap V3-style)
- **LP**: Liquidity Provider
- **Burn+**: Composite event = burn (liquidity removal) + collect (fee/token payout)
- **Win-score (omega)**: Area-based path-sensitive profitability metric
- **Position taxonomy**: 15-type classification based on P_start, P_end vs range [a, b]

## Module Dependency Order
1. src/tick_math.py (no deps)
2. src/event_schema.py (no deps)
3. src/event_processor.py (depends on event_schema)
4. src/price_attacher.py (depends on tick_math, event_schema)
5. src/burn_plus_matcher.py (depends on event_processor)
6. src/dynamic_balance.py (depends on burn_plus_matcher)
7. src/pnl_calculator.py (depends on dynamic_balance)
8. src/win_score.py (depends on pnl_calculator)
9. src/taxonomy.py (depends on tick_math, pnl_calculator)
10. src/cross_pool.py (depends on win_score)

## Data Strategy
- Real Base-chain data requires web3.py + archive node or subgraph API
- For testing/demo: use data/generate_synthetic_data.py
- Outputs stored in results/ as .parquet, .csv, .json, and RESULTS.md

## Testing
- All tests use synthetic data; no live API calls required
- Run: pytest tests/ -v

## Token Ordering Convention
- token0 = WETH (18 decimals)
- token1 = USDC (6 decimals)
- Price = USDC per WETH (token1/token0)
- Capital W = amount1 + amount0 * price (USDC)

## Implementation Notes (added after src/ creation)

### tick_math.py
- tick_to_price: `(1.0001**tick) * (10**token0_decimals) / (10**token1_decimals)`.
- sqrt_price_x96_to_price: `(sqrtPX96 / 2**96)**2 * (10**token0_dec) / (10**token1_dec)`.
- price_to_tick: inverse via `math.log(price * 10**token1_dec / 10**token0_dec) / math.log(1.0001)`.

### event_processor.py
- `standardize_events` normalizes `lp_address` to lowercase and fills missing required columns.
- `filter_zero_value_events` removes rows where **all three** of liquidity==0, amount0==0, amount1==0 are true.
- `filter_lps_with_min_mint_burn` normalizes addresses to lowercase before set intersection.
- `build_event_stream` adds `signed_liquidity` (positive=mint, negative=burn+).

### price_attacher.py
- `compute_ordering_key`: `block_number * 1e12 + transaction_index * 1e6 + log_index` cast to int64.
- `attach_price_to_events` uses `pd.merge_asof` with `allow_exact_matches=True` on the ordering key.
- Falls back to tick-based price if `sqrt_price_x96` column is missing/zero.

### burn_plus_matcher.py
- Primary match key: (lp_address normalized, tick_lower, tick_upper, transaction_hash).
- Fallback: nearest subsequent collect for same (lp_address, tick_lower, tick_upper) using ordering key.
- Unmatched burns get NaN in amount0_collected / amount1_collected.

### dynamic_balance.py
- `filter_left_censored`: runs per-(lp_address, tick_lower, tick_upper) group; marks burn+ as
  left_censored if abs(burn_liq) > running_balance (mints − non-censored burns so far).
- `reconstruct_positions`: FIFO — earliest mint is "open"; burn+ events consume it;
  proportional overshoot split if a single burn+ exceeds remaining mint liquidity.
- Returned position dicts include `opening_capital` (W) and `closing_capital` (Q) in USDC.

### pnl_calculator.py
- `positions_to_dataframe` flattens nested `fragments` list to separate columns
  (fragment count, total amounts, etc.) rather than expanding to one row per fragment.
- `aggregate_lp_pool_pnl` groups by (lp_address, pool_id); output column is `total_pnl`.

### win_score.py
- `compute_win_score` integrates step functions using consecutive timestamp differences
  (timestamps coerced to float seconds).
- Returns 0.5 when S==0 (no variation) or when only a single observation exists (no intervals).
- A constant positive cumulative PnL path returns exactly 1.0 (A_minus = 0).

### taxonomy.py — classify_position_type boundary notes
- **Type 5** uses strict boundaries (`a < P_start <= P_end < b`) to avoid consuming cases
  meant for Types 12 (P_start==a) and 14 (P_end==b).
- **Type 7** uses strict upper bound (`P_start < b`) to avoid consuming cases meant for
  Type 13 (P_start==b, P_end>b).
- **Type 2** adds `P_start <= b` to avoid consuming Type 10 (P_start > b AND P_end < a).
- **Type 11** is practically unreachable with the current first-match ordering: every
  (P_start <= a AND P_end <= a) sub-case is consumed by Types 1, 2, 6, 12, or 15 first.
  It is retained as a placeholder pending Figure 3 clarification from the paper.

### cross_pool.py
- `aggregate_cross_pool` normalizes addresses to lowercase before grouping.
- `pool_participation_distribution` uses pandas-version-safe `value_counts` pattern.
- `compute_pool_intersections` uses "at least" (superset) semantics: wallet counted for
  subset S if it participated in every pool in S (possibly more).

## Experiment Scripts (exp/)

### Common Data-Preparation Helpers (used in exp_1, shared by exp_2/3/4)
- `_to_unix_seconds(df)`: converts pandas Timestamp column → Unix int seconds before pipeline.
  Required because reconstruct_positions does `int(row.get("timestamp", 0))`.
- `_drop_price_col(df)`: drops the synthetic generator's 'price' column before calling
  `attach_price_to_events`, preventing a merge_asof column-name conflict.
- swap_df: rename `sqrtPriceX96` → `sqrt_price_x96` before `build_price_series` (the
  generator uses camelCase; the src module expects snake_case).
- POOL_REGISTRY token0_decimals=6 (USDC), token1_decimals=18 (WETH) for all 4 pools.
  Pass these to `compute_position_capitals` and `compute_position_metrics`.
  taxonomy.py defaults are (18, 6) — always override with the POOL_REGISTRY values.

### exp_1_reconstruction.py
- Full pipeline per pool: standardize → sort → zero-filter → price-series → attach-price
  → match-burn-collect → filter-eligible-LPs → per-LP stream → left-censor-filter →
  reconstruct → compute-capitals → validate.
- Outputs: positions_reconstructed.parquet, lp_pool_pnl.parquet, validation_tables.json,
  and 3 PNGs (position PnL dist, LP-pool PnL dist, positions-per-LP).

### exp_2_win_score.py
- Loads positions_reconstructed + lp_pool_pnl from results/ (regenerates if missing).
- Calls classify_lps → aggregate_success_by_pool → Table 4-style print.
- Outputs: lp_win_score.parquet, 3 PNGs (omega dist, PnL-vs-omega scatter, success-by-pool).

### exp_3_taxonomy.py
- Loads positions_reconstructed.parquet; calls compute_position_metrics (with t0=6, t1=18),
  assign_types_to_positions, aggregate_by_type, width_bucket for Type 5.
- Outputs: positions_typed.parquet, type_summary_table.csv, 3 PNGs.
- Synthetic type distribution: Types 5+6 dominate (as in paper), but ratio differs
  (~27% Type 5 vs paper ~46%) because synthetic price path has different GBM properties.

### exp_4_cross_pool.py
- Loads lp_win_score.parquet (regenerates chain if missing).
- Calls aggregate_cross_pool, pool_participation_distribution, compute_pool_intersections,
  multi_lp_success_distribution, builds pairwise matrix via _build_intersection_matrix.
- Outputs: wallet_cross_pool_summary.parquet, pool_intersection_matrix.csv, 3 PNGs.
- NOTE: Synthetic data shares LP addresses across pools by design (rng.choice with
  replacement=False from a shared address pool), so nearly all wallets are multi-LP.
  This differs from real data where ~10% are multi-LP.

## Test Files (tests/)

### Structure
- `test_exp_1.py` — 17 tests covering tick_math, event_processor, price_attacher,
  burn_plus_matcher, dynamic_balance, and pnl_calculator.
- `test_exp_2.py` — 10 tests for win-score (omega) computation and LP classification.
- `test_exp_3.py` — 10 tests for 15-type position taxonomy and aggregate_by_type.
- `test_exp_4.py` —  9 tests for cross-pool LP identity aggregation.

### Key Patterns
- All tests use in-memory synthetic DataFrames; no file I/O or API calls.
- **numpy boolean gotcha**: DataFrame flag columns (profitable_flag, consistent_flag) store
  `numpy.bool_` values. Use `bool(row["flag"])` or `not row["flag"]` rather than
  `row["flag"] is True` / `row["flag"] is False`, which fail because
  `numpy.bool_(True) is True` evaluates to `False` in Python 3.
- `reconstruct_positions` tests set `left_censored=False` manually on the stream DataFrame
  to bypass `filter_left_censored` and test FIFO mechanics in isolation.
- For tests requiring a registered pool_id (standardize_events, build_price_series),
  use `"uniswap_v3_005"` (token0=USDC/6dec, token1=WETH/18dec).
- Tests for token0=WETH(18)/token1=USDC(6) convention use `price_to_tick` inline to
  compute tick_lower/tick_upper for boundaries a≈2500, b≈3000 USDC/WETH.
- Type 11 reachability: `(P_start=a, P_end=a)` → Type 6 (consumed first);
  `(P_start < a, P_end=a)` → Type 11 (the one reachable boundary case).
- `aggregate_cross_pool` output has column `wallet` (not `lp_address`); filter with
  `result[result["wallet"] == normalize_address(addr)]`.
