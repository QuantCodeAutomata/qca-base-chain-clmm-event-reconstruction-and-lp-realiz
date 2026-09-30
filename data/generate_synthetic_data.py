"""Synthetic Base-chain CLMM event data generator for testing and demonstration.

Generates realistic mint, burn, collect, and swap events for four WETH/USDC pools.
Since actual Base-chain on-chain CLMM data requires archive node access, this module
provides synthetic data that preserves the structural properties needed to test
the LP PnL reconstruction pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List

DATA_DIR = Path(__file__).parent
PROJECT_ROOT = DATA_DIR.parent

POOL_IDS = [
    'uniswap_v3_005',
    'aerodrome_ss_005',
    'pancakeswap_v3_001',
    'sushiswap_v3_001',
]

POOL_SIZES = {
    'uniswap_v3_005': 800,      # largest pool
    'aerodrome_ss_005': 600,
    'pancakeswap_v3_001': 200,
    'sushiswap_v3_001': 50,
}

TOKEN0_DECIMALS = 18  # WETH
TOKEN1_DECIMALS = 6   # USDC

# Sep 2024 to Jul 2025
START_TS = pd.Timestamp('2024-09-01', tz='UTC')
END_TS = pd.Timestamp('2025-07-31', tz='UTC')
TOTAL_SECONDS = int((END_TS - START_TS).total_seconds())
BASE_BLOCK = 20_000_000
BLOCKS_PER_SECOND = 0.5  # Base chain ~2s block time


def generate_price_path(n_steps: int, start_price: float = 2800.0, seed: int = 42) -> np.ndarray:
    """Generate a GBM-like price path for WETH/USDC."""
    rng = np.random.default_rng(seed)
    mu = 0.0001
    sigma = 0.02
    dt = 1.0 / n_steps
    log_returns = rng.normal(mu * dt, sigma * np.sqrt(dt), n_steps)
    log_returns = np.clip(log_returns, -0.15, 0.15)
    prices = start_price * np.exp(np.cumsum(log_returns))
    return prices


def price_to_tick(price: float, token0_dec: int = 18, token1_dec: int = 6) -> int:
    """Convert price (token1/token0) to Uniswap V3 tick."""
    adjusted_price = price * (10 ** token0_dec) / (10 ** token1_dec)
    return int(np.floor(np.log(adjusted_price) / np.log(1.0001)))


def tick_to_price(tick: int, token0_dec: int = 18, token1_dec: int = 6) -> float:
    """Convert tick to price."""
    raw = 1.0001 ** tick
    return raw * (10 ** token1_dec) / (10 ** token0_dec)


def price_to_sqrt_x96(price: float, token0_dec: int = 18, token1_dec: int = 6) -> int:
    """Convert price to sqrtPriceX96."""
    adjusted = price * (10 ** token0_dec) / (10 ** token1_dec)
    sqrt_val = np.sqrt(adjusted)
    return int(sqrt_val * (2 ** 96))


def generate_swap_events(
    pool_id: str,
    price_path: np.ndarray,
    timestamps: np.ndarray,
    seed: int = 0
) -> pd.DataFrame:
    """Generate swap events with sqrtPriceX96."""
    rng = np.random.default_rng(seed)
    n = len(price_path)
    # sample a fraction of timesteps as swaps
    n_swaps = max(n // 3, 100)
    indices = np.sort(rng.choice(n, size=n_swaps, replace=False))
    prices = price_path[indices]
    ts = timestamps[indices]
    blocks = (BASE_BLOCK + ts * BLOCKS_PER_SECOND).astype(int)
    records = []
    for i, (idx, price, block, t) in enumerate(zip(indices, prices, blocks, ts)):
        records.append({
            'pool_id': pool_id,
            'event_type': 'swap',
            'block_number': int(block),
            'transaction_index': int(rng.integers(0, 50)),
            'log_index': int(rng.integers(0, 5)),
            'timestamp': START_TS + pd.Timedelta(seconds=int(t)),
            'transaction_hash': f'0xswap_{pool_id}_{i:08d}',
            'sqrtPriceX96': price_to_sqrt_x96(price),
            'price': price,
            'amount0': float(rng.uniform(-10, 10)),
            'amount1': float(rng.uniform(-30000, 30000)),
            'liquidity': 0,
            'tick_lower': 0,
            'tick_upper': 0,
            'lp_address': '',
        })
    df = pd.DataFrame(records)
    df = df.sort_values(['block_number', 'transaction_index', 'log_index']).reset_index(drop=True)
    return df


def generate_lp_events(
    pool_id: str,
    lp_addresses: List[str],
    price_path: np.ndarray,
    timestamps: np.ndarray,
    seed: int = 0,
    zero_value_fraction: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate mint, burn, and collect events for a list of LP addresses."""
    rng = np.random.default_rng(seed)
    n_prices = len(price_path)
    mints, burns, collects = [], [], []
    global_tx_counter = 0

    for lp_idx, lp_addr in enumerate(lp_addresses):
        # Each LP may have 1-10 active ranges
        n_ranges = int(rng.integers(1, 8))
        for range_idx in range(n_ranges):
            # pick a random open time
            open_step = int(rng.integers(0, n_prices // 2))
            close_step = int(rng.integers(open_step + 10, n_prices))
            if close_step >= n_prices:
                close_step = n_prices - 1

            open_price = price_path[open_step]
            center_tick = price_to_tick(open_price)
            # width in ticks: narrow to wide
            half_width = int(rng.integers(10, 2000))
            tick_lower = center_tick - half_width
            tick_upper = center_tick + half_width

            # normalize ticks to fee-tier spacing (simplified)
            tick_spacing = 10 if '001' in pool_id else 10
            tick_lower = (tick_lower // tick_spacing) * tick_spacing
            tick_upper = (tick_upper // tick_spacing) * tick_spacing
            if tick_lower >= tick_upper:
                tick_upper = tick_lower + tick_spacing * 2

            # liquidity amount
            liquidity = int(rng.uniform(1e12, 1e16))
            a = tick_to_price(tick_lower)
            b = tick_to_price(tick_upper)

            # compute token amounts at open (simplified approximation)
            p = open_price
            if p < a:
                amount0 = float(liquidity) / 1e12 * (1/np.sqrt(a) - 1/np.sqrt(b))
                amount1 = 0.0
            elif p > b:
                amount0 = 0.0
                amount1 = float(liquidity) / 1e12 * (np.sqrt(b) - np.sqrt(a))
            else:
                amount0 = float(liquidity) / 1e12 * (1/np.sqrt(p) - 1/np.sqrt(b))
                amount1 = float(liquidity) / 1e12 * (np.sqrt(p) - np.sqrt(a))

            open_block = int(BASE_BLOCK + timestamps[open_step] * BLOCKS_PER_SECOND)
            open_ts = START_TS + pd.Timedelta(seconds=int(timestamps[open_step]))
            open_tx = global_tx_counter
            global_tx_counter += 1
            mint_hash = f'0xmint_{pool_id}_{lp_idx}_{range_idx}'

            mints.append({
                'pool_id': pool_id,
                'event_type': 'mint',
                'block_number': open_block,
                'transaction_index': open_tx % 200,
                'log_index': 0,
                'timestamp': open_ts,
                'transaction_hash': mint_hash,
                'lp_address': lp_addr,
                'tick_lower': tick_lower,
                'tick_upper': tick_upper,
                'liquidity': liquidity,
                'amount0': amount0,
                'amount1': amount1,
                'sqrtPriceX96': 0,
                'price': p,
            })

            # generate 1-3 burn+collect pairs for this range
            n_burns = int(rng.integers(1, 4))
            remaining_liq = liquidity
            prev_close_step = open_step + 5
            for b_idx in range(n_burns):
                if remaining_liq <= 0 or prev_close_step >= close_step:
                    break
                if b_idx == n_burns - 1 or prev_close_step + 1 >= close_step:
                    burn_step = close_step
                    burn_liq = remaining_liq
                else:
                    burn_step = int(rng.integers(prev_close_step + 1, close_step))
                    burn_liq = int(remaining_liq * rng.uniform(0.2, 0.8))

                burn_liq = min(burn_liq, remaining_liq)
                remaining_liq -= burn_liq
                close_price = price_path[min(burn_step, n_prices-1)]
                pa = tick_to_price(tick_lower)
                pb = tick_to_price(tick_upper)
                p_c = close_price
                frac = burn_liq / liquidity

                if p_c < pa:
                    b_amount0 = frac * float(liquidity) / 1e12 * (1/np.sqrt(pa) - 1/np.sqrt(pb))
                    b_amount1 = 0.0
                elif p_c > pb:
                    b_amount0 = 0.0
                    b_amount1 = frac * float(liquidity) / 1e12 * (np.sqrt(pb) - np.sqrt(pa))
                else:
                    b_amount0 = frac * float(liquidity) / 1e12 * (1/np.sqrt(p_c) - 1/np.sqrt(pb))
                    b_amount1 = frac * float(liquidity) / 1e12 * (np.sqrt(p_c) - np.sqrt(pa))

                fee_amount0 = float(rng.uniform(0, b_amount0 * 0.05)) if b_amount0 > 0 else 0.0
                fee_amount1 = float(rng.uniform(0, max(b_amount1 * 0.05, 10.0)))
                collect_amount0 = b_amount0 + fee_amount0
                collect_amount1 = b_amount1 + fee_amount1

                close_block = int(BASE_BLOCK + timestamps[min(burn_step, n_prices-1)] * BLOCKS_PER_SECOND)
                close_ts = START_TS + pd.Timedelta(seconds=int(timestamps[min(burn_step, n_prices-1)]))
                burn_tx = global_tx_counter
                global_tx_counter += 1
                burn_hash = f'0xburn_{pool_id}_{lp_idx}_{range_idx}_{b_idx}'

                burns.append({
                    'pool_id': pool_id,
                    'event_type': 'burn',
                    'block_number': close_block,
                    'transaction_index': burn_tx % 200,
                    'log_index': 1,
                    'timestamp': close_ts,
                    'transaction_hash': burn_hash,
                    'lp_address': lp_addr,
                    'tick_lower': tick_lower,
                    'tick_upper': tick_upper,
                    'liquidity': burn_liq,
                    'amount0': b_amount0,
                    'amount1': b_amount1,
                    'sqrtPriceX96': 0,
                    'price': close_price,
                })

                collects.append({
                    'pool_id': pool_id,
                    'event_type': 'collect',
                    'block_number': close_block,
                    'transaction_index': burn_tx % 200,
                    'log_index': 2,
                    'timestamp': close_ts,
                    'transaction_hash': burn_hash,  # same tx as burn
                    'lp_address': lp_addr,
                    'tick_lower': tick_lower,
                    'tick_upper': tick_upper,
                    'liquidity': 0,
                    'amount0': collect_amount0,
                    'amount1': collect_amount1,
                    'sqrtPriceX96': 0,
                    'price': close_price,
                })

                prev_close_step = burn_step + 1

        # Add some zero-value events for filtering test
        if rng.random() < zero_value_fraction:
            mints.append({
                'pool_id': pool_id,
                'event_type': 'mint',
                'block_number': open_block,
                'transaction_index': 0,
                'log_index': 9,
                'timestamp': open_ts,
                'transaction_hash': f'0xzero_{pool_id}_{lp_idx}',
                'lp_address': lp_addr,
                'tick_lower': 0,
                'tick_upper': 60,
                'liquidity': 0,
                'amount0': 0.0,
                'amount1': 0.0,
                'sqrtPriceX96': 0,
                'price': 0.0,
            })

    mint_df = pd.DataFrame(mints)
    burn_df = pd.DataFrame(burns)
    collect_df = pd.DataFrame(collects)
    return mint_df, burn_df, collect_df


def generate_all_pool_data(n_lps: int = 200, seed: int = 42) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Generate synthetic event data for all four pools.

    Args:
        n_lps: Target number of LP addresses in the largest pool.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping pool_id to dict with keys: 'swap', 'mint', 'burn', 'collect'.
    """
    rng = np.random.default_rng(seed)
    n_steps = 5000
    timestamps = np.linspace(0, TOTAL_SECONDS, n_steps)
    price_path = generate_price_path(n_steps, start_price=2800.0, seed=seed)

    # Generate a shared address pool — some addresses appear in multiple pools
    total_addresses = int(n_lps * 1.5)
    all_addresses = [f'0x{i:040x}' for i in range(1, total_addresses + 1)]

    result = {}
    for pool_seed, pool_id in enumerate(POOL_IDS):
        pool_size = POOL_SIZES[pool_id]
        # Sample addresses for this pool (with some overlap across pools)
        lp_sample = rng.choice(all_addresses, size=min(pool_size, len(all_addresses)), replace=False).tolist()
        swap_df = generate_swap_events(pool_id, price_path, timestamps, seed=seed + pool_seed)
        mint_df, burn_df, collect_df = generate_lp_events(
            pool_id, lp_sample, price_path, timestamps,
            seed=seed + pool_seed + 100
        )
        result[pool_id] = {
            'swap': swap_df,
            'mint': mint_df,
            'burn': burn_df,
            'collect': collect_df,
        }
    return result


def save_pool_data(pool_data: Dict[str, Dict[str, pd.DataFrame]], output_dir: Path = DATA_DIR) -> None:
    """Save generated pool event DataFrames to Parquet files.

    sqrtPriceX96 is a uint160 (exceeds int64) so it is stored as a string
    column; the float `price` column is the primary numeric price reference.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for pool_id, events in pool_data.items():
        for event_type, df in events.items():
            path = output_dir / f'{pool_id}_{event_type}.parquet'
            if 'sqrtPriceX96' in df.columns:
                df = df.copy()
                df['sqrtPriceX96'] = df['sqrtPriceX96'].astype(str)
            df.to_parquet(path, index=False)
            print(f'Saved {len(df)} rows to {path}')


if __name__ == '__main__':
    print('Generating synthetic CLMM event data...')
    pool_data = generate_all_pool_data(n_lps=200, seed=42)
    save_pool_data(pool_data)
    print('Done.')
