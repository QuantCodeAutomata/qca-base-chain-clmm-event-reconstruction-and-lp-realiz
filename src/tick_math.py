"""CLMM tick/price conversion utilities for Uniswap V3-style pools.

All price values are expressed as token1 per token0 (e.g. USDC per WETH),
adjusted by the decimal differences of each token so that human-readable
units are returned.
"""

import math


def tick_to_price(tick: int, token0_decimals: int, token1_decimals: int) -> float:
    """Convert a Uniswap V3 tick to a human-readable price.

    The raw tick encodes the price as 1.0001^tick in terms of the raw
    (integer) token amounts.  The decimal adjustment converts raw units
    to human-readable units.

    Formula:
        price = (1.0001 ** tick) * (10 ** token0_decimals) / (10 ** token1_decimals)

    Args:
        tick: Uniswap V3 signed tick index.
        token0_decimals: Decimal places for token0 (e.g. 18 for WETH).
        token1_decimals: Decimal places for token1 (e.g. 6 for USDC).

    Returns:
        Human-readable price of token0 expressed in token1 units.
    """
    raw_price = 1.0001 ** tick
    return raw_price * (10 ** token0_decimals) / (10 ** token1_decimals)


def sqrt_price_x96_to_price(
    sqrt_price_x96: int,
    token0_decimals: int,
    token1_decimals: int,
) -> float:
    """Convert a Uniswap V3 sqrtPriceX96 slot0 value to a human-readable price.

    The on-chain representation stores sqrt(price_raw) * 2**96 as a uint160.
    Squaring and dividing by 2**192 recovers the raw price in token1/token0
    integer-amount terms; the decimal factor then converts to human-readable
    units.

    Formula:
        price = (sqrt_price_x96 / 2**96) ** 2
                * (10 ** token0_decimals) / (10 ** token1_decimals)

    Args:
        sqrt_price_x96: On-chain sqrtPriceX96 value (non-negative integer).
        token0_decimals: Decimal places for token0.
        token1_decimals: Decimal places for token1.

    Returns:
        Human-readable price of token0 expressed in token1 units.
    """
    sqrt_price = sqrt_price_x96 / (2 ** 96)
    raw_price = sqrt_price ** 2
    return raw_price * (10 ** token0_decimals) / (10 ** token1_decimals)


def price_to_tick(price: float, token0_decimals: int, token1_decimals: int) -> int:
    """Convert a human-readable price back to the nearest Uniswap V3 tick.

    This is the inverse of :func:`tick_to_price`.  The adjusted price is
    divided by the decimal factor to recover the raw price, and then the
    tick is obtained via a change of base:
        tick = log(price_raw) / log(1.0001)

    Args:
        price: Human-readable price of token0 in token1 units (must be > 0).
        token0_decimals: Decimal places for token0.
        token1_decimals: Decimal places for token1.

    Returns:
        Nearest integer tick corresponding to the given price.

    Raises:
        ValueError: If *price* is not positive.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")

    # Undo the decimal adjustment to obtain the raw price.
    price_raw = price * (10 ** token1_decimals) / (10 ** token0_decimals)
    tick = math.log(price_raw) / math.log(1.0001)
    return int(math.floor(tick))
