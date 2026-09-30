"""Schema definitions, dataclasses, and pool metadata for the CLMM PnL pipeline.

All on-chain events are normalised into the dataclasses defined here before
any further processing.  The :data:`POOL_REGISTRY` dict provides static
metadata for the pools included in the research dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Base event dataclass
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """Minimal fields shared by every on-chain pool event.

    Attributes:
        block_number: Block height at which the event was emitted.
        transaction_hash: Hex-encoded transaction hash (with or without 0x prefix).
        transaction_index: Position of the transaction within the block.
        log_index: Position of the log within the transaction.
        timestamp: Unix timestamp (seconds) of the containing block.
        pool_id: Logical pool identifier that maps into :data:`POOL_REGISTRY`.
    """

    block_number: int
    transaction_hash: str
    transaction_index: int
    log_index: int
    timestamp: int
    pool_id: str


# ---------------------------------------------------------------------------
# Liquidity events
# ---------------------------------------------------------------------------


@dataclass
class MintEvent(Event):
    """A Mint log emitted when liquidity is added to a CLMM pool.

    Attributes:
        lp_address: Wallet address (owner) that provided the liquidity.
        tick_lower: Lower tick boundary of the position.
        tick_upper: Upper tick boundary of the position.
        liquidity: Raw liquidity units added (uint128 on-chain).
        amount0: Token0 amount deposited (raw integer units).
        amount1: Token1 amount deposited (raw integer units).
    """

    lp_address: str
    tick_lower: int
    tick_upper: int
    liquidity: int
    amount0: int
    amount1: int


@dataclass
class BurnEvent(Event):
    """A Burn log emitted when liquidity is removed from a CLMM pool.

    Attributes:
        lp_address: Wallet address that initiated the burn.
        tick_lower: Lower tick boundary of the position.
        tick_upper: Upper tick boundary of the position.
        liquidity: Raw liquidity units removed.
        amount0: Token0 amount withdrawn by the burn (excluding fees).
        amount1: Token1 amount withdrawn by the burn (excluding fees).
    """

    lp_address: str
    tick_lower: int
    tick_upper: int
    liquidity: int
    amount0: int
    amount1: int


@dataclass
class CollectEvent(Event):
    """A Collect log emitted when accumulated fees or owed tokens are withdrawn.

    Attributes:
        lp_address: Recipient wallet address.
        tick_lower: Lower tick boundary of the associated position.
        tick_upper: Upper tick boundary of the associated position.
        amount0: Token0 collected (principal + fees, raw integer units).
        amount1: Token1 collected (principal + fees, raw integer units).
    """

    lp_address: str
    tick_lower: int
    tick_upper: int
    amount0: int
    amount1: int


@dataclass
class SwapEvent(Event):
    """A Swap log recording a trade executed against the pool.

    Attributes:
        sender: Address that initiated the swap.
        recipient: Address that received the output tokens.
        amount0: Signed token0 delta (negative = out of pool).
        amount1: Signed token1 delta (negative = out of pool).
        sqrt_price_x96: Pool's sqrtPriceX96 *after* the swap.
        liquidity: Active liquidity at the time of the swap.
        tick: Pool tick *after* the swap.
    """

    sender: str
    recipient: str
    amount0: int
    amount1: int
    sqrt_price_x96: int
    liquidity: int
    tick: int


# ---------------------------------------------------------------------------
# Composite / derived event
# ---------------------------------------------------------------------------


@dataclass
class BurnPlusEvent(Event):
    """A Burn+ event combining a Burn with its matching Collect.

    A Burn+ represents the complete withdrawal of liquidity: the burn
    itself (principal recovery) plus the collect (principal + fees).
    The ``amount0_collected`` / ``amount1_collected`` fields record the
    *total* tokens received by the LP including fees.

    Attributes:
        lp_address: Owner wallet address.
        tick_lower: Lower tick boundary.
        tick_upper: Upper tick boundary.
        liquidity: Liquidity units removed (from the Burn record).
        amount0: Token0 from the Burn record (principal only, raw units).
        amount1: Token1 from the Burn record (principal only, raw units).
        amount0_collected: Token0 from the paired Collect (principal + fees).
        amount1_collected: Token1 from the paired Collect (principal + fees).
    """

    lp_address: str
    tick_lower: int
    tick_upper: int
    liquidity: int
    amount0: int
    amount1: int
    amount0_collected: float
    amount1_collected: float


# ---------------------------------------------------------------------------
# Position record
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """A reconstructed CLMM liquidity position (open → closed lifecycle).

    Attributes:
        lp_address: Owner wallet address.
        pool_id: Logical pool identifier.
        tick_lower: Lower tick boundary.
        tick_upper: Upper tick boundary.
        open_block: Block number of the opening Mint event.
        open_tx_index: Transaction index of the opening Mint.
        open_log_index: Log index of the opening Mint.
        open_timestamp: Unix timestamp of the opening block.
        close_timestamp: Unix timestamp of the final closing event.
        mint_liquidity: Liquidity minted at open.
        mint_amount0: Token0 deposited at open (raw units).
        mint_amount1: Token1 deposited at open (raw units).
        open_price: Pool price (human-readable) at the time of the Mint.
        fragments: List of partial-close dicts, each containing keys
            ``liquidity``, ``amount0``, ``amount1``, ``price``,
            ``timestamp``.
        total_close_amount0: Sum of amount0 across all fragments.
        total_close_amount1: Sum of amount1 across all fragments.
        opening_capital: W = amount1_M + amount0_M * P_M (USDC value at open).
        closing_capital: Q = Σ(amount1_s + amount0_s * P_s) across fragments.
        realized_pnl: Q − W.
    """

    lp_address: str
    pool_id: str
    tick_lower: int
    tick_upper: int
    open_block: int
    open_tx_index: int
    open_log_index: int
    open_timestamp: int
    close_timestamp: int
    mint_liquidity: int
    mint_amount0: int
    mint_amount1: int
    open_price: float
    fragments: List[dict] = field(default_factory=list)
    total_close_amount0: float = 0.0
    total_close_amount1: float = 0.0
    opening_capital: float = 0.0
    closing_capital: float = 0.0
    realized_pnl: float = 0.0


# ---------------------------------------------------------------------------
# LP-pool aggregate record
# ---------------------------------------------------------------------------


@dataclass
class LPPoolRecord:
    """Aggregate PnL record for one LP wallet across one pool.

    Attributes:
        lp_address: Owner wallet address.
        pool_id: Logical pool identifier.
        total_pnl: Sum of realized PnL across all closed positions.
        position_count: Number of fully closed positions.
        first_open_timestamp: Earliest open timestamp among positions.
        last_close_timestamp: Latest close timestamp among positions.
        omega: Win-score (Ω) as defined in the paper.
        profitable_flag: True if total_pnl > 0.
        consistent_flag: True if total_pnl > 0 *and* omega > 0.5.
    """

    lp_address: str
    pool_id: str
    total_pnl: float
    position_count: int
    first_open_timestamp: int
    last_close_timestamp: int
    omega: float = 0.0
    profitable_flag: bool = False
    consistent_flag: bool = False


# ---------------------------------------------------------------------------
# Pool registry
# ---------------------------------------------------------------------------


#: Static metadata for each pool included in the research dataset.
#:
#: Keys are logical pool identifiers used throughout the pipeline.
#: Values are dicts with:
#:   - ``address``         — checksummed on-chain contract address.
#:   - ``token0``          — token0 symbol.
#:   - ``token1``          — token1 symbol.
#:   - ``token0_decimals`` — ERC-20 decimal places for token0.
#:   - ``token1_decimals`` — ERC-20 decimal places for token1.
#:   - ``fee_tier``        — pool fee in hundredths of a basis point (bps*100).
#:   - ``protocol``        — DEX name / version string.
POOL_REGISTRY: Dict[str, Dict] = {
    "uniswap_v3_005": {
        "address": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
        "token0": "USDC",
        "token1": "WETH",
        "token0_decimals": 6,
        "token1_decimals": 18,
        "fee_tier": 500,   # 0.05 %
        "protocol": "Uniswap V3",
    },
    "aerodrome_ss_005": {
        "address": "0xcDAC0d6c6C59727a65F871236188350531885C43",
        "token0": "USDC",
        "token1": "WETH",
        "token0_decimals": 6,
        "token1_decimals": 18,
        "fee_tier": 500,   # 0.05 %
        "protocol": "Aerodrome Slipstream",
    },
    "pancakeswap_v3_001": {
        "address": "0x6CA298D2983aB03Aa1dA7679389D955A4eFEE15C",
        "token0": "USDC",
        "token1": "WETH",
        "token0_decimals": 6,
        "token1_decimals": 18,
        "fee_tier": 100,   # 0.01 %
        "protocol": "PancakeSwap V3",
    },
    "sushiswap_v3_001": {
        "address": "0xD340B57AAcDD10F96FC1CF10e15921936F41E29c",
        "token0": "USDC",
        "token1": "WETH",
        "token0_decimals": 6,
        "token1_decimals": 18,
        "fee_tier": 100,   # 0.01 %
        "protocol": "SushiSwap V3",
    },
}
