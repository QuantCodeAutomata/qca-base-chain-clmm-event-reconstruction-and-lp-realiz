# Base-Chain CLMM Event Reconstruction and LP Realized PnL

This repository implements a research pipeline for reconstructing realized liquidity-provider (LP) profit and loss from raw on-chain events for four Base-chain WETH/USDC concentrated-liquidity pools over September 2024 to July 2025.

## Experiments
- **exp_1**: CLMM event reconstruction and LP realized PnL pipeline
- **exp_2**: LP win-score (omega) path-dependent success measurement
- **exp_3**: 15-type position taxonomy and behavioral analysis
- **exp_4**: Cross-pool LP identity aggregation and mono vs multi-LP analysis

## Pools
- Uniswap V3 0.05% (WETH/USDC)
- Aerodrome SlipStream 0.05% (WETH/USDC)
- PancakeSwap V3 0.01% (WETH/USDC)
- SushiSwap V3 0.01% (WETH/USDC)

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Run all experiments:
```bash
python exp/exp_1_reconstruction.py
python exp/exp_2_win_score.py
python exp/exp_3_taxonomy.py
python exp/exp_4_cross_pool.py
```

Run tests:
```bash
pytest tests/ -v
```

## Data
Since Base-chain on-chain CLMM events are not available via standard market data APIs, a synthetic data generator is provided:
```bash
python data/generate_synthetic_data.py
```

## Results
See `results/RESULTS.md` for experiment findings and metric tables.
