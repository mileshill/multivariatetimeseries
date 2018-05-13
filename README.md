# Multivariate Timeseries Prediction

## Motiviation

Predict percent change in a stock's price based on sequece of OHLCV percent differences.


## Implementation Steps
1. Data acquisition
2. Data preparation
3. Network topology
4. Train/test cycles


### Data Acquisition
Data were acquired via public API

### Data preparation
**File Format**
Features and targets are saved as NPY files; numpy binary format.

Features are a matrix of [row, col] = [timesteps, [open, high, low, close, volumeto]].

Naming convention follows the format: TradingPair_TargetUTCTime_[feature||target].npy

### Network topology


### Train/test cycles

