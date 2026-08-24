# Project specification

## Purpose

QuantWeaveLM 0.1.0 deterministically fits and audits a calibrated mixture of precomputed probability distributions over ordered future-return bins. It is an evaluation tool, not a price forecaster, backtester, portfolio optimizer, or trading system.

## Partition invariant

The JSONL order is the time order. The complete dataset is partitioned as:

`calibration prefix | embargo rows | untouched test suffix`

Only the calibration prefix may influence mixture weights, temperature, or climatology. Embargo and test observations may influence only their hashes, declared boundaries, and corresponding evaluation. Tests mutate those partitions to enforce this invariant.

## Mathematical contract

For model probability vectors `p_m` and simplex weights `w_m`, the pool is:

`p = sum_m(w_m * p_m)`

Weights use deterministic exponentiated-gradient minimization of calibration log loss. Temperature scaling transforms each positive probability as `p_i^(1/T)` and renormalizes; `T` is selected by bounded log-grid search on calibration log loss.

Reports include logarithmic score, multiclass Brier score, ranked probability score for ordered classes, accuracy, mean confidence, top-label ECE, and per-class reliability cells. On-chain quantities, asset prices, trades, and monetary results are outside the schema.

## Stable commands

- `quantweavelm run CONFIG DATASET OUTPUT`
- `quantweavelm verify CONFIG DATASET REPORT`
- `quantweavelm summary REPORT`
- `quantweavelm prompt REPORT [OUTPUT]`
- `quantweavelm demo`

JSON is canonicalized with sorted keys, compact separators, finite numbers only, ASCII output, and a trailing newline.
