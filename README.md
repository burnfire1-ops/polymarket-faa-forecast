# Polymarket FAA Forecast — Model V5

Forecasts each market's Yes price at every minute of the next 240 minutes
as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95), from the market's
1-minute price history up to the case checkpoint.

## Model

- **Median:** the mean of the market's last 3 prices, flat across all
  240 horizons.
- **Volatility:** GARCH(1,1) fitted on the 1-minute price differences
  (at most the last 360 diffs). Grid search over alpha (0.02–0.30, step
  0.02) and beta (step 0.05, alpha + beta < 0.97) maximising the Gaussian
  log-likelihood; omega is set so the long-run variance equals the sample
  variance of the diffs. Fallback (fewer than 30 diffs, a failed fit, or
  all-zero diffs): the standard deviation of the diffs, capped at 5x the
  median |diff| (0.002 used as the scale when that median is 0), floored
  at 0.0005.
- **Multi-step variance:** var_h = long_run_var
  + persistence^(h-1) * (var_1 - long_run_var), where var_1 is the
  1-step-ahead GARCH variance.
- **V5 change — near-persistence override:** for near horizons (h <= 60)
  the persistence is the fixed value NEAR_PERSISTENCE = 0.98, not the
  fitted alpha + beta. The fitted alpha/beta still determine var_1, and
  the omega / long-run variance are unchanged from V4.
- **DHF (far horizons):** for h > 60 the variance relaxes from the h=60
  variance at a second persistence, FAR_PERSISTENCE = 0.99.
- **Quantiles:** parametric normal with z = -1.644854, -0.67449, 0,
  +0.67449, +1.644854. The outer tails (0.05, 0.95) are widened by
  TAIL_MULT_OUTER = 2.75.
- **Spread floor:** (q95 - q05) / 2 must be at least 0.02; if smaller,
  widen symmetrically around the median, re-clamp to [0, 1], then enforce
  monotonicity across the five quantiles. (Tuned in the builder's own
  backtest: a 0.02 floor beat 0.01 and 0.005 on both the 6 most recent
  origins and all origins.)
- **Not used:** longshot shrinkage and empirical residual quantiles as
  the primary spread (both proven to hurt in earlier rounds).

## Running

Python 3, standard library only. Reads every `*.in` file in
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` for each
to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Deterministic: two runs on
the same input produce byte-identical output.
