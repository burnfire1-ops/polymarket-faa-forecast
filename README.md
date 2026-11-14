# Polymarket FAA Forecast — Model V4

Forecasts 1-minute Yes prices for each market over a 240-minute horizon,
as quantiles at levels 0.05, 0.25, 0.5, 0.75 and 0.95.

## Model

- **Median:** the mean of the market's last 3 prices, flat across all
  horizons (no drift). If fewer than 3 prices exist, the mean of those.
- **Volatility — GARCH(1,1):** the 1-minute price differences are modelled
  with a GARCH(1,1) process. Omega, alpha and beta are estimated from the
  market's own history (last up to 360 differences) by a grid search over
  alpha (0.02–0.30) and beta (alpha + beta < 0.97) maximising the Gaussian
  log-likelihood, with omega set so the long-run variance matches the
  sample variance of the differences. If there are fewer than 30
  differences, the fit fails, or all differences are zero, a fallback is
  used: the standard deviation of the differences capped at 5 times their
  median absolute value (0.002 cap scale if that median is 0), floored
  at 0.0005, as a constant sigma.
- **Multi-step variance:** the h-step variance relaxes toward the
  long-run variance omega / (1 − alpha − beta) with the standard GARCH
  formula, var_h = long_run_var + (alpha + beta)^(h−1) · (var_1 − long_run_var).
- **Decoupled horizon forecasting (DHF):** horizons 1–60 use the fitted
  GARCH persistence; horizons 61–240 continue from the h=60 variance but
  relax toward the long-run variance at a second, separately calibrated
  persistence rate (hardcoded in `main.py`).
- **Quantiles:** parametric normal — quantile = clamp(median + z_tau ·
  mult_tau · sigma_h, 0, 1), with z = −1.644854, −0.67449, 0, 0.67449,
  1.644854 and mult_tau = 2.75 for the outer quantiles (0.05, 0.95),
  1.0 for the inner ones.
- **Spread floor:** at every horizon (q95 − q05) / 2 is at least 0.01;
  narrower rows are widened symmetrically around the median, then
  re-clamped to [0, 1] with monotonicity re-enforced within the row.
- No longshot shrinkage of the median toward 0.5, and no empirical
  residual quantiles.

## Calibration (backtest over the practice cases)

Origins cut every 240 minutes before each case checkpoint (186
market-origin pairs, deduplicated across cases). Mean pinball loss:
**0.003602 over the 6 most recent origins** (the decision metric) and
0.018099 over all origins — the older origins include a few very
volatile slots (per-slot pinball up to 0.072). Calibrated values
hardcoded in `main.py`: far persistence 0.99 (0.90/0.99/1.00 were
indistinguishable on recent-6), outer tail multiplier 2.75 (ties 2.50
on recent-6, better on all origins; 2.90 already worsens recent-6),
near persistence = fitted GARCH alpha + beta, fine GARCH grid
(alpha step 0.02, beta step 0.05).

## Running

    MAKEFUN_INPUT_DIR=<dir with <id>.in> MAKEFUN_OUTPUT_DIR=<dir> python3 main.py

Writes `<id>.out` (UTF-8 JSON, `{"forecasts": [{"id": ..., "q": [[...], ...]}]}`).

Python 3 standard library only; deterministic — the same input always
produces byte-identical output, with no network or wall-clock dependence.
