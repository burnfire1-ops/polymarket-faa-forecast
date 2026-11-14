"""Polymarket FAA forecast, model V4.

Reads one case file <id>.in from MAKEFUN_INPUT_DIR (default /input) and
writes <id>.out to MAKEFUN_OUTPUT_DIR (default /output) with one forecast
row per market: 240 horizons x 5 quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

Model, in brief:
  * Median at every horizon: mean of the market's last 3 prices.
  * Volatility: GARCH(1,1) on 1-minute price differences (last <=360),
    fitted by grid search maximising the Gaussian log-likelihood, with
    omega set so the long-run variance matches the sample variance.
  * Multi-step variance from the standard GARCH formula, with a decoupled
    horizon split: horizons 1-60 use the fitted persistence, horizons
    61-240 relax from the h=60 variance toward the long-run variance at
    a second (far) persistence rate.
  * Quantiles: parametric normal, outer quantiles widened by a tail
    multiplier; spread floor: (q95 - q05) / 2 >= 0.01 at every horizon.

Standard library only. Deterministic: no randomness, no network, no
wall-clock dependence.
"""

import json
import math
import os
import sys

# --- Calibrated parameters (chosen by backtest over the practice cases) ---
# GARCH grid: alpha from 0.02 to 0.30 in steps of 0.02; beta in steps of
# 0.05 subject to alpha + beta < 0.97.
GARCH_ALPHA_MIN = 0.02
GARCH_ALPHA_MAX = 0.30
GARCH_ALPHA_STEP = 0.02
GARCH_BETA_STEP = 0.05
GARCH_PERSISTENCE_CAP = 0.97
# Decoupled horizon forecasting: far-horizon (h > 60) persistence.
# Backtest (186 market-origin pairs, practice cases): far persistence
# 0.90/0.99/1.00 were indistinguishable on the decision metric
# (recent-6 origins pinball 0.003601-0.003603), so 0.99 is kept as the
# central value: the far variance relaxes slowly toward long-run.
FAR_PERSISTENCE = 0.99
# Near-horizon persistence: None means use the fitted alpha + beta.
NEAR_PERSISTENCE_OVERRIDE = None
# Tail multiplier for the outer quantiles (0.05 and 0.95).
# Backtest: 2.50/2.60/2.75 tie on the recent-6 decision metric
# (0.003601-0.003602) while 2.75 improves all-origins pinball
# (0.018099 vs 0.018190 at 2.50); 2.90 already worsens recent-6
# (0.003608). 2.75 is the elbow, so it is hardcoded.
TAIL_MULT_OUTER = 2.75

Z_VALUES = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
SPREAD_FLOOR_HALF = 0.01
FALLBACK_SIGMA_FLOOR = 0.0005
MAX_DIFFS = 360
MIN_DIFFS_FOR_GARCH = 30


def _clamp(x, lo=0.0, hi=1.0):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _mean(values):
    return sum(values) / len(values)


def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _fallback_sigma(diffs):
    """Constant sigma from capped differences (see module docstring)."""
    abs_diffs = [abs(d) for d in diffs]
    med = _median(abs_diffs)
    cap = 5.0 * med if med > 0.0 else 0.002
    capped = [d if -cap <= d <= cap else (-cap if d < 0.0 else cap)
              for d in diffs]
    mu = _mean(capped)
    var = _mean([(c - mu) ** 2 for c in capped])
    sigma = math.sqrt(var) if var > 0.0 else 0.0
    return max(sigma, FALLBACK_SIGMA_FLOOR)


def _fit_garch(diffs):
    """Fit GARCH(1,1) to the differences.

    Returns (omega, alpha, beta, var_1, long_run_var) or None when the
    fit is not usable (too few differences, zero variance).
    """
    n = len(diffs)
    if n < MIN_DIFFS_FOR_GARCH:
        return None
    mu = _mean(diffs)
    shocks = [d - mu for d in diffs]
    sq = [e * e for e in shocks]
    sample_var = _mean(sq)
    if sample_var <= 0.0:
        return None

    best = None  # (loglik, alpha, beta, var_1)
    alpha = GARCH_ALPHA_MIN
    while alpha <= GARCH_ALPHA_MAX + 1e-12:
        beta = GARCH_BETA_STEP
        while alpha + beta < GARCH_PERSISTENCE_CAP:
            omega = sample_var * (1.0 - alpha - beta)
            var = sample_var
            ll = 0.0
            ok = True
            for e2 in sq:
                if var <= 0.0:
                    ok = False
                    break
                ll += -0.5 * (math.log(var) + e2 / var)
                var = omega + alpha * e2 + beta * var
            if ok and (best is None or ll > best[0]):
                best = (ll, alpha, beta, var)
            beta += GARCH_BETA_STEP
        alpha += GARCH_ALPHA_STEP

    if best is None:
        return None
    _ll, alpha, beta, var_1 = best
    omega = sample_var * (1.0 - alpha - beta)
    long_run_var = omega / (1.0 - alpha - beta)
    return (omega, alpha, beta, var_1, long_run_var)


def _sigma_path(prices, horizon):
    """Return (median, list of sigma_h for h = 1..horizon)."""
    if prices:
        median = _mean(prices[-3:])
    else:
        median = 0.5
    diffs_all = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    diffs = diffs_all[-MAX_DIFFS:]

    if not diffs:
        sigma_const = FALLBACK_SIGMA_FLOOR
        return median, [sigma_const] * horizon

    fit = _fit_garch(diffs)
    if fit is None:
        sigma_const = _fallback_sigma(diffs)
        return median, [sigma_const] * horizon

    _omega, alpha, beta, var_1, long_run_var = fit
    pers_near = (NEAR_PERSISTENCE_OVERRIDE
                 if NEAR_PERSISTENCE_OVERRIDE is not None
                 else alpha + beta)
    if pers_near >= 1.0:
        pers_near = 0.999999
    if pers_near < 0.0:
        pers_near = 0.0

    sigmas = []
    var_h = var_1
    # h = 1 uses var_1 directly; each later step relaxes toward long-run.
    for h in range(1, horizon + 1):
        if h == 1:
            var_h = var_1
        elif h <= 60:
            var_h = long_run_var + pers_near * (var_h - long_run_var)
        else:
            var_h = long_run_var + FAR_PERSISTENCE * (var_h - long_run_var)
        if var_h < 0.0:
            var_h = 0.0
        sigmas.append(math.sqrt(var_h))
    return median, sigmas


def forecast_market(prices, horizon):
    """Forecast one market: list of `horizon` rows of 5 quantiles."""
    median, sigmas = _sigma_path(prices, horizon)
    mults = (TAIL_MULT_OUTER, 1.0, 1.0, 1.0, TAIL_MULT_OUTER)
    rows = []
    for h in range(horizon):
        sigma = sigmas[h] if h < len(sigmas) else sigmas[-1]
        row = []
        for z, mult in zip(Z_VALUES, mults):
            row.append(_clamp(median + z * mult * sigma))
        # Spread floor: (q95 - q05) / 2 >= 0.01, widen symmetrically
        # around the median.
        if (row[4] - row[0]) / 2.0 < SPREAD_FLOOR_HALF:
            row[0] = _clamp(median - SPREAD_FLOOR_HALF)
            row[4] = _clamp(median + SPREAD_FLOOR_HALF)
        # Re-enforce monotonicity within the row.
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append([round(v, 6) for v in row])
    return rows


def forecast_case(case):
    horizon = int(case.get("horizon_minutes", 240))
    forecasts = []
    for market in case.get("markets", []):
        prices = [float(p) for p in market.get("prices", [])]
        forecasts.append({
            "id": market["id"],
            "q": forecast_market(prices, horizon),
        })
    return {"forecasts": forecasts}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    in_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    if not in_files:
        raise SystemExit("no .in file found in %s" % input_dir)
    in_name = in_files[0]
    case_id = in_name[:-3]
    with open(os.path.join(input_dir, in_name), "r", encoding="utf-8") as fh:
        case = json.load(fh)
    result = forecast_case(case)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, case_id + ".out")
    text = json.dumps(result, separators=(",", ":"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
