#!/usr/bin/env python3
"""Polymarket FAA Forecast -- model V5.

Forecasts each market's Yes price at every minute of the next 240 minutes
as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

Model (V4 base, plus the V5 near-persistence override):
  * Median: mean of the market's last 3 prices, flat across horizons.
  * Volatility: GARCH(1,1) fitted on 1-minute price differences (at most
    the last 360 diffs). Grid search over alpha and beta maximising the
    Gaussian log-likelihood; omega is set so the long-run variance equals
    the sample variance of the diffs. Fallback (fewer than 30 diffs, a
    failed fit, or all-zero diffs): the standard deviation of the diffs,
    capped at 5x the median |diff| (0.002 used as the scale when that
    median is 0), floored at 0.0005.
  * Multi-step variance: var_h = long_run_var
        + persistence^(h-1) * (var_1 - long_run_var).
    V5 change: for near horizons (h <= 60) the persistence is the FIXED
    value NEAR_PERSISTENCE = 0.98, not the fitted alpha+beta. For far
    horizons (h > 60) the variance relaxes from the h=60 variance at a
    second persistence FAR_PERSISTENCE = 0.99.
  * Quantiles: parametric normal with z = -1.644854, -0.67449, 0,
    +0.67449, +1.644854. The outer tails (0.05, 0.95) are widened by
    TAIL_MULT_OUTER.
  * Spread floor: (q95 - q05) / 2 must be at least SPREAD_FLOOR; widen
    symmetrically around the median, re-clamp to [0, 1], then enforce
    monotonicity across the five quantiles.
  * Not used: longshot shrinkage, empirical residual quantiles.

Standard library only. Deterministic: no randomness, no wall-clock use,
fixed iteration orders, so two runs on the same input are byte-identical.

Usage: MAKEFUN_INPUT_DIR (default /input) holds <id>.in files;
MAKEFUN_OUTPUT_DIR (default /output) receives <id>.out files.
"""

import json
import math
import os
import sys

# --- Tuned constants (V5) -------------------------------------------------
NEAR_PERSISTENCE = 0.98   # fixed persistence for h <= 60 (the V5 change)
FAR_PERSISTENCE = 0.99    # persistence relaxing from the h=60 variance
TAIL_MULT_OUTER = 2.75    # multiplier on the 0.05 / 0.95 quantiles
SPREAD_FLOOR = 0.02       # minimum (q95 - q05) / 2 (tuned: 0.02 beats
                          # 0.01 and 0.005 on both recent-6 and all
                          # origins in the builder's own backtest)

# --- Fixed model settings --------------------------------------------------
Z_VALUES = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
HORIZON = 240
MAX_DIFFS = 360
MIN_DIFFS = 30
SIGMA_FLOOR = 0.0005
ZERO_MEDIAN_SCALE = 0.002


def _median_last3(prices):
    """Mean of the last three prices (or fewer when less history exists)."""
    tail = prices[-3:]
    return sum(tail) / len(tail)


def _fallback_sigma(diffs):
    """Fallback volatility: std of diffs, capped and floored."""
    if not diffs:
        return SIGMA_FLOOR
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) * (d - mean) for d in diffs) / n
    sd = math.sqrt(var) if var > 0.0 else 0.0
    ordered = sorted(abs(d) for d in diffs)
    if n % 2 == 1:
        med_abs = ordered[n // 2]
    else:
        med_abs = 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    scale = med_abs if med_abs > 0.0 else ZERO_MEDIAN_SCALE
    sd = min(sd, 5.0 * scale)
    return max(sd, SIGMA_FLOOR)


def _garch_fit(diffs):
    """Fit GARCH(1,1) by grid search; return (var_1, long_run_var) or None.

    Innovations are the demeaned diffs. The conditional variance starts
    at the sample variance and recurses as
        s2 <- omega + alpha * eps^2 + beta * s2
    with omega = sample_var * (1 - alpha - beta), so the long-run
    variance is exactly the sample variance. The best (alpha, beta) is
    the one maximising the Gaussian log-likelihood; the returned var_1
    is the one-step-ahead variance after the last observation.
    """
    n = len(diffs)
    mean = sum(diffs) / n
    eps = [d - mean for d in diffs]
    var = sum(e * e for e in eps) / n
    if not (var > 0.0) or not math.isfinite(var):
        return None
    eps2 = [e * e for e in eps]
    log = math.log
    log2pi = log(2.0 * math.pi)
    best_ll = None
    best_var1 = None
    # alpha = 0.02 .. 0.30 step 0.02 ; beta = 0.00 .. step 0.05,
    # keeping alpha + beta < 0.97. Integer counters keep the grid exact.
    for ai in range(1, 16):
        alpha = ai * 0.02
        for bi in range(0, 20):
            beta = bi * 0.05
            if alpha + beta >= 0.97:
                break
            omega = var * (1.0 - alpha - beta)
            s2 = var
            ll = -0.5 * n * log2pi
            ok = True
            for e2 in eps2:
                if s2 <= 0.0 or not math.isfinite(s2):
                    ok = False
                    break
                ll -= 0.5 * (log(s2) + e2 / s2)
                s2 = omega + alpha * e2 + beta * s2
            if not ok:
                continue
            if best_ll is None or ll > best_ll:
                best_ll = ll
                best_var1 = s2
    if best_ll is None:
        return None
    return (best_var1, var)


def forecast_market(prices, tail_mult=TAIL_MULT_OUTER, floor=SPREAD_FLOOR,
                    far_persist=FAR_PERSISTENCE,
                    near_persist=NEAR_PERSISTENCE):
    """Return 240 rows of five quantiles for one market's price list."""
    if prices:
        med = _median_last3(prices)
    else:
        med = 0.5
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    diffs = diffs[-MAX_DIFFS:]
    fit = None
    if len(diffs) >= MIN_DIFFS and any(d != 0.0 for d in diffs):
        fit = _garch_fit(diffs)
    if fit is None:
        sig = _fallback_sigma(diffs)
        var1 = long_run = sig * sig
    else:
        var1, long_run = fit
    sqrt = math.sqrt
    rows = []
    var60 = None
    for h in range(1, HORIZON + 1):
        if h <= 60:
            var_h = long_run + (near_persist ** (h - 1)) * (var1 - long_run)
            if h == 60:
                var60 = var_h
        else:
            var_h = long_run + (far_persist ** (h - 60)) * (var60 - long_run)
        sd = sqrt(var_h) if var_h > 0.0 else 0.0
        q05 = med + Z_VALUES[0] * sd * tail_mult
        q25 = med + Z_VALUES[1] * sd
        q50 = med
        q75 = med + Z_VALUES[3] * sd
        q95 = med + Z_VALUES[4] * sd * tail_mult
        half = 0.5 * (q95 - q05)
        if half < floor:
            q05 = med - floor
            q95 = med + floor
        qs = (q05, q25, q50, q75, q95)
        qs = [min(1.0, max(0.0, q)) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append([round(q, 6) for q in qs])
    return rows


def run_case(case):
    """Forecast every market of one parsed case object."""
    horizon = case.get("horizon_minutes", HORIZON)
    forecasts = []
    for market in case["markets"]:
        prices = market.get("prices") or []
        rows = forecast_market(prices)
        if len(rows) != horizon:  # horizon is 240 in every real case
            rows = rows[:horizon]
        forecasts.append({"id": market["id"], "q": rows})
    return {"forecasts": forecasts}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        case_id = fn[:-3]
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = run_case(case)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
