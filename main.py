#!/usr/bin/env python3
"""polymarket-faa-attribution-02 entry: short-horizon quantile forecaster.

Model (backtested on the kit's six practice cases, ~-4% pinball loss vs the
house persistence baseline):
  - Median: persistence. The last price. Trend/drift extrapolation HURTS
    badly on these markets (1-min returns are slightly mean-reverting and
    the 4h trend does not continue), so no drift term.
  - Spread: per-market adaptive volatility. Robust (winsorized) stdev of the
    1-minute price diffs over the last 6 hours, floored at 0.0005. A 6h
    window adapts to volatility clustering better than 24h.
  - Tails: the 5% / 95% quantiles use a 2.5x wider normal z than the
    25% / 75% ones. Prediction-market prices jump on news, so the true
    conditional distribution is far fatter-tailed than a normal; the plain
    normal tails are far too narrow and pay heavily on jumps. The 25%/75%
    quantiles keep the plain normal z.

  q_tau(h) = clamp(x + z_tau' * sigma * sqrt(h), 0, 1), rounded to 6 dp,
  where x is the market's last price, sigma the adaptive volatility,
  z' = z * 2.5 for tau in {0.05, 0.95} and z otherwise.

Reads every <id>.in in the input directory (MAKEFUN_INPUT_DIR or /input)
and writes <id>.out to the output directory (MAKEFUN_OUTPUT_DIR or /output).
Python 3.13 stdlib only. No network, no clock, no randomness: deterministic.
"""
import json
import math
import os
import statistics
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
# Tail widening: 5% and 95% get 2.5x the normal z; 25%/75%/median unchanged.
Z_MULT = [2.5, 1.0, 1.0, 1.0, 2.5]
# Volatility window: last 6 hours of 1-minute diffs.
SIG_WINDOW = 360
# Floor on sigma so a flat market still gets a nonzero spread.
SIG_FLOOR = 0.0005


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def robust_sigma(diffs, window=SIG_WINDOW):
    """Winsorized stdev of the last `window` 1-minute diffs.

    Jumps are winsorized at 5x the median absolute diff so a single news
    jump does not inflate the typical-volatility estimate; the fat tails
    are handled separately by Z_MULT.
    """
    d = diffs[-window:] if len(diffs) > window else diffs
    if not d:
        return 0.0025
    med = statistics.median([abs(v) for v in d]) or 0.002
    cap = 5.0 * med
    dw = [max(-cap, min(cap, v)) for v in d]
    s = statistics.pstdev(dw) if len(dw) > 1 else 0.0025
    return max(s, SIG_FLOOR)


def forecast(prices, horizon):
    """240 rows of five quantiles for one market."""
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    sigma = robust_sigma(diffs)
    z = [Z[i] * Z_MULT[i] for i in range(5)]
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        rows.append([price6(x + zi * spread) for zi in z])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        quantiles = case["quantiles"]
        if len(quantiles) != len(QUANTILES) or any(
            q != QUANTILES[i] for i, q in enumerate(quantiles)
        ):
            raise ValueError(
                "%s asks for quantiles %s, not %s" % (fname, quantiles, QUANTILES)
            )
        forecasts = [
            {"id": m["id"], "q": forecast(m["prices"], horizon)}
            for m in case["markets"]
        ]
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
