#!/usr/bin/env python3
"""Forecast Polymarket Yes prices as five quantiles, 240 minutes ahead.

Entry for make.fun bounty polymarket-faa-attribution-02.

Model, per market:
  - Median: persistence. The forecast median at every horizon is the
    market's last observed price. No drift or trend term: on these
    markets the 1-minute returns are slightly mean-reverting and a
    4-hour trend does not continue, so extrapolating it hurts.
  - Spread: adaptive volatility. Take the consecutive 1-minute price
    differences, keep the last 360 (6 hours), and winsorize them at
    5x the median absolute diff so one news jump cannot dominate the
    estimate. Sigma is the population standard deviation of the
    winsorized diffs, floored at 0.0005.
  - Tails: fat. The 5% and 95% quantiles use a 2.5x wider normal z
    than the 25% / 75% ones, because prediction-market prices jump on
    news and plain normal tails are far too narrow.

  q(h) = clamp(x + z * mult * sigma * sqrt(h), 0, 1), rounded to 6 dp,
  where x is the last price and h is the horizon in minutes.

Reads every <id>.in in MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to MAKEFUN_OUTPUT_DIR (default /output). Python 3.13 stdlib
only. No network, no clock, no randomness: deterministic.
"""
import json
import math
import os
import statistics
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
Z_MULT = [2.5, 1.0, 1.0, 1.0, 2.5]
SIG_WINDOW = 360
SIG_FLOOR = 0.0005
SIG_EMPTY = 0.0025


def price6(value):
    """Clamp a price to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, value)) * 1e6) / 1e6 + 0.0


def robust_sigma(diffs):
    """Winsorized stdev of the last SIG_WINDOW 1-minute diffs.

    The diffs are clipped to 5x the median absolute diff before the
    standard deviation is taken, so a single large jump does not set
    the volatility for the next hours; the fat tails are handled
    separately by Z_MULT.
    """
    window = diffs[-SIG_WINDOW:] if len(diffs) > SIG_WINDOW else diffs
    if not window:
        return SIG_EMPTY
    med = statistics.median([abs(v) for v in window]) or 0.002
    cap = 5.0 * med
    # 5x the median: wide enough that ordinary volatility passes through
    # untouched, tight enough that one news jump cannot set the spread.
    clipped = [max(-cap, min(cap, v)) for v in window]
    if len(clipped) <= 1:
        return SIG_EMPTY
    return max(statistics.pstdev(clipped), SIG_FLOOR)


def forecast(prices, horizon):
    """One market: horizon rows of five quantiles each."""
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    sigma = robust_sigma(diffs)
    scaled_z = [Z[i] * Z_MULT[i] for i in range(5)]
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        rows.append([price6(x + z * spread) for z in scaled_z])
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
