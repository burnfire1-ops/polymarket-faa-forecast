"""Polymarket FAA outage-attribution forecast (make.fun bounty
polymarket-faa-attribution-02, model V2).

For every market in an input case, forecast the Yes price 240 minutes ahead,
one row per minute, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

Model: the median is the mean of the market's last 3 prices (no drift).
The spread comes from a per-market adaptive volatility: the winsorized
population stdev of the last 360 one-minute price diffs (a 6-hour window),
floored at 0.0005. The spread grows as h**0.45 with horizon h, slower than
a random walk, because prediction-market levels mean-revert. The outer
5%/95% quantiles use a 2.5x wider normal z (fat tails).

Stdlib only, deterministic, offline. Reads MAKEFUN_INPUT_DIR (default
/input), writes MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import os
import statistics

HORIZON = 240
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# Normal z-values for the five quantile levels.
ZS = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
# Tail multipliers: the outer 5% / 95% quantiles get a wider z.
MULTS = [2.5, 1.0, 1.0, 1.0, 2.5]

# Volatility window: the last 360 one-minute diffs (6 hours).
SIGMA_WINDOW = 360
SIGMA_FALLBACK = 0.0025
SIGMA_FLOOR = 0.0005
# If the median absolute diff is 0, use this as the winsorization scale.
ZERO_MEDIAN_SCALE = 0.002
# Spread growth exponent: slower than sqrt(h) (random walk).
GROWTH_EXPONENT = 0.45


def clamp_round(v):
    """Clamp v into [0, 1] and round to 6 decimals (no negative zero)."""
    return round(min(1.0, max(0.0, v)) * 1e6) / 1e6 + 0.0


def robust_sigma(prices):
    """Winsorized stdev of the market's recent 1-minute price diffs.

    Keeps only the last SIGMA_WINDOW diffs, clips them at 5x the median
    absolute diff, and returns their population stdev. Falls back to
    SIGMA_FALLBACK when there are 0 or 1 diffs; floors the result at
    SIGMA_FLOOR so a flat market still gets a nonzero spread.
    """
    diffs = [b - a for a, b in zip(prices, prices[1:])]
    diffs = diffs[-SIGMA_WINDOW:]
    if len(diffs) <= 1:
        return SIGMA_FALLBACK
    abs_diffs = sorted(abs(d) for d in diffs)
    m = statistics.median(abs_diffs)
    if m == 0:
        m = ZERO_MEDIAN_SCALE
    # A 5x cap clips one-off price jumps while leaving the typical diff
    # scale untouched, so a single outlier cannot inflate the sigma.
    cap = 5.0 * m
    winsorized = [min(cap, max(-cap, d)) for d in diffs]
    sigma = statistics.pstdev(winsorized)
    return max(sigma, SIGMA_FLOOR)


def forecast_market(prices):
    """Return the 240 x 5 quantile rows for one market's price series."""
    if not prices:
        raise ValueError("market has no prices")
    center = sum(prices[-3:]) / len(prices[-3:])
    sigma = robust_sigma(prices)
    rows = []
    for h in range(1, HORIZON + 1):
        spread = sigma * (h ** GROWTH_EXPONENT)
        row = [
            clamp_round(center + ZS[i] * MULTS[i] * spread)
            for i in range(len(QUANTILES))
        ]
        rows.append(row)
    return rows


def process_case(path):
    """Read one .in case file and return the forecast dict for it."""
    with open(path, "r", encoding="utf-8") as f:
        case = json.load(f)
    if case.get("quantiles") != QUANTILES:
        raise ValueError(
            f"{path}: quantiles must be exactly {QUANTILES}, "
            f"got {case.get('quantiles')}"
        )
    forecasts = []
    for market in case["markets"]:
        forecasts.append(
            {"id": market["id"], "q": forecast_market(market["prices"])}
        )
    return {"forecasts": forecasts}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        result = process_case(os.path.join(input_dir, name))
        out_name = name[:-3] + ".out"
        with open(
            os.path.join(output_dir, out_name), "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
