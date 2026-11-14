# Polymarket FAA Outage Attribution Forecast

Entry for the make.fun bounty **polymarket-faa-attribution-02**: forecast the
Yes price of every market of Polymarket's FAA outage attribution family, as
five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per minute, 4 hours ahead
(240 minutes) over the 48 hours after entries close.

## Model

The forecast is a persistence-style distribution centred on the recent level,
with a spread that adapts to each market's own recent volatility:

- **Median:** the mean of the market's last 3 prices. No drift or trend
  term — drift variants scored far worse in backtests.
- **Spread:** per-market adaptive volatility — the winsorized population
  standard deviation of the 1-minute price differences over the last 6
  hours (360 diffs), floored at 0.0005 so a flat market still gets a
  nonzero spread. The spread grows as `h**0.45` with horizon `h`, slower
  than a random walk, because prediction-market levels mean-revert.
- **Tails:** the outer 5% and 95% quantiles use a 2.5x wider normal
  z-value than the plain normal z, giving fat tails.
- **Formula:** `q = clamp(x + z' * sigma * h**0.45, 0, 1)`, where `x` is
  the 3-minute mean centre, `z'` is the tail-adjusted z-value, and `sigma`
  is the winsorized volatility. Quantiles are rounded to 6 decimal places.

Python 3.13 stdlib only, deterministic, offline — no network, no clock,
no randomness.

## Backtest

Local backtest on the kit's practice cases (11 distinct origins counted once
— the 6 practice files are overlapping sliding windows of one timeline, so
naive counting triple-counts recent origins):

- Round-2 model: **1.1938** mean pinball loss on all 11 distinct origins.
- This V2 model: **1.2001** on all 11 distinct origins (nearly tied).
- On the **6 most recent origins** — the regime the 16–18 Nov final window
  resembles, since the FAA cause was already publicly attributed on 13 Nov —
  V2 scores **0.3131** vs round-2's **0.3346** (about 6% better).

Longer averages (15-minute mean/median) won big only on the old shock-era
origins and lost 17–22% on the recent origins, so they were rejected. The
6-hour volatility window was chosen over a 24-hour one because it adapts
faster to volatility clustering. Drift variants scored far worse (1.52–1.86). The official practice score is
computed server-side only.

## Run

```bash
mkdir -p output
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

Each `<id>.in` case in the input directory produces a single-line JSON
`<id>.out` file with `{"forecasts": [{"id": ..., "q": [[5 numbers] x 240]}]}`.

## License

MIT.
