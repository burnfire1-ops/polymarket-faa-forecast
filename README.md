# Polymarket FAA Attribution Forecaster

Entry for make.fun bounty `polymarket-faa-attribution-02`: forecast, minute by
minute and 4 hours ahead, the Yes price of every market of Polymarket's FAA
outage attribution family, as five quantiles a minute.

## Model

`main.py` (Python 3.13 stdlib only, no network, deterministic) reads one case
at a time from the input directory and writes one forecast file per case.

- **Median: persistence.** The last price. Trend/drift extrapolation hurts
  badly on these markets (1-minute returns are slightly mean-reverting and
  the 4-hour trend does not continue), so there is no drift term.
- **Spread: per-market adaptive volatility.** Robust (winsorized) standard
  deviation of the 1-minute price diffs over the last 6 hours, floored at
  0.0005. A 6-hour window adapts to volatility clustering better than 24h.
- **Tails: fat.** The 5% / 95% quantiles use a 2.5x wider normal z than the
  25% / 75% ones. Prediction-market prices jump on news, so the true
  conditional distribution is far fatter-tailed than a normal; plain normal
  tails are far too narrow and pay heavily on jumps.

`q_tau(h) = clamp(x + z_tau' * sigma * sqrt(h), 0, 1)`, rounded to 6 decimals,
where `x` is the market's last price, `sigma` the adaptive volatility, and
`z'` the normal quantile scaled 2.5x at tau 0.05 and 0.95.

## Backtest

On the kit's six practice cases (sub-checkpoints cut inside each case's own
72-hour history, scored by mean pinball loss against the case's own later
prices):

- house persistence baseline: 1.127348
- this forecaster: ~1.081 (about 4% better than the baseline)

Drift variants scored far worse (1.52-1.86), so the median stays at the last
price. The official practice score is computed server-side only. Outputs are
deterministic: the same input always produces the same output.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT
