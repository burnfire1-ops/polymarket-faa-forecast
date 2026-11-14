# Polymarket FAA Attribution Forecaster

Entry for the make.fun bounty `polymarket-faa-attribution-02`: forecast the
Yes price of every market in Polymarket's FAA outage attribution family,
minute by minute, four hours ahead, as five quantiles per minute.

## Model

`main.py` (Python 3.13, standard library only, deterministic, no network)
reads one case at a time from the input directory and writes one forecast
file per case.

- **Median: persistence.** The forecast median at every horizon is the
  market's last price. There is no drift term: drift extrapolations scored
  far worse than persistence in the backtest, because 1-minute returns on
  these markets are slightly mean-reverting and a 4-hour trend does not
  carry forward.
- **Spread: per-market adaptive volatility.** Sigma is the winsorized
  standard deviation of the 1-minute price diffs over the last 6 hours,
  floored at 0.0005. Winsorizing at 5x the median absolute diff keeps a
  single news jump from inflating the volatility estimate for hours.
  A 6-hour window adapts to volatility clustering better than a 24-hour
  one.
- **Tails: fat.** The 5% and 95% quantiles use a 2.5x wider normal z
  than the 25% and 75% ones. Prediction-market prices jump on news, so
  the true conditional distribution is far fatter-tailed than a normal,
  and plain normal tails are far too narrow.

`q = clamp(x + z' * sigma * sqrt(h), 0, 1)`, rounded to 6 decimals, where
`x` is the last price, `sigma` the adaptive volatility, `h` the horizon
in minutes, and `z'` the normal quantile scaled 2.5x at the 5% and 95%
levels.

## Backtest

On the kit's six practice cases (sub-checkpoints cut inside each case's
own 72-hour history, scored by mean pinball loss against the case's own
later prices):

- house persistence baseline: 1.127348
- this forecaster: about 1.081 (about 4% better than the baseline)

Drift variants scored 1.52-1.86, far worse than persistence. The 6h
volatility window was chosen over 24h because it adapts faster to
volatility clustering. The official practice score is computed
server-side only. Outputs are deterministic: the same input always
produces the same output.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT
