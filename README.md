# gta6-twitch-ccu-forecast-01-entry

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Rahulboss7788/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Rahulboss7788/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast, hour by hour, how many people watch Grand Theft Auto VI live on Twitch over the 24 UTC hours of its release day (2026-11-19), from everything recorded up to the end of the day's first hour. Entry for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty ($250 in USDC).

## Model

A launch-day **shape model**, not persistence. For every past big launch shipped inside the case file, the release-day hourly curve is expressed as log-ratios to its own hour-1 print, ln(v_h / v_1). The forecast is `q_tau(h) = p1 * exp(m_h + z_tau * s_h)`, where `m_h` is a weighted median of the comparables' log-ratios at hour h and `s_h` a weighted spread of the same distribution.

A rolling-midnight console launch does not stay flat: hour 1 already holds the UK, Europe and Australasia unlocks, so the count first dips through the small hours, then ramps as the US East Coast unlocks at 05:00 UTC and the West Coast at 08:00 UTC, peaks in the evening (hours 19–22 UTC), and decays into the night. The pooled median curve captures that whole arc.

Each comparable is weighted by how much it resembles the target launch: tier (comparable 1.0, early-access-contaminated 0.35, weak 0.25), unlock shape (rolling midnight-local ×1.5, simultaneous global ×0.7), launch size (Gaussian kernel on ln hour-1 print, bandwidth 0.9 — GTA VI is expected to be the biggest launch on record), and pre-launch buzz (Gaussian kernel on the category's mean hourly viewers in the 24 h before release day, relative to hour 1). Launches whose hour 1 is pre-launch noise (under 3% of their release-day peak) are dropped, as is the GTA VI category's own pre-release history (not a launch).

The spread blends two robust estimators — the 90% range and the interquartile range of the weighted log-ratios — clipped to [0.25, 0.90] and smoothed across neighbouring hours.

Backtest (leave-one-out over the 29 usable comparables in the kit, scored with the contest's own log-scale pinball loss): **0.270 vs the house baseline's 0.360** on the same launches; typical miss 57.9% vs 83.5%; 90% coverage 0.868 vs 0.757.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic — two runs write byte-identical outputs. Reads every `*.in` file in the input directory and writes `<id>.out` for each: `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2–24.

---

<sub>Entered on [make.fun](https://make.fun/?ref=xt0p49wj&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
