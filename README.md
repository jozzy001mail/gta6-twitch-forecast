# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jozzy001mail/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jozzy001mail/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour. Built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty, $250 in USDC.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it works

`main.py` (Python 3.13, standard library only) reads one case file and writes, for each of hours 2–24 of the release day, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's Twitch "Grand Theft Auto VI" category viewer count.

The model works on the log scale, anchored on the case's hour-1 print. GTA VI unlocks at midnight local time, rolling country by country, so hour 1 already holds Europe/UK/NZ/Australia and the count climbs as the Americas unlock, peaking in US prime time. For each hour the median log-ratio to hour 1 is a linear function of the (log) hour-1 level itself, fit at run time from the case's own comparables — the rolling-midnight console launches (ELDEN RING, Ghost of Yōtei, Zelda: Tears of the Kingdom, and the rest) — with a per-hour residual spread setting the quantile widths. The key fact the fit captures: the bigger hour 1 is, the less upside is left (small launches peak ~3.5× hour 1; ELDEN RING, hour 1 ~875k, peaked in hour 1).

Leave-one-out cross-validation on the ten rolling launches, scored with the competition's own pinball loss on the log scale: **0.275**, against **0.432** for the house baseline persistence forecast. A hard-coded fallback table of the same fit is included in case a case holds too few usable comparables. Deterministic, offline, no model calls.

---

<sub>Entered on [make.fun](https://make.fun/?ref=112z8vbj&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
