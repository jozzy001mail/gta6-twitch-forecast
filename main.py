"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun
gta6-twitch-ccu-forecast-01). Python 3.13, standard library only.

Model
-----
Work on the log scale and anchor on the case's hour-1 print p1. For each
hour h = 2..24 the median log-ratio to hour 1 is modelled as a linear
function of the (log) hour-1 level itself:

    ln(v_h / p1)  ~  a_h + b_h * ln(p1)

with a per-hour residual spread s_h. The quantiles are then

    q_tau(h) = p1 * exp(a_h + b_h * ln(p1) + z_tau * s_h)

where z_tau is the standard normal quantile at each scored level.

Two variance guards, both validated by leave-one-out cross-validation:
the regression's median is shrunk 15% toward the plain median log-ratio
profile of the same launches (the ten-point line is noisy), and the slope
is clipped at -0.30 so that a GTA-sized hour 1 -- above the largest
hour 1 in the training set (~875k) -- does not extrapolate the peak
ratio down without bound.

Why this shape: GTA VI unlocks at midnight local time, rolling country by
country (New Zealand 11:00Z Nov 18, Europe 23:00Z, UK 00:00Z, US East
05:00Z, US West 08:00Z), so hour 1 already holds Europe/UK/NZ/Australia
and the count climbs as the Americas unlock, peaking in US prime time
(hours 15-21 UTC) before fading. The house baseline (persistence) misses
that shape badly. The coefficients are fit at run time from the case's own
`comparables`: the rolling-midnight console launches in it (ten in the
practice cases, plus the four practice launches in the final case).
The fit is a plain least-squares line per hour over those launches'
release-day hourly counts, which is exactly the quantity being forecast.

The key empirical fact the fit captures: the bigger hour 1 is, the less
upside is left. Small launches (hour 1 ~ 27k) peak at ~3.5x hour 1; the
biggest (ELDEN RING, hour 1 ~ 875k) peaked in hour 1 itself. The fitted
slopes run from ~0 at hour 2 to about -0.37 at hours 19-22, and the
residual spread is ~0.11 early and ~0.4-0.6 later.

Leave-one-out cross-validation on the ten rolling launches (each held-out
launch forecast from the other nine, pinball loss on the log scale, the
competition's own scoring): 0.275, against 0.432 for the house baseline
persistence forecast on the same launches (the unshrunk regression alone
scores 0.282). On all 37 usable comparables the model is also better
than persistence (0.79 vs 0.93), but the rolling subset is the relevant
one: GTA VI is a rolling-midnight console launch.

A fallback table (the same fit on the ten rolling launches, hard-coded
below) is used if the case holds fewer than six usable rolling
comparables, so the program still forecasts. Everything is deterministic:
same case in, same bytes out. No network, no randomness, no model calls.

Run: python3 main.py reads $MAKEFUN_INPUT_DIR (else /input) and writes
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Rolling-midnight, console-only (or console-first) single-player launches
# whose release-day shape matches GTA VI's unlock pattern. The four
# practice launches are included: they are present in the final case's
# comparables and are the same kind of launch.
ROLLING_NAMES = {
    "Ghost of Y\u014dtei",
    "FINAL FANTASY VII REBIRTH",
    "Stellar Blade",
    "Saros",
    "The Last of Us Part II",
    "Ghost of Tsushima",
    "The Legend of Zelda: Tears of the Kingdom",
    "ELDEN RING",
    "Monster Hunter Wilds",
    "Resident Evil Requiem",
    "Marvel's Wolverine",
    "Red Dead Redemption 2",
    "God of War Ragnar\u00f6k",
    "Marvel's Spider-Man 2",
}

# Fallback (a_h, b_h, sigma_h, median_log_ratio_h) per hour, fit on the ten
# rolling launches above (excluding the four practice launches, which the
# practice cases' comparables do not contain). Used only if the case holds
# too few usable rolling comparables to fit at run time. Slopes already
# clipped at -0.30, as in fit_coefficients.
FALLBACK = {
    2: (-0.218941, 0.003506, 0.111380, -0.187967),
    3: (-1.033249, 0.050969, 0.293803, -0.347202),
    4: (-0.257452, -0.019592, 0.241214, -0.408276),
    5: (0.010146, -0.018461, 0.220889, -0.199021),
    6: (0.810286, -0.063731, 0.471779, 0.031009),
    7: (1.450348, -0.118282, 0.577968, 0.042099),
    8: (1.859977, -0.157142, 0.595609, -0.000602),
    9: (2.254285, -0.192050, 0.559926, -0.050161),
    10: (2.093614, -0.178162, 0.567744, -0.014044),
    11: (2.668314, -0.222609, 0.540228, 0.062877),
    12: (2.794154, -0.226445, 0.524109, 0.205575),
    13: (3.523434, -0.275591, 0.501323, 0.310483),
    14: (3.809900, -0.288019, 0.492051, 0.487155),
    15: (4.120209, -0.300000, 0.443842, 0.780537),
    16: (4.211228, -0.300000, 0.458794, 0.894138),
    17: (4.239017, -0.300000, 0.472521, 0.936564),
    18: (4.308171, -0.300000, 0.481392, 0.939457),
    19: (4.375851, -0.300000, 0.491222, 0.957197),
    20: (4.394482, -0.300000, 0.472251, 0.925957),
    21: (4.384975, -0.300000, 0.450147, 0.877896),
    22: (4.333921, -0.300000, 0.428538, 0.839588),
    23: (4.236048, -0.300000, 0.440331, 0.791419),
    24: (4.132194, -0.300000, 0.474370, 0.723518),
}

# Weight on the regression line vs the plain median profile (0.85 / 0.15),
# and the slope clip. Both chosen by leave-one-out cross-validation on the
# rolling launches; neighbouring settings score within 0.002 of each other.
BLEND_W = 0.85
SLOPE_MIN = -0.30


def round2(x: float):
    """Round to 2 decimals, half up, as the house baseline does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def release_day_hours(comp: dict) -> dict:
    """{hour 1..24: viewers} for a comparable's release day (UTC)."""
    try:
        base = comp["day_start"]
        y, mo, d = int(base[0:4]), int(base[5:7]), int(base[8:10])
    except Exception:
        return {}
    # Day start is midnight UTC; hourly rows carry ISO timestamps.
    out = {}
    for row in comp.get("hourly") or []:
        try:
            viewers = row["viewers"]
            if viewers is None or viewers <= 0:
                continue
            frm = row["from"]
            if int(frm[0:4]) != y or int(frm[5:7]) != mo or int(frm[8:10]) != d:
                continue
            out[int(frm[11:13]) + 1] = float(viewers)
        except Exception:
            continue
    return out


def fit_coefficients(case: dict) -> dict:
    """Per-hour (a, b, sigma, median) fit from the case's rolling comparables.

    Falls back to FALLBACK when fewer than six usable launches are found.
    The slope is clipped at SLOPE_MIN: the fit's steepest slopes come from
    the small launches, and an uncapped slope would extrapolate the peak
    ratio down without bound for a GTA-sized hour 1.
    """
    series = []
    for comp in case.get("comparables") or []:
        if comp.get("name") not in ROLLING_NAMES:
            continue
        hours = release_day_hours(comp)
        if 1 in hours and len(hours) >= 22:
            series.append(hours)
    if len(series) < 6:
        return dict(FALLBACK)
    fitted = {}
    for h in range(2, 25):
        pts = [(math.log(s[1]), math.log(s[h] / s[1]))
               for s in series if h in s]
        if len(pts) < 5:
            fitted[h] = FALLBACK[h]
            continue
        mx = statistics.mean(p[0] for p in pts)
        my = statistics.mean(p[1] for p in pts)
        vx = sum((p[0] - mx) ** 2 for p in pts)
        b = (sum((p[0] - mx) * (p[1] - my) for p in pts) / vx) if vx else 0.0
        b = max(b, SLOPE_MIN)
        a = my - b * mx
        sigma = statistics.pstdev([p[1] - (a + b * p[0]) for p in pts])
        fitted[h] = (a, b, max(sigma, 0.05),
                     statistics.median([p[1] for p in pts]))
    return fitted


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)
    coefs = fit_coefficients(case)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        a, b, sigma, median = coefs[h]
        # Shrink the regression 15% toward the plain median profile: the
        # ten-launch line is noisy, and the median is the robust centre.
        mu = log_p1 + BLEND_W * (a + b * log_p1) + (1 - BLEND_W) * median
        qs = []
        prev = 0.0
        for z in Z:
            q = math.exp(mu + z * sigma)
            q = min(max(q, 1.0), 1_000_000_000.0)
            q = round2(q)
            if q < prev:  # keep quantiles non-decreasing after rounding
                q = prev
            qs.append(q)
            prev = q
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
