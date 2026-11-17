"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Model: a launch-day *shape* model, not persistence.

For every past big launch shipped inside the case file ("comparables"), we take
its release-day hourly curve and express it as log-ratios to its own hour-1
print, ln(v_h / v_1) for h = 2..24.  The forecast for the target launch is

    q_tau(h) = p1 * exp(x_tau(h))

where p1 is the target's hour-1 print and x_tau(h) is the tau-quantile of a
two-component normal mixture over the comparables' log-ratios at hour h:
one component for early-peak / flat launches, one for late-surge launches
(peak hour 17 or later), each fitted by its weighted median and robust
spread, mixed by the surge group's share of the kernel weight. A single
normal cannot express the two shapes past launches actually take; the
mixture can (see "Two shapes, not one" below).

Why this shape, and why these weights
--------------------------------------
A rolling-midnight console launch does not stay flat. Hour 1 (00:00-01:00 UTC)
already holds the UK, Europe and Australasia unlocks, so the count first
*dips* through the small hours (hours 2-5), then *ramps* as the US East Coast
unlocks at 05:00 UTC and the US West Coast at 08:00 UTC, *peaks* in the
evening (hours 19-22 UTC, US afternoon/evening + European evening), and
decays into the night. Persistence (the house baseline) misses that whole
arc; a pooled median curve captures it.

Each comparable is weighted by how much it resembles the target launch:

* tier: "comparable" launches count 1.0, "early-access-contaminated" 0.35
  (their hour 1 is polluted by days of early access), "weak" analogues 0.25.
* unlock shape: a rolling midnight-local unlock counts x1.5, a simultaneous
  global unlock x0.7 -- GTA VI unlocks at midnight local, rolling.
* launch size: a Gaussian kernel on ln(hour-1 print) with bandwidth 0.9.
  Big launches ramp and decay differently from small ones, and GTA VI is
  expected to be the biggest launch on record, so the biggest past launches
  (Elden Ring, Cyberpunk 2077, Diablo IV, Hogwarts Legacy) dominate the fit.
* pre-launch buzz: a Gaussian kernel (bandwidth 1.0) on ln(prebuzz), where
  prebuzz is the category's mean hourly viewers in the 24 h before release
  day divided by its hour-1 print. A quiet category before launch behaves
  differently from one already hot with early access or betas.

Comparables are dropped when they cannot anchor a ratio at all: the
Grand Theft Auto VI category's own pre-release history (not a launch), any
launch whose hour 1 is under 3% of its release-day peak (its real unlock
came hours later, so hour 1 is pre-launch noise), and any launch with fewer
than 18 usable release-day hours.

Two shapes, not one: past launches split into early-peak / flat ones (their
release-day peak lands before the evening, or hour 1 already is the peak)
and late-surge ones (peak in the US-afternoon/evening window, hour 17 or
later). A single normal pooled over both puts its median between the two
groups and its symmetric spread cannot express that the mass below the
median is tight while the mass above it is spread out. So each group is
fitted separately -- weighted median and the robust spread below -- and
the scored quantiles are the inverses of the two-normal mixture's CDF,
with the surge weight at each hour equal to the surge group's share of the
kernel weight. When a group is empty at an hour, that hour uses the pooled
single normal instead.

Spread: s_h is the mean of two robust spread estimators of the same
weighted log-ratio distribution -- the 90% range / (2 x 1.645) and the
interquartile range / 1.349 -- clipped to [0.25, 0.90]. Every quantile
curve is smoothed across neighbouring hours with a [0.25, 0.5, 0.25]
kernel, because a launch curve is smooth and hourly quantiles are noisy;
smoothing with non-negative weights preserves the ordering of the five
quantiles at each hour.

If the case carries too few usable comparables (< 5 with a usable hour),
the program falls back to FALLBACK_MED / FALLBACK_SIG, the same quantities
computed over the full comparable set in the kit (a typical mid-size
rolling-midnight launch), so the entry still forecasts instead of crashing.

Backtest (leave-one-out over the 29 usable comparables in the kit, scored
with the contest's own log-scale pinball loss): this model scores 0.265
against the house baseline's 0.359 on the same launches, typical miss
56.9% against 83.5%, 90% coverage 0.900 against 0.757 (the single-normal
version scored 0.270, miss 57.9%, coverage 0.868). The gain is broad -- 18 of 29 launches improve,
including the big early-peak launches most like GTA VI (ELDEN RING 0.125
to 0.089, Cyberpunk 2077 0.155 to 0.126, Starfield 0.498 to 0.400,
Assassin's Creed Shadows 0.428 to 0.349) -- and the biggest-15 launches
improve from 0.242 to 0.230. Honest weak spot, unchanged: the extreme
delayed-surge launches (Resident Evil Requiem 0.996 to 1.056, its hour-19
count sits 2.45 log units above its hour 1, beyond even the surge group's
own 95th percentile) -- no hour-1-observable signal separates them, and
every fatter-tailed variant tested bought that coverage back at a worse
overall loss, so the mixture keeps the honest spread rather than chasing
one outlier.

Deterministic: no randomness, no wall-clock, no dict-order dependence --
every collection is sorted before use, so two runs write byte-identical
outputs. Standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuning constants (chosen on the kit's comparables, see docstring) ---
SIZE_BANDWIDTH = 0.9    # Gaussian kernel width on ln(hour-1 print)
BUZZ_BANDWIDTH = 1.0    # Gaussian kernel width on ln(prebuzz ratio)
SIGMA_FLOOR = 0.25
SIGMA_CAP = 0.90
MIN_USABLE_HOURS = 18
MIN_H1_PEAK_FRACTION = 0.03
MIN_COMPARABLES = 5
SURGE_PEAK_HOUR = 17   # a launch peaks "late" when its release-day peak
                        # hour is 17 or later (the US-afternoon/evening window)

# --- fallback prior: pooled median log-ratio curve and spread for a typical
# rolling-midnight launch, computed over the kit's full comparable set. Used
# only when the case carries too few usable comparables to fit its own. ---
FALLBACK_MED = {
    2: -0.1628, 3: -0.2249, 4: -0.2789, 5: -0.2151, 6: -0.0637,
    7: 0.0123, 8: -0.0001, 9: -0.0177, 10: -0.0342, 11: -0.0369,
    12: 0.0066, 13: 0.1149, 14: 0.2634, 15: 0.4002, 16: 0.4984,
    17: 0.5783, 18: 0.6562, 19: 0.7279, 20: 0.7868, 21: 0.8027,
    22: 0.7769, 23: 0.7256, 24: 0.6757,
}
FALLBACK_SIG = {
    2: 0.2500, 3: 0.2500, 4: 0.2516, 5: 0.3232, 6: 0.4869,
    7: 0.6061, 8: 0.6263, 9: 0.6101, 10: 0.6092, 11: 0.6499,
    12: 0.6853, 13: 0.6811, 14: 0.6712, 15: 0.6640, 16: 0.6496,
    17: 0.6323, 18: 0.6224, 19: 0.6185, 20: 0.6057, 21: 0.5993,
    22: 0.6007, 23: 0.5913, 24: 0.5838,
}


def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round x to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def release_day_hours(comp):
    """Map hour (1..24) -> viewers for a comparable's release UTC day."""
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    hours = {}
    for row in comp.get("hourly") or []:
        try:
            at = parse_ts(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        delta = (at - day_start).total_seconds() / 3600.0
        hour = int(delta) + 1
        if 1 <= hour <= 24 and abs(delta - int(delta)) < 1e-6:
            if isinstance(viewers, (int, float)) and viewers > 0:
                hours[hour] = float(viewers)
    return hours


def tier_weight(comp):
    """Base weight for a comparable from the tier / unlock note it carries."""
    unlock = str(comp.get("unlock") or "")
    if "Not a launch" in unlock:
        return 0.0
    weight = 1.0
    if "Tier: early-access-contaminated" in unlock:
        weight *= 0.35
    elif "Tier: weak" in unlock:
        weight *= 0.25
    lowered = unlock.lower()
    if "midnight local" in lowered or "rolling" in lowered:
        weight *= 1.5
    if "simultaneous" in lowered:
        weight *= 0.7
    return weight


def prebuzz_ratio(hours, day_start, hourly_rows):
    """Mean viewers in the 24 h before release day, divided by hour-1 print.

    Returns None when the history is too sparse to say anything.
    """
    if 1 not in hours:
        return None
    start = day_start - timedelta(hours=24)
    values = []
    for row in hourly_rows or []:
        try:
            at = parse_ts(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if start <= at < day_start and isinstance(viewers, (int, float)) and viewers >= 0:
            values.append(float(viewers))
    if len(values) < 12:
        return None
    return (sum(values) / len(values)) / hours[1]


def weighted_percentile(pairs, pct):
    """Weighted percentile of (value, weight) pairs; pairs need not be sorted."""
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= pct * total:
            return value
    return ordered[-1][0]


def smooth_curve(curve):
    """Three-point [0.25, 0.5, 0.25] smoothing over neighbouring hours."""
    out = {}
    for hour in sorted(curve):
        weighted = []
        for shift, weight in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            if hour + shift in curve:
                weighted.append((curve[hour + shift], weight))
        out[hour] = sum(v * w for v, w in weighted) / sum(w for _, w in weighted)
    return out


def norm_cdf(x):
    """Standard normal CDF, via the error function (stdlib only)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def group_log_ratio_stats(pairs):
    """(Weighted median, robust spread) of (log-ratio, weight) pairs.

    The spread is the mean of two robust estimators of the same weighted
    distribution -- the 90% range / (2 x 1.645) and the interquartile range
    / 1.349 -- clipped to [SIGMA_FLOOR, SIGMA_CAP]. Returns None for an
    empty group.
    """
    if not pairs:
        return None
    median = weighted_percentile(pairs, 0.5)
    spread_90 = (weighted_percentile(pairs, 0.95)
                 - weighted_percentile(pairs, 0.05)) / (2.0 * 1.645)
    spread_iqr = (weighted_percentile(pairs, 0.75)
                  - weighted_percentile(pairs, 0.25)) / 1.349
    sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, 0.5 * (spread_90 + spread_iqr)))
    return median, sigma


def mixture_quantile(m_flat, s_flat, m_surge, s_surge, w_surge, tau):
    """One quantile of a two-normal mixture, by bisection on its CDF.

    Component "flat" (weight 1 - w_surge) is the early-peak / flat launches,
    component "surge" (weight w_surge) the late-peak ones. The bisection
    brackets the quantile between the components' +/-4-sigma points, far
    outside any scored level, and runs a fixed 60 iterations, so the result
    is deterministic.
    """
    lo = min(m_flat - 4.0 * s_flat, m_surge - 4.0 * s_surge)
    hi = max(m_flat + 4.0 * s_flat, m_surge + 4.0 * s_surge)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = (w_surge * norm_cdf((mid - m_surge) / s_surge)
               + (1.0 - w_surge) * norm_cdf((mid - m_flat) / s_flat))
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_shape(p1, target_buzz, comparables):
    """Fit per-hour log-ratio quantile curves from the case's comparables.

    Returns {level: {hour: log-quantile}} for the five scored levels; falls
    back to the embedded prior (median curve +/- z * spread) when too few
    comparables survive the filters.

    The quantiles come from a two-component mixture, not a single normal.
    Past launches genuinely split into two shapes: early-peak / flat ones
    (their release-day peak lands before the evening, or hour 1 already is
    the peak) and late-surge ones (peak in the US-afternoon/evening window,
    hour 17 or later). A single normal pooled over both puts its median
    between the two groups -- too high for the flat launches (Starfield,
    Assassin's Creed Shadows: the old model's median at hour 19 sat 0.7-1.1
    log units above their counts) and its symmetric spread cannot express
    that the mass below the median is tight while the mass above it is
    spread out. Each component is fitted separately (weighted median and
    robust spread, same size/buzz kernels as before), the surge weight at
    each hour is the surge group's share of the kernel weight, and the
    scored quantiles are the mixture CDF's inverses. When a group is empty
    at an hour, that hour falls back to the pooled single normal.
    """
    usable = []
    for comp in comparables:
        base = tier_weight(comp)
        if base <= 0.0:
            continue
        hours = release_day_hours(comp)
        if 1 not in hours:
            continue
        present = [h for h in range(2, 25) if h in hours]
        if len(present) < MIN_USABLE_HOURS:
            continue
        if hours[1] < MIN_H1_PEAK_FRACTION * max(hours.values()):
            continue
        try:
            day_start = parse_ts(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        buzz = prebuzz_ratio(hours, day_start, comp.get("hourly") or [])
        peak_hour = max(hours, key=lambda h: hours[h])
        usable.append((base, hours, buzz, peak_hour >= SURGE_PEAK_HOUR))

    if len(usable) < MIN_COMPARABLES:
        curves = {}
        for tau, z in zip(LEVELS, Z):
            curves[tau] = {h: FALLBACK_MED[h] + z * FALLBACK_SIG[h]
                           for h in range(2, 25)}
        return curves

    log_p1 = math.log(p1)
    log_buzz = math.log(max(target_buzz, 1e-4)) if target_buzz is not None else None
    curves = {tau: {} for tau in LEVELS}
    for hour in range(2, 25):
        groups = {True: [], False: []}
        for base, hours, buzz, is_surge in usable:
            if hour not in hours:
                continue
            weight = base * math.exp(
                -0.5 * ((math.log(hours[1]) - log_p1) / SIZE_BANDWIDTH) ** 2
            )
            if log_buzz is not None and buzz is not None:
                weight *= math.exp(
                    -0.5 * ((math.log(max(buzz, 1e-4)) - log_buzz) / BUZZ_BANDWIDTH) ** 2
                )
            groups[is_surge].append((math.log(hours[hour] / hours[1]), weight))
        flat = group_log_ratio_stats(groups[False])
        surge = group_log_ratio_stats(groups[True])
        if flat is None or surge is None:
            pooled = group_log_ratio_stats(groups[False] + groups[True])
            if pooled is None:
                for tau, z in zip(LEVELS, Z):
                    curves[tau][hour] = FALLBACK_MED[hour] + z * FALLBACK_SIG[hour]
                continue
            median, sigma = pooled
            for tau, z in zip(LEVELS, Z):
                curves[tau][hour] = median + z * sigma
            continue
        (m_flat, s_flat), (m_surge, s_surge) = flat, surge
        total_weight = sum(w for _, w in groups[True]) + sum(w for _, w in groups[False])
        w_surge = (sum(w for _, w in groups[True]) / total_weight
                   if total_weight > 0 else 0.5)
        for tau in LEVELS:
            curves[tau][hour] = mixture_quantile(
                m_flat, s_flat, m_surge, s_surge, w_surge, tau)
    for tau in LEVELS:
        curves[tau] = smooth_curve(curves[tau])
    return curves


def hour_quantiles(p1, log_quantiles):
    """The five quantiles for one hour, rounded and kept ordered and valid.

    log_quantiles are the five log-ratio quantiles for the hour, in level
    order; the caller guarantees they are non-decreasing (smoothing a set
    of ordered curves with non-negative weights preserves the order), and
    the clamp here is only a safety net.
    """
    quantiles = []
    previous = 0.0
    for log_q in log_quantiles:
        value = p1 * math.exp(log_q)
        value = round2(value)
        value = min(max(value, 1.0), 1e9)
        if value < previous:
            value = previous
        quantiles.append(value)
        previous = value
    return quantiles


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    try:
        day_start = parse_ts(case["day_start"])
    except (KeyError, TypeError, ValueError):
        day_start = None
    target_buzz = None
    if day_start is not None:
        target_buzz = prebuzz_ratio(
            {1: p1}, day_start, case.get("history_hourly") or []
        )

    curves = fit_shape(p1, target_buzz, case.get("comparables") or [])
    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        rows.append({
            "hour": hour,
            "q": hour_quantiles(p1, [curves[tau][hour] for tau in LEVELS]),
        })
    rows.sort(key=lambda row: row["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
