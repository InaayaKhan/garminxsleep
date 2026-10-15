"""Generate the synthetic dataset in ``sample_data/``.

Everything here is invented: one simulated sleeper whose nights are recorded
by Sleepmeter (March to early June) and by a Garmin watch (mid-May to
end of August). The two overlap for two weeks so the merge logic has real work to
do. The output is deterministic, so re-running produces identical files.

    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --start 2024-08-31 --output sample_data_2y
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

OUTPUT = Path(__file__).resolve().parents[1] / "sample_data"
TIMEZONE = ZoneInfo("America/New_York")
SEED = 7

FIRST_DAY = date(2026, 3, 2)
LAST_DAY = date(2026, 8, 30)
SLEEPMETER_EXPORT_DAY = date(2026, 6, 7)
GARMIN_FIRST_DAY = date(2026, 5, 11)
GARMIN_START_DAY = date(2026, 5, 25)  # Garmin is authoritative from here on
# Watch not worn, so Garmin has no sleep: 3 June falls back to Sleepmeter;
# 14 July is after the Sleepmeter export, so it has no data at all.
WATCH_OFF_DAYS = {date(2026, 6, 3), date(2026, 7, 14)}
MISSING_DAY = date(2026, 4, 17)  # nothing logged in Sleepmeter
ZERO_PLACEHOLDER_DAYS = {date(2026, 3, 28), date(2026, 5, 2)}
FIRST_CYCLE_START = date(2026, 3, 5)

AIDS = {
    "CUSTOM_1001": "Melatonin",
    "CUSTOM_1002": "Magnesium",
    "CUSTOM_1003": "Chamomile tea",
}
AID_SCHEDULE = [  # (first day, aid id)
    (date(2026, 3, 2), "NONE"),
    (date(2026, 3, 10), "CUSTOM_1003"),
    (date(2026, 4, 1), "CUSTOM_1001"),
    (date(2026, 5, 1), "NONE"),
    (date(2026, 5, 21), "CUSTOM_1002"),
    (date(2026, 6, 8), "NONE"),
    (date(2026, 7, 1), "CUSTOM_1003"),
    (date(2026, 7, 20), "NONE"),
    (date(2026, 8, 1), "CUSTOM_1001"),
    (date(2026, 8, 21), "NONE"),
]
ACTIVITY_TYPES = [
    ("running", "Running", 10.5),
    ("cycling", "Cycling", 8.0),
    ("strength_training", "Strength", 6.0),
    ("walking", "Walk", 4.5),
]


@dataclass
class Night:
    wake_day: date
    sleep_start: datetime
    wake_time: datetime
    bedtime: datetime
    holes: list[tuple[int, int]]  # minute offsets from sleep_start
    aid_id: str
    zero: bool = False

    @property
    def gross_minutes(self) -> int:
        return int((self.wake_time - self.sleep_start).total_seconds() // 60)

    @property
    def asleep_minutes(self) -> int:
        return self.gross_minutes - sum(end - start for start, end in self.holes)


@dataclass
class Nap:
    day: date
    start: datetime
    minutes: int


@dataclass
class Activity:
    activity_id: int
    day: date
    start: datetime
    minutes: int
    type_key: str
    name: str
    kcal_per_minute: float
    zones: list[float] = field(default_factory=list)


def local(day: date, hour: float) -> datetime:
    minutes = round(hour * 60)
    return datetime.combine(day, time.min, tzinfo=TIMEZONE) + timedelta(minutes=minutes)


def ms(moment: datetime) -> int:
    return int(moment.astimezone(UTC).timestamp() * 1000)


def gmt_text(moment: datetime, separator: str = "T") -> str:
    return moment.astimezone(UTC).strftime(f"%Y-%m-%d{separator}%H:%M:%S")


def days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def aid_for(day: date) -> str:
    current = "NONE"
    for first, aid_id in AID_SCHEDULE:
        if day >= first:
            current = aid_id
    return current


def cycle_info(day: date) -> tuple[int, date, int]:
    """Return (day in cycle, cycle start, phase 1-4) for a 28/30-day rhythm."""
    start, length_index = FIRST_CYCLE_START, 0
    while day < start:  # step back for days before the reference cycle
        length_index -= 1
        start -= timedelta(days=28 if length_index % 2 == 0 else 30)
    while True:
        length = 28 if length_index % 2 == 0 else 30
        if day < start + timedelta(days=length):
            break
        start += timedelta(days=length)
        length_index += 1
    cycle_day = (day - start).days + 1
    phase = 1 if cycle_day <= 5 else 2 if cycle_day <= 13 else 3 if cycle_day <= 16 else 4
    return cycle_day, start, phase


def simulate(rng: random.Random) -> tuple[list[Night], list[Nap], list[Activity]]:
    nights, naps, activities = [], [], []
    activity_id = 18_000_000_001
    for day in days(FIRST_DAY, LAST_DAY):
        weekend = (day - timedelta(days=1)).weekday() >= 5  # Fri/Sat nights
        aid_id = aid_for(day)
        if day in ZERO_PLACEHOLDER_DAYS:
            moment = local(day, 7.0)
            nights.append(Night(day, moment, moment, moment, [], aid_id, zero=True))
        else:
            asleep_hour = 23.4 + (1.0 if weekend else 0.0) + rng.gauss(0, 0.6)
            hours = 7.1 + (0.7 if weekend else 0.0) + rng.gauss(0, 0.9)
            hours += 0.35 if aid_id != "NONE" else 0.0
            hours = min(max(hours, 3.5), 10.5)
            start = local(day - timedelta(days=1), asleep_hour)
            gross = round(hours * 60)
            holes = []
            for _ in range(rng.choice([0, 0, 1, 1, 2])):
                hole_start = rng.randint(60, max(61, gross - 60))
                holes.append((hole_start, hole_start + rng.randint(5, 25)))
            holes.sort()
            merged: list[tuple[int, int]] = []
            for hole in holes:
                if merged and hole[0] <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hole[1]))
                else:
                    merged.append(hole)
            gross += sum(end - begin for begin, end in merged)
            wake = start + timedelta(minutes=gross)
            bedtime = start - timedelta(minutes=rng.randint(5, 25))
            nights.append(Night(day, start, wake, bedtime, merged, aid_id))

        if rng.random() < 0.12:
            naps.append(Nap(day, local(day, rng.uniform(13.5, 16.0)), rng.randint(20, 60)))

        if rng.random() < 0.45:
            type_key, name, kcal = rng.choice(ACTIVITY_TYPES)
            hour = rng.choice([7.0, 12.3, 18.0, 18.5, 19.0])
            minutes = rng.randint(25, 90)
            zones = [rng.random() for _ in range(5)]
            total = sum(zones)
            activities.append(
                Activity(
                    activity_id,
                    day,
                    local(day, hour + rng.uniform(-0.3, 0.3)),
                    minutes,
                    type_key,
                    name,
                    kcal,
                    [minutes * 60 * share / total for share in zones],
                )
            )
            activity_id += rng.randint(3, 40)
    return nights, naps, activities


def write_sleepmeter(nights: list[Night], naps: list[Nap], rng: random.Random) -> None:
    def stamp(moment: datetime) -> str:
        return moment.strftime("%Y-%m-%d %H:%M%z")

    rows = []
    for night in nights:
        if night.wake_day > SLEEPMETER_EXPORT_DAY or night.wake_day == MISSING_DAY:
            continue
        quality = "" if night.zero else str(
            min(10, max(1, round(night.asleep_minutes / 60 + rng.gauss(0, 1) - 0.5)))
        )
        rows.append(
            {
                "wake": stamp(night.wake_time),
                "sleep": stamp(night.sleep_start),
                "bedtime": stamp(night.bedtime),
                "holes": ",".join(f"{start}-{end}" for start, end in night.holes),
                "type": "NIGHT_SLEEP",
                "dreams": "NONE",
                "aid": night.aid_id,
                "hindrances": "NONE",
                "tags": "",
                "quality": quality,
                "notes": "",
            }
        )
    for nap in naps:
        if nap.day > SLEEPMETER_EXPORT_DAY:
            continue
        end = nap.start + timedelta(minutes=nap.minutes)
        rows.append(
            {
                "wake": stamp(end),
                "sleep": stamp(nap.start),
                "bedtime": stamp(nap.start),
                "holes": "",
                "type": "NAP",
                "dreams": "NONE",
                "aid": "NONE",
                "hindrances": "NONE",
                "tags": "",
                "quality": "5",  # Sleepmeter requires a value for naps
                "notes": "",
            }
        )
    rows.sort(key=lambda row: row["wake"], reverse=True)

    path = OUTPUT / f"sleep_history_{SLEEPMETER_EXPORT_DAY.isoformat()}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        aid_writer = csv.writer(handle, lineterminator="\n")
        aid_writer.writerow(["custom_aid_id", "class", "name"])
        for aid_id, name in AIDS.items():
            aid_writer.writerow([aid_id, "SUPPLEMENT", name])
        handle.write("\n")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stage_levels(night: Night, rng: random.Random) -> list[tuple[int, int, int]]:
    """Minute runs of (start, end, level): 0 deep, 1 light, 2 REM, 3 awake, -1 unmeasurable."""
    hole_minutes = {m for start, end in night.holes for m in range(start, end)}
    deep_shift = rng.uniform(-0.05, 0.05)
    unmeasurable_start = rng.randint(30, night.gross_minutes) if rng.random() < 0.1 else -1
    per_minute = []
    for minute in range(night.gross_minutes):
        cycle, position = divmod(minute, 90)
        deep_fraction = max(0.04, 0.33 - 0.08 * cycle + deep_shift)
        rem_fraction = min(0.4, 0.12 + 0.06 * cycle)
        if minute in hole_minutes:
            level = 3
        elif 0 <= minute - unmeasurable_start < 12:
            level = -1
        elif position < 13:
            level = 1
        elif position < 13 + deep_fraction * 90:
            level = 0
        elif position >= 90 - rem_fraction * 90:
            level = 2
        else:
            level = 1
        per_minute.append(level)
    runs = []
    for minute, level in enumerate(per_minute):
        if runs and runs[-1][2] == level:
            runs[-1] = (runs[-1][0], minute + 1, level)
        else:
            runs.append((minute, minute + 1, level))
    return runs


def write_garmin(
    nights: list[Night], naps: list[Nap], activities: list[Activity], rng: random.Random
) -> None:
    raw = OUTPUT / "raw"
    nights_by_day = {night.wake_day: night for night in nights}
    nap_by_day = {nap.day: nap for nap in naps}
    activities_by_day: dict[date, list[Activity]] = {}
    for activity in activities:
        activities_by_day.setdefault(activity.day, []).append(activity)

    def dump(collector: str, day: date, payload: object) -> None:
        path = raw / collector / f"{day.isoformat()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    hrv_history: list[float] = []
    body_battery = 60.0
    for day in days(GARMIN_FIRST_DAY, LAST_DAY):
        night = nights_by_day[day]
        yesterday_load = sum(a.minutes for a in activities_by_day.get(day - timedelta(days=1), []))
        _, cycle_start, phase = cycle_info(day)

        # Sleep, stages and score
        if day in WATCH_OFF_DAYS or night.zero:
            dump("sleep", day, {"dailySleepDTO": {"calendarDate": day.isoformat()}})
            hrv = None
        else:
            runs = stage_levels(night, rng)
            seconds = {level: 0 for level in (-1, 0, 1, 2, 3)}
            levels = []
            for start, end, level in runs:
                seconds[level] += (end - start) * 60
                levels.append(
                    {
                        "startGMT": gmt_text(night.sleep_start + timedelta(minutes=start)) + ".0",
                        "endGMT": gmt_text(night.sleep_start + timedelta(minutes=end)) + ".0",
                        "activityLevel": float(level),
                    }
                )
            asleep = seconds[0] + seconds[1] + seconds[2]
            hours = asleep / 3600
            score = round(
                min(97, max(32, 48 + 7 * (hours - 5) + 60 * (seconds[0] / max(asleep, 1) - 0.15) + rng.gauss(0, 5)))
            )
            qualifier = "EXCELLENT" if score >= 90 else "GOOD" if score >= 80 else "FAIR" if score >= 60 else "POOR"
            hrv = 47 + 2.2 * (hours - 7) - 0.04 * yesterday_load + rng.gauss(0, 4)
            if phase == 4:  # luteal phase: HRV 20-25% lower
                hrv *= 1 - rng.uniform(0.20, 0.25)
            hrv = round(hrv, 1)
            nap = nap_by_day.get(day) if day >= GARMIN_START_DAY else None
            resting = round(56 - 0.3 * (hrv - 47) + rng.gauss(0, 1.5))

            def rated(value: float) -> dict[str, object]:
                key = "EXCELLENT" if value >= 22 else "GOOD" if value >= 17 else "FAIR" if value >= 12 else "POOR"
                return {"value": round(value), "qualifierKey": key}

            dump(
                "sleep",
                day,
                {
                    "dailySleepDTO": {
                        "calendarDate": day.isoformat(),
                        "sleepTimeSeconds": asleep,
                        "napTimeSeconds": nap.minutes * 60 if nap else 0,
                        "dailyNapDTOS": [
                            {
                                "napStartTimestampGMT": gmt_text(nap.start),
                                "napEndTimestampGMT": gmt_text(nap.start + timedelta(minutes=nap.minutes)),
                                "napTimeSec": nap.minutes * 60,
                            }
                        ]
                        if nap
                        else [],
                        "sleepStartTimestampGMT": ms(night.sleep_start),
                        "sleepEndTimestampGMT": ms(night.wake_time),
                        "sleepStartTimestampLocal": ms(night.sleep_start.replace(tzinfo=UTC)),
                        "sleepEndTimestampLocal": ms(night.wake_time.replace(tzinfo=UTC)),
                        "deepSleepSeconds": seconds[0],
                        "lightSleepSeconds": seconds[1],
                        "remSleepSeconds": seconds[2],
                        "awakeSleepSeconds": seconds[3],
                        "avgSleepStress": round(rng.uniform(10, 25), 1),
                        "avgHeartRate": round(resting + rng.uniform(2, 6), 1),
                        "averageSpO2Value": round(rng.uniform(93, 98), 1),
                        "averageRespirationValue": round(rng.uniform(13, 16), 1),
                        "awakeCount": len(night.holes) + rng.randint(0, 2),
                        "restlessMomentCount": rng.randint(15, 60),
                        "sleepScores": {
                            "overall": {"value": score, "qualifierKey": qualifier},
                            "totalDuration": {"qualifierKey": "GOOD" if hours >= 7 else "FAIR" if hours >= 6 else "POOR"},
                            "stress": {"qualifierKey": rng.choice(["EXCELLENT", "GOOD", "GOOD", "FAIR"])},
                            "restlessness": {"qualifierKey": rng.choice(["GOOD", "GOOD", "FAIR"])},
                            "awakeCount": {"qualifierKey": "GOOD" if len(night.holes) < 2 else "FAIR"},
                            "deepPercentage": rated(100 * seconds[0] / max(asleep, 1)),
                            "lightPercentage": {"value": round(100 * seconds[1] / max(asleep, 1)), "qualifierKey": "GOOD"},
                            "remPercentage": rated(100 * seconds[2] / max(asleep, 1)),
                        },
                    },
                    "sleepLevels": levels,
                    "restingHeartRate": resting,
                    "bodyBatteryChange": round(min(85, 8 * hours + rng.gauss(0, 6))),
                    "avgOvernightHrv": hrv,
                },
            )

        # HRV
        if hrv is not None:
            hrv_history.append(hrv)
            weekly = sum(hrv_history[-7:]) / len(hrv_history[-7:])
            dump(
                "hrv",
                day,
                {
                    "hrvSummary": {
                        "calendarDate": day.isoformat(),
                        "lastNightAvg": round(hrv),
                        "lastNight5MinHigh": round(hrv + rng.uniform(12, 25)),
                        "weeklyAvg": round(weekly),
                        "status": "BALANCED" if 42 <= weekly <= 54 else "UNBALANCED",
                    }
                },
            )

        # Activities (Garmin stores the day's list, local start date)
        todays = activities_by_day.get(day, [])
        dump(
            "activities",
            day,
            [
                {
                    "activityId": activity.activity_id,
                    "activityName": activity.name,
                    "activityType": {"typeKey": activity.type_key},
                    "startTimeGMT": gmt_text(activity.start, " "),
                    "endTimeGMT": gmt_text(activity.start + timedelta(minutes=activity.minutes), " "),
                    "elapsedDuration": float(activity.minutes * 60),
                    "calories": round(activity.minutes * activity.kcal_per_minute),
                    "averageHR": round(rng.uniform(115, 150)),
                    "maxHR": round(rng.uniform(155, 182)),
                    "moderateIntensityMinutes": round(activity.minutes * 0.5),
                    "vigorousIntensityMinutes": round(activity.minutes * (0.35 if activity.type_key == "running" else 0.1)),
                    **{f"hrTimeInZone_{zone}": round(seconds, 1) for zone, seconds in enumerate(activity.zones, 1)},
                }
                for activity in todays
            ],
        )

        # Daily summary
        active_minutes = sum(activity.minutes for activity in todays)
        steps = round(rng.uniform(4500, 9500) + active_minutes * 90)
        active_kcal = round(steps * 0.04 + sum(a.minutes * a.kcal_per_minute for a in todays))
        moderate = sum(round(a.minutes * 0.5) for a in todays) + rng.randint(0, 15)
        vigorous = sum(round(a.minutes * (0.35 if a.type_key == "running" else 0.1)) for a in todays)
        dump(
            "daily_summary",
            day,
            {
                "calendarDate": day.isoformat(),
                "totalSteps": steps,
                "floorsAscended": rng.randint(2, 18),
                "totalDistanceMeters": round(steps * 0.76),
                "activeKilocalories": active_kcal,
                "totalKilocalories": 1650 + active_kcal,
                "moderateIntensityMinutes": moderate,
                "vigorousIntensityMinutes": vigorous,
                "restingHeartRate": round(rng.uniform(52, 60)),
                "averageStressLevel": round(rng.uniform(22, 42)),
                "bodyBatteryHighestValue": round(rng.uniform(70, 98)),
                "bodyBatteryLowestValue": round(rng.uniform(8, 30)),
                "bodyBatteryChargedValue": round(rng.uniform(40, 75)),
                "bodyBatteryDrainedValue": round(rng.uniform(45, 80)),
            },
        )

        # Intraday heart rate (every 2 min), stress and body battery (every 3 min)
        day_start = local(day, 0)
        asleep_windows = [
            (n.sleep_start, n.wake_time)
            for n in (night, nights_by_day.get(day + timedelta(days=1)))
            if n is not None and not n.zero and day not in WATCH_OFF_DAYS
        ]
        active_windows = [(a.start, a.start + timedelta(minutes=a.minutes)) for a in todays]

        def state(moment: datetime) -> str:
            if any(start <= moment < end for start, end in active_windows):
                return "active"
            if any(start <= moment < end for start, end in asleep_windows):
                return "asleep"
            return "awake"

        heart_rate, stress, battery = [], [], []
        for minute in range(0, 24 * 60, 2):
            moment = day_start + timedelta(minutes=minute)
            current = state(moment)
            bpm = {"asleep": 54, "awake": 72, "active": 138}[current] + rng.gauss(0, 4)
            heart_rate.append([ms(moment), round(bpm)])
        for minute in range(0, 24 * 60, 3):
            moment = day_start + timedelta(minutes=minute)
            current = state(moment)
            level = {"asleep": 15, "awake": 32, "active": -1}[current]
            stress.append([ms(moment), level if level < 0 else max(0, round(level + rng.gauss(0, 8)))])
            body_battery += {"asleep": 0.45, "awake": -0.22, "active": -0.9}[current]
            body_battery = min(100.0, max(5.0, body_battery))
            battery.append([ms(moment), "MEASURED", round(body_battery), 2.0])
        dump("heart_rate", day, {"heartRateValues": heart_rate})
        dump("stress", day, {"stressValuesArray": stress, "bodyBatteryValuesArray": battery})

        # Menstrual cycle
        cycle_day, cycle_start, phase = cycle_info(day)
        dump(
            "menstrual/daily",
            day,
            {
                "daySummary": {
                    "dayInCycle": cycle_day,
                    "startDate": cycle_start.isoformat(),
                    "currentPhase": phase,
                    "predictedCycleLength": 29,
                    "predictedCycle": False,
                }
            },
        )


def write_config() -> None:
    (OUTPUT / "settings.json").write_text(
        json.dumps(
            {"timezone": TIMEZONE.key, "garmin_start_date": GARMIN_START_DAY.isoformat()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "aid_colors.json").write_text(
        json.dumps(
            {
                "melatonin": {"color": "#f97316", "nap": "#fdba74"},
                "magnesium": {"color": "#22c55e", "nap": "#4ade80"},
                "chamomile tea": {"color": "#facc15", "nap": "#fde68a"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # Garmin nights carry no aid, so record which aid was in use from when.
    (OUTPUT / "garmin_aid_overrides.json").write_text(
        json.dumps(
            {
                GARMIN_START_DAY.isoformat(): AIDS["CUSTOM_1002"],
                date(2026, 6, 8).isoformat(): "None",
                date(2026, 7, 1).isoformat(): AIDS["CUSTOM_1003"],
                date(2026, 7, 20).isoformat(): "None",
                date(2026, 8, 1).isoformat(): AIDS["CUSTOM_1001"],
                date(2026, 8, 21).isoformat(): "None",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    global FIRST_DAY, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, default=FIRST_DAY,
                        help="First Sleepmeter day (default: %(default)s).")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="Folder to (re)create (default: sample_data/).")
    arguments = parser.parse_args()
    FIRST_DAY, OUTPUT = arguments.start, arguments.output.resolve()
    rng = random.Random(SEED)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    nights, naps, activities = simulate(rng)
    write_sleepmeter(nights, naps, rng)
    write_garmin(nights, naps, activities, rng)
    write_config()
    files = sum(1 for path in OUTPUT.rglob("*") if path.is_file())
    size = sum(path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file())
    print(f"Wrote {files} files ({size / 1_000_000:.1f} MB) to {OUTPUT}")


if __name__ == "__main__":
    main()
