from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M%z"
REQUIRED_COLUMNS = {
    "wake",
    "sleep",
    "bedtime",
    "holes",
    "type",
    "dreams",
    "aid",
    "hindrances",
    "tags",
    "quality",
    "notes",
}


@dataclass(frozen=True)
class SleepDataset:
    source_path: Path
    coverage_start: date
    coverage_end: date
    aids: pd.DataFrame
    sessions: pd.DataFrame
    intervals: pd.DataFrame
    sleep_stages: pd.DataFrame
    activities: pd.DataFrame
    menstrual: pd.DataFrame
    intraday: pd.DataFrame
    daily: pd.DataFrame
    garmin_metrics: pd.DataFrame


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def _merged_holes(raw: str, duration_minutes: float) -> list[tuple[float, float]]:
    """Parse Sleepmeter minute-offset holes and merge any overlaps."""
    if not raw.strip() or duration_minutes <= 0:
        return []

    matches = re.findall(r"(\d+)\s*-\s*(\d+)", raw)
    if not matches:
        raise ValueError(f"Unsupported Sleepmeter holes value: {raw!r}")

    holes: list[tuple[float, float]] = []
    for start_text, end_text in matches:
        start = float(start_text)
        end = float(end_text)
        if end < start:
            raise ValueError(f"Hole ends before it starts: {raw!r}")
        start = min(max(start, 0.0), duration_minutes)
        end = min(max(end, 0.0), duration_minutes)
        if end > start:
            holes.append((start, end))

    holes.sort()
    merged: list[tuple[float, float]] = []
    for start, end in holes:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _asleep_ranges(
    sleep_start: datetime, wake_time: datetime, holes: list[tuple[float, float]]
) -> list[tuple[datetime, datetime]]:
    duration = (wake_time - sleep_start).total_seconds() / 60
    if duration <= 0:
        return []

    ranges: list[tuple[datetime, datetime]] = []
    cursor = 0.0
    for hole_start, hole_end in holes:
        if hole_start > cursor:
            ranges.append(
                (
                    sleep_start + timedelta(minutes=cursor),
                    sleep_start + timedelta(minutes=hole_start),
                )
            )
        cursor = max(cursor, hole_end)
    if cursor < duration:
        ranges.append((sleep_start + timedelta(minutes=cursor), wake_time))
    return ranges


def _split_sections(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8-sig")
    sections = re.split(r"\r?\n\s*\r?\n", text.strip(), maxsplit=1)
    if len(sections) != 2:
        raise ValueError("Expected an aid lookup section and a sleep history section")
    aids = list(csv.DictReader(io.StringIO(sections[0])))
    rows = list(csv.DictReader(io.StringIO(sections[1])))
    if not rows:
        raise ValueError("The Sleepmeter export contains no sleep records")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Sleepmeter export is missing columns: {sorted(missing)}")
    return aids, rows


def load_sleepmeter(path: str | Path) -> SleepDataset:
    source_path = Path(path).expanduser().resolve()
    aid_rows, raw_rows = _split_sections(source_path)

    aid_lookup = {"NONE": "None"}
    for row in aid_rows:
        aid_lookup[row["custom_aid_id"]] = row["name"]

    sessions: list[dict[str, object]] = []
    intervals: list[dict[str, object]] = []

    for source_row, raw in enumerate(raw_rows, start=1):
        sleep_start = _parse_timestamp(raw["sleep"])
        wake_time = _parse_timestamp(raw["wake"])
        bedtime = _parse_timestamp(raw["bedtime"])
        gross_minutes = (wake_time - sleep_start).total_seconds() / 60
        if gross_minutes < 0:
            raise ValueError(f"Record {source_row} wakes before it sleeps")

        holes = _merged_holes(raw["holes"], gross_minutes)
        asleep_ranges = _asleep_ranges(sleep_start, wake_time, holes)
        asleep_minutes = sum(
            (end - start).total_seconds() / 60 for start, end in asleep_ranges
        )
        interruption_minutes = gross_minutes - asleep_minutes
        session_key = "|".join(
            [raw["sleep"], raw["wake"], raw["type"], raw["aid"], str(source_row)]
        )
        session_id = hashlib.sha256(session_key.encode()).hexdigest()[:16]
        aid_name = aid_lookup.get(raw["aid"], raw["aid"])
        # Sleepmeter required a quality value for naps, but Garmin does not
        # provide nap sleep scores. Treat those entered 5s as placeholders.
        quality = (
            int(raw["quality"])
            if raw["type"] == "NIGHT_SLEEP"
            and gross_minutes > 0
            and raw["quality"].strip()
            else None
        )

        sessions.append(
            {
                "session_id": session_id,
                "source_row": source_row,
                "sleep_date": wake_time.date(),
                "sleep_start": sleep_start,
                "wake_time": wake_time,
                "bedtime": bedtime,
                "session_type": raw["type"],
                "gross_minutes": gross_minutes,
                "interruption_minutes": interruption_minutes,
                "sleep_minutes": asleep_minutes,
                "zero_duration": gross_minutes == 0,
                "aid_id": raw["aid"],
                "aid_name": aid_name,
                "legacy_garmin_score_10": quality,
                "holes": raw["holes"],
                "dreams": raw["dreams"],
                "hindrances": raw["hindrances"],
                "tags": raw["tags"],
                "notes": raw["notes"],
                "source": "sleepmeter",
                "source_file": source_path.name,
            }
        )

        for interval_index, (start, end) in enumerate(asleep_ranges, start=1):
            intervals.append(
                {
                    "session_id": session_id,
                    "interval_index": interval_index,
                    "sleep_date": wake_time.date(),
                    "interval_start": start,
                    "interval_end": end,
                    "session_type": raw["type"],
                    "aid_name": aid_name,
                    "legacy_garmin_score_10": quality,
                    "source": "sleepmeter",
                }
            )

    sessions_df = pd.DataFrame(sessions).sort_values(
        ["wake_time", "sleep_start", "source_row"]
    ).reset_index(drop=True)
    intervals_df = pd.DataFrame(intervals)
    if not intervals_df.empty:
        intervals_df = intervals_df.sort_values(
            ["interval_start", "interval_end", "session_id"]
        ).reset_index(drop=True)

    coverage_start = min(sessions_df["sleep_date"])
    last_session_date = max(sessions_df["sleep_date"])
    export_date_match = re.search(r"(\d{4}-\d{2}-\d{2})(?=\.csv$)", source_path.name)
    export_date = (
        date.fromisoformat(export_date_match.group(1))
        if export_date_match
        else last_session_date
    )
    coverage_end = max(last_session_date, export_date)
    daily_index = pd.date_range(coverage_start, coverage_end, freq="D").date

    daily_rows: list[dict[str, object]] = []
    for current_date in daily_index:
        day_sessions = sessions_df[sessions_df["sleep_date"] == current_date]
        night_minutes = day_sessions.loc[
            day_sessions["session_type"] == "NIGHT_SLEEP", "sleep_minutes"
        ].sum()
        nap_minutes = day_sessions.loc[
            day_sessions["session_type"] == "NAP", "sleep_minutes"
        ].sum()
        total_minutes = float(night_minutes + nap_minutes)
        if day_sessions.empty:
            source_state = "sleepmeter_no_session_zero"
        elif total_minutes == 0:
            source_state = "sleepmeter_zero_placeholder"
        else:
            source_state = "sleepmeter_recorded"

        daily_rows.append(
            {
                "date": current_date,
                "night_minutes": float(night_minutes),
                "nap_minutes": float(nap_minutes),
                "total_sleep_minutes": total_minutes,
                "session_count": int(len(day_sessions)),
                "zero_placeholder_count": int(day_sessions["zero_duration"].sum()),
                "zero_sleep": total_minutes == 0,
                "source_state": source_state,
                "source": "sleepmeter",
            }
        )

    aids_df = pd.DataFrame(
        [
            {"aid_id": aid_id, "aid_name": aid_name}
            for aid_id, aid_name in aid_lookup.items()
        ]
    ).sort_values("aid_name", key=lambda column: column.str.lower())

    return SleepDataset(
        source_path=source_path,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        aids=aids_df.reset_index(drop=True),
        sessions=sessions_df,
        intervals=intervals_df,
        sleep_stages=pd.DataFrame(
            columns=[
                "session_id",
                "sleep_date",
                "interval_start",
                "interval_end",
                "stage",
                "source",
            ]
        ),
        activities=pd.DataFrame(
            columns=[
                "activity_id",
                "activity_date",
                "activity_start",
                "activity_end",
                "activity_name",
                "activity_type",
                "duration_seconds",
                "source",
            ]
        ),
        menstrual=pd.DataFrame(
            columns=[
                "date",
                "cycle_day",
                "cycle_start_date",
                "current_phase",
                "predicted_cycle_length",
                "predicted_cycle",
                "source",
            ]
        ),
        intraday=pd.DataFrame(
            columns=["timestamp", "date", "heart_rate", "stress", "body_battery"]
        ),
        daily=pd.DataFrame(daily_rows),
        garmin_metrics=pd.DataFrame(),
    )


def _garmin_timestamp(value: str, timezone: ZoneInfo) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(timezone)


def _raw_files(raw_roots: list[Path], data_type: str) -> dict[date, Path]:
    files: dict[date, Path] = {}
    for root in raw_roots:
        for path in (root / data_type).glob("*.json") if root.exists() else []:
            try:
                file_date = date.fromisoformat(path.stem)
            except ValueError:
                continue
            current = files.get(file_date)
            if current is None or path.stat().st_mtime_ns > current.stat().st_mtime_ns:
                files[file_date] = path
    return files


def load_combined_sleep_data(
    sleepmeter_path: str | Path,
    raw_roots: list[str | Path],
    garmin_start_date: date,
    aid_overrides_path: str | Path | None = None,
    score_overrides_path: str | Path | None = None,
    merge_overrides_path: str | Path | None = None,
    timezone_name: str = "UTC",
) -> SleepDataset:
    """Use Sleepmeter before the explicit Garmin handover date, then Garmin."""
    historical = load_sleepmeter(sleepmeter_path)
    roots = [Path(root).expanduser().resolve() for root in raw_roots]
    aid_overrides: dict[str, str] = {}
    if aid_overrides_path is not None:
        overrides_path = Path(aid_overrides_path).expanduser().resolve()
        if overrides_path.exists():
            aid_overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    aid_change_dates = sorted(
        (date.fromisoformat(change_date), aid_name)
        for change_date, aid_name in aid_overrides.items()
    )
    score_overrides: dict[str, float] = {}
    if score_overrides_path is not None:
        overrides_path = Path(score_overrides_path).expanduser().resolve()
        if overrides_path.exists():
            score_overrides = {
                session_id: float(score)
                for session_id, score in json.loads(
                    overrides_path.read_text(encoding="utf-8")
                ).items()
            }
    merge_overrides: dict[str, dict[str, object]] = {}
    if merge_overrides_path is not None:
        overrides_path = Path(merge_overrides_path).expanduser().resolve()
        if overrides_path.exists():
            merge_overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    sleep_files = _raw_files(roots, "sleep")
    hrv_files = _raw_files(roots, "hrv")
    daily_summary_files = _raw_files(roots, "daily_summary")
    garmin_dates = sorted(sleep_files)
    if not garmin_dates:
        return historical

    local_timezone = ZoneInfo(timezone_name)
    sessions: list[dict[str, object]] = []
    intervals: list[dict[str, object]] = []
    sleep_stages: list[dict[str, object]] = []
    garmin_local_windows: dict[str, tuple[datetime, datetime]] = {}
    activities_by_id: dict[str, dict[str, object]] = {}
    intraday_rows: dict[datetime, dict[str, object]] = {}
    menstrual_by_date: dict[date, dict[str, object]] = {}
    daily_by_date: dict[date, dict[str, object]] = {}
    metrics: list[dict[str, object]] = []

    for sleep_date in garmin_dates:
        sleep_path = sleep_files[sleep_date]
        payload = json.loads(sleep_path.read_text(encoding="utf-8")) or {}
        dto = payload.get("dailySleepDTO") or {}
        sleep_seconds = float(dto.get("sleepTimeSeconds") or 0)
        nap_seconds = float(dto.get("napTimeSeconds") or 0)
        start_ms = dto.get("sleepStartTimestampGMT")
        end_ms = dto.get("sleepEndTimestampGMT")
        score_100 = ((dto.get("sleepScores") or {}).get("overall") or {}).get("value")
        sleep_scores = dto.get("sleepScores") or {}
        aid_name = "None"
        for change_date, changed_aid in aid_change_dates:
            if change_date > sleep_date:
                break
            aid_name = changed_aid
        valid_session = sleep_seconds > 0 and start_ms and end_ms

        daily_by_date[sleep_date] = {
            "date": sleep_date,
            "night_minutes": sleep_seconds / 60 if valid_session else float("nan"),
            "nap_minutes": nap_seconds / 60 if valid_session else float("nan"),
            "total_sleep_minutes": (
                (sleep_seconds + nap_seconds) / 60 if valid_session else float("nan")
            ),
            "session_count": 1 if valid_session else 0,
            "zero_placeholder_count": 0,
            "zero_sleep": False,
            "source_state": "garmin_recorded" if valid_session else "garmin_unknown",
            "source": "garmin",
        }

        hrv_payload = {}
        if sleep_date in hrv_files:
            hrv_payload = json.loads(
                hrv_files[sleep_date].read_text(encoding="utf-8")
            ) or {}
        hrv_summary = hrv_payload.get("hrvSummary") or {}
        daily_summary = {}
        if sleep_date in daily_summary_files:
            daily_summary = json.loads(
                daily_summary_files[sleep_date].read_text(encoding="utf-8")
            ) or {}
        moderate_minutes = float(
            daily_summary.get("moderateIntensityMinutes") or 0
        )
        vigorous_minutes = float(
            daily_summary.get("vigorousIntensityMinutes") or 0
        )
        metrics.append(
            {
                "date": sleep_date,
                "garmin_sleep_score": score_100,
                "deep_sleep_minutes": (dto.get("deepSleepSeconds") or 0) / 60,
                "light_sleep_minutes": (dto.get("lightSleepSeconds") or 0) / 60,
                "rem_sleep_minutes": (dto.get("remSleepSeconds") or 0) / 60,
                "awake_minutes": (dto.get("awakeSleepSeconds") or 0) / 60,
                "average_sleep_stress": dto.get("avgSleepStress"),
                "average_sleep_hr": dto.get("avgHeartRate"),
                "resting_heart_rate": payload.get("restingHeartRate"),
                "average_spo2": dto.get("averageSpO2Value"),
                "average_respiration": dto.get("averageRespirationValue"),
                "body_battery_change": payload.get("bodyBatteryChange"),
                "overnight_hrv": payload.get("avgOvernightHrv"),
                "awake_count": dto.get("awakeCount"),
                "restless_moment_count": dto.get("restlessMomentCount"),
                "score_overall_qualifier": (sleep_scores.get("overall") or {}).get(
                    "qualifierKey"
                ),
                "score_duration_qualifier": (
                    sleep_scores.get("totalDuration") or {}
                ).get("qualifierKey"),
                "score_stress_qualifier": (sleep_scores.get("stress") or {}).get(
                    "qualifierKey"
                ),
                "score_restlessness_qualifier": (
                    sleep_scores.get("restlessness") or {}
                ).get("qualifierKey"),
                "score_awake_count_qualifier": (
                    sleep_scores.get("awakeCount") or {}
                ).get("qualifierKey"),
                "score_deep_qualifier": (
                    sleep_scores.get("deepPercentage") or {}
                ).get("qualifierKey"),
                "score_light_qualifier": (
                    sleep_scores.get("lightPercentage") or {}
                ).get("qualifierKey"),
                "score_rem_qualifier": (
                    sleep_scores.get("remPercentage") or {}
                ).get("qualifierKey"),
                "score_deep_percentage": (
                    sleep_scores.get("deepPercentage") or {}
                ).get("value"),
                "score_light_percentage": (
                    sleep_scores.get("lightPercentage") or {}
                ).get("value"),
                "score_rem_percentage": (
                    sleep_scores.get("remPercentage") or {}
                ).get("value"),
                "hrv_last_night_average": hrv_summary.get("lastNightAvg"),
                "hrv_five_minute_high": hrv_summary.get("lastNight5MinHigh"),
                "hrv_weekly_average": hrv_summary.get("weeklyAvg"),
                "hrv_status": hrv_summary.get("status"),
                "daily_steps": daily_summary.get("totalSteps"),
                "daily_floors_ascended": daily_summary.get("floorsAscended"),
                "daily_distance_meters": daily_summary.get("totalDistanceMeters"),
                "daily_active_calories": daily_summary.get("activeKilocalories"),
                "daily_total_calories": daily_summary.get("totalKilocalories"),
                "daily_intensity_minutes": moderate_minutes
                + 2 * vigorous_minutes,
                "daily_resting_heart_rate": daily_summary.get("restingHeartRate"),
                "daily_average_stress": daily_summary.get("averageStressLevel"),
                "daily_body_battery_high": daily_summary.get(
                    "bodyBatteryHighestValue"
                ),
                "daily_body_battery_low": daily_summary.get("bodyBatteryLowestValue"),
                "daily_body_battery_charged": daily_summary.get(
                    "bodyBatteryChargedValue"
                ),
                "daily_body_battery_drained": daily_summary.get(
                    "bodyBatteryDrainedValue"
                ),
                "source_file": sleep_path.name,
            }
        )
        if sleep_date >= garmin_start_date:
            # Naps are only listed with their times here; before the handover
            # Sleepmeter's own nap records are used instead.
            for nap in dto.get("dailyNapDTOS") or []:
                nap_start_text = nap.get("napStartTimestampGMT")
                nap_end_text = nap.get("napEndTimestampGMT")
                if not nap_start_text or not nap_end_text:
                    continue
                nap_start = _garmin_timestamp(nap_start_text, local_timezone)
                nap_end = _garmin_timestamp(nap_end_text, local_timezone)
                nap_gross = (nap_end - nap_start).total_seconds() / 60
                nap_minutes = float(nap.get("napTimeSec") or nap_gross * 60) / 60
                nap_id = hashlib.sha256(
                    f"garmin-nap|{sleep_date.isoformat()}|{nap_start_text}".encode()
                ).hexdigest()[:16]
                sessions.append(
                    {
                        "session_id": nap_id,
                        "source_row": None,
                        "sleep_date": sleep_date,
                        "sleep_start": nap_start,
                        "wake_time": nap_end,
                        "bedtime": nap_start,
                        "session_type": "NAP",
                        "gross_minutes": nap_gross,
                        "interruption_minutes": max(nap_gross - nap_minutes, 0.0),
                        "sleep_minutes": nap_minutes,
                        "zero_duration": False,
                        "aid_id": "NONE",
                        "aid_name": "None",
                        "legacy_garmin_score_10": None,
                        "holes": "",
                        "dreams": "",
                        "hindrances": "",
                        "tags": "",
                        "notes": "",
                        "source": "garmin",
                        "source_file": sleep_path.name,
                    }
                )
                intervals.append(
                    {
                        "session_id": nap_id,
                        "interval_index": 1,
                        "sleep_date": sleep_date,
                        "interval_start": nap_start,
                        "interval_end": nap_end,
                        "session_type": "NAP",
                        "aid_name": "None",
                        "legacy_garmin_score_10": None,
                        "source": "garmin",
                    }
                )
        if not valid_session:
            continue

        sleep_start = datetime.fromtimestamp(start_ms / 1000, tz=UTC).astimezone(
            local_timezone
        )
        wake_time = datetime.fromtimestamp(end_ms / 1000, tz=UTC).astimezone(
            local_timezone
        )
        session_id = hashlib.sha256(
            f"garmin|{sleep_date.isoformat()}|{start_ms}|{end_ms}".encode()
        ).hexdigest()[:16]
        local_start_ms = dto.get("sleepStartTimestampLocal")
        local_end_ms = dto.get("sleepEndTimestampLocal")
        if local_start_ms and local_end_ms:
            garmin_local_windows[session_id] = (
                datetime.fromtimestamp(local_start_ms / 1000, tz=UTC).replace(
                    tzinfo=local_timezone
                ),
                datetime.fromtimestamp(local_end_ms / 1000, tz=UTC).replace(
                    tzinfo=local_timezone
                ),
            )
        gross_minutes = (wake_time - sleep_start).total_seconds() / 60
        sessions.append(
            {
                "session_id": session_id,
                "source_row": None,
                "sleep_date": sleep_date,
                "sleep_start": sleep_start,
                "wake_time": wake_time,
                "bedtime": sleep_start,
                "session_type": "NIGHT_SLEEP",
                "gross_minutes": gross_minutes,
                "interruption_minutes": gross_minutes - sleep_seconds / 60,
                "sleep_minutes": sleep_seconds / 60,
                "zero_duration": False,
                "aid_id": "NONE" if aid_name == "None" else "LOCAL_OVERRIDE",
                "aid_name": aid_name,
                "legacy_garmin_score_10": (
                    float(score_100) / 10 if score_100 is not None else None
                ),
                "holes": "",
                "dreams": "",
                "hindrances": "",
                "tags": "",
                "notes": "",
                "source": "garmin",
                "source_file": sleep_path.name,
            }
        )
        for interval_index, level in enumerate(payload.get("sleepLevels") or [], 1):
            start_text, end_text = level.get("startGMT"), level.get("endGMT")
            if not start_text or not end_text:
                continue
            raw_activity_level = level.get("activityLevel")
            activity_level = (
                int(raw_activity_level) if raw_activity_level is not None else -1
            )
            stage_name = {
                0: "Deep",
                1: "Light",
                2: "REM",
                3: "Awake",
                -1: "Unmeasurable",
            }.get(activity_level, "Unmeasurable")
            interval_start = _garmin_timestamp(start_text, local_timezone)
            interval_end = _garmin_timestamp(end_text, local_timezone)
            sleep_stages.append(
                {
                    "session_id": session_id,
                    "sleep_date": sleep_date,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                    "stage": stage_name,
                    "source": "garmin",
                }
            )
            if activity_level not in (0, 1, 2):
                continue
            intervals.append(
                {
                    "session_id": session_id,
                    "interval_index": interval_index,
                    "sleep_date": sleep_date,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                    "session_type": "NIGHT_SLEEP",
                    "aid_name": aid_name,
                    "legacy_garmin_score_10": (
                        float(score_100) / 10 if score_100 is not None else None
                    ),
                    "source": "garmin",
                }
            )

    historical_sessions = historical.sessions.copy()
    historical_intervals = historical.intervals.copy()
    garmin_sessions = pd.DataFrame(sessions)
    garmin_intervals = pd.DataFrame(intervals)
    garmin_stages = pd.DataFrame(
        sleep_stages,
        columns=[
            "session_id",
            "sleep_date",
            "interval_start",
            "interval_end",
            "stage",
            "source",
        ],
    )
    matched_session_ids: dict[str, str] = {}
    matched_time_shifts: dict[str, timedelta] = {}
    subset_stage_session_ids: dict[str, str] = {}
    subset_time_shifts: dict[str, timedelta] = {}
    subset_windows: dict[str, tuple[datetime, datetime]] = {}
    used_historical_ids: set[str] = set()
    match_tolerance_seconds = 120

    if not garmin_sessions.empty:
        for garmin_row in garmin_sessions.itertuples(index=False):
            if garmin_row.sleep_date >= garmin_start_date:
                continue
            candidates = historical_sessions[
                (historical_sessions["sleep_date"] == garmin_row.sleep_date)
                & (historical_sessions["session_type"] == garmin_row.session_type)
                & (~historical_sessions["session_id"].isin(used_historical_ids))
            ].copy()
            if candidates.empty:
                continue
            candidates["absolute_start_delta"] = candidates["sleep_start"].map(
                lambda value: abs((value - garmin_row.sleep_start).total_seconds())
            )
            candidates["absolute_wake_delta"] = candidates["wake_time"].map(
                lambda value: abs((value - garmin_row.wake_time).total_seconds())
            )
            local_window = garmin_local_windows.get(str(garmin_row.session_id))
            if local_window is not None:
                local_start, local_wake = local_window
                candidates["local_start_delta"] = candidates["sleep_start"].map(
                    lambda value: abs(
                        (
                            value.replace(tzinfo=None)
                            - local_start.replace(tzinfo=None)
                        ).total_seconds()
                    )
                )
                candidates["local_wake_delta"] = candidates["wake_time"].map(
                    lambda value: abs(
                        (
                            value.replace(tzinfo=None)
                            - local_wake.replace(tzinfo=None)
                        ).total_seconds()
                    )
                )
            else:
                candidates["local_start_delta"] = float("inf")
                candidates["local_wake_delta"] = float("inf")
            candidates["absolute_match"] = (
                candidates["absolute_start_delta"] <= match_tolerance_seconds
            ) & (candidates["absolute_wake_delta"] <= match_tolerance_seconds)
            candidates["local_match"] = (
                candidates["local_start_delta"] <= match_tolerance_seconds
            ) & (candidates["local_wake_delta"] <= match_tolerance_seconds)
            candidates = candidates[
                candidates["absolute_match"] | candidates["local_match"]
            ].copy()
            candidates["match_delta"] = candidates[
                [
                    "absolute_start_delta",
                    "absolute_wake_delta",
                    "local_start_delta",
                    "local_wake_delta",
                ]
            ].min(axis=1)
            candidates = candidates.sort_values("match_delta")
            if candidates.empty:
                continue
            historical_index = candidates.index[0]
            historical_id = str(historical_sessions.at[historical_index, "session_id"])
            use_local_window = bool(candidates.at[historical_index, "local_match"]) and not bool(
                candidates.at[historical_index, "absolute_match"]
            )
            effective_start = (
                local_window[0] if use_local_window and local_window is not None else garmin_row.sleep_start
            )
            effective_wake = (
                local_window[1] if use_local_window and local_window is not None else garmin_row.wake_time
            )
            matched_session_ids[str(garmin_row.session_id)] = historical_id
            matched_time_shifts[str(garmin_row.session_id)] = (
                effective_start - garmin_row.sleep_start
            )
            used_historical_ids.add(historical_id)
            historical_sessions.at[historical_index, "sleep_start"] = effective_start
            historical_sessions.at[historical_index, "wake_time"] = effective_wake
            historical_sessions.at[historical_index, "bedtime"] = effective_start
            for column in (
                "gross_minutes",
                "interruption_minutes",
                "sleep_minutes",
                "zero_duration",
                "legacy_garmin_score_10",
            ):
                historical_sessions.at[historical_index, column] = getattr(
                    garmin_row, column
                )
            historical_sessions.at[historical_index, "source"] = "sleepmeter+garmin"
            historical_sessions.at[historical_index, "source_file"] = (
                f"{historical_sessions.at[historical_index, 'source_file']};"
                f"{garmin_row.source_file}"
            )

        for historical_id, override in merge_overrides.items():
            historical_matches = historical_sessions[
                historical_sessions["session_id"] == historical_id
            ]
            if historical_matches.empty:
                continue
            garmin_date = date.fromisoformat(str(override["garmin_date"]))
            garmin_matches = garmin_sessions[
                garmin_sessions["sleep_date"] == garmin_date
            ]
            if garmin_matches.empty:
                continue
            historical_index = historical_matches.index[0]
            garmin_row = next(garmin_matches.itertuples(index=False))
            garmin_id = str(garmin_row.session_id)
            use_garmin_session = bool(override.get("use_garmin_session"))
            effective_start = (
                garmin_row.sleep_start
                if override.get("start") == "garmin"
                else historical_sessions.at[historical_index, "sleep_start"]
            )
            effective_wake = (
                garmin_row.wake_time
                if override.get("end") == "garmin"
                else historical_sessions.at[historical_index, "wake_time"]
            )
            historical_sessions.at[historical_index, "sleep_start"] = effective_start
            historical_sessions.at[historical_index, "bedtime"] = effective_start
            historical_sessions.at[historical_index, "wake_time"] = effective_wake
            if use_garmin_session:
                for column in (
                    "session_type",
                    "gross_minutes",
                    "interruption_minutes",
                    "sleep_minutes",
                    "zero_duration",
                    "legacy_garmin_score_10",
                ):
                    historical_sessions.at[historical_index, column] = getattr(
                        garmin_row, column
                    )
            else:
                sleep_minutes = (
                    effective_wake - effective_start
                ).total_seconds() / 60
                historical_sessions.at[historical_index, "gross_minutes"] = sleep_minutes
                historical_sessions.at[historical_index, "sleep_minutes"] = sleep_minutes
                historical_sessions.at[historical_index, "interruption_minutes"] = 0.0
            historical_sessions.at[
                historical_index, "source"
            ] = "sleepmeter+garmin" if use_garmin_session else "sleepmeter+garmin-stages"
            historical_sessions.at[historical_index, "source_file"] = (
                f"{historical_sessions.at[historical_index, 'source_file']};"
                f"{garmin_row.source_file}"
            )
            if use_garmin_session:
                matched_session_ids[garmin_id] = historical_id
                matched_time_shifts[garmin_id] = timedelta(0)
            else:
                session_interval_mask = historical_intervals["session_id"] == historical_id
                if session_interval_mask.any():
                    first_interval_index = historical_intervals.loc[
                        session_interval_mask, "interval_start"
                    ].idxmin()
                    historical_intervals.at[
                        first_interval_index, "interval_start"
                    ] = effective_start
                subset_stage_session_ids[garmin_id] = historical_id
                subset_time_shifts[garmin_id] = timedelta(0)
                subset_windows[garmin_id] = (
                    garmin_row.sleep_start,
                    garmin_row.wake_time,
                )
            used_historical_ids.add(historical_id)

        for garmin_row in garmin_sessions.itertuples(index=False):
            garmin_id = str(garmin_row.session_id)
            if (
                garmin_row.sleep_date >= garmin_start_date
                or garmin_id in matched_session_ids
                or garmin_id in subset_stage_session_ids
            ):
                continue
            candidates = historical_sessions[
                (historical_sessions["sleep_date"] == garmin_row.sleep_date)
                & (historical_sessions["session_type"] == garmin_row.session_type)
                & (~historical_sessions["session_id"].isin(used_historical_ids))
            ]
            local_window = garmin_local_windows.get(garmin_id)
            subset_candidates: list[
                tuple[float, int, datetime, datetime, timedelta]
            ] = []
            for historical_index, candidate in candidates.iterrows():
                absolute_subset = (
                    garmin_row.sleep_start
                    >= candidate["sleep_start"]
                    - timedelta(seconds=match_tolerance_seconds)
                    and garmin_row.wake_time
                    <= candidate["wake_time"]
                    + timedelta(seconds=match_tolerance_seconds)
                )
                if absolute_subset:
                    uncovered = (
                        garmin_row.sleep_start - candidate["sleep_start"]
                    ).total_seconds() + (
                        candidate["wake_time"] - garmin_row.wake_time
                    ).total_seconds()
                    subset_candidates.append(
                        (
                            uncovered,
                            historical_index,
                            garmin_row.sleep_start,
                            garmin_row.wake_time,
                            timedelta(0),
                        )
                    )
                if local_window is not None:
                    local_start, local_wake = local_window
                    candidate_start = candidate["sleep_start"].replace(tzinfo=None)
                    candidate_wake = candidate["wake_time"].replace(tzinfo=None)
                    local_subset = (
                        local_start.replace(tzinfo=None)
                        >= candidate_start
                        - timedelta(seconds=match_tolerance_seconds)
                        and local_wake.replace(tzinfo=None)
                        <= candidate_wake
                        + timedelta(seconds=match_tolerance_seconds)
                    )
                    if local_subset:
                        uncovered = (
                            local_start.replace(tzinfo=None) - candidate_start
                        ).total_seconds() + (
                            candidate_wake - local_wake.replace(tzinfo=None)
                        ).total_seconds()
                        time_shift = local_start - garmin_row.sleep_start
                        subset_candidates.append(
                            (
                                uncovered,
                                historical_index,
                                local_start,
                                local_wake,
                                time_shift,
                            )
                        )
            if not subset_candidates:
                continue
            _, historical_index, subset_start, subset_wake, time_shift = min(
                subset_candidates, key=lambda item: item[0]
            )
            historical_id = str(historical_sessions.at[historical_index, "session_id"])
            subset_stage_session_ids[garmin_id] = historical_id
            subset_time_shifts[garmin_id] = time_shift
            subset_windows[garmin_id] = (subset_start, subset_wake)
            used_historical_ids.add(historical_id)
            historical_sessions.at[
                historical_index, "source"
            ] = "sleepmeter+garmin-stages"
            historical_sessions.at[historical_index, "source_file"] = (
                f"{historical_sessions.at[historical_index, 'source_file']};"
                f"{garmin_row.source_file}"
            )

    new_garmin_sessions = garmin_sessions[
        garmin_sessions["sleep_date"] >= garmin_start_date
    ]
    # Garmin is authoritative from the handover date: Sleepmeter records from
    # then on would count those nights twice. The exception is a night Garmin
    # did not record (watch not worn or not synced), where Sleepmeter's record
    # is used instead of leaving the day empty.
    garmin_night_dates = (
        set(
            new_garmin_sessions.loc[
                new_garmin_sessions["session_type"] == "NIGHT_SLEEP", "sleep_date"
            ]
        )
        if not new_garmin_sessions.empty
        else set()
    )
    fallback_dates = {
        sleep_date
        for sleep_date in historical_sessions["sleep_date"]
        if sleep_date >= garmin_start_date and sleep_date not in garmin_night_dates
    }
    if not new_garmin_sessions.empty:
        new_garmin_sessions = new_garmin_sessions[
            ~new_garmin_sessions["sleep_date"].isin(fallback_dates)
        ]
    historical_sessions = historical_sessions[
        (historical_sessions["sleep_date"] < garmin_start_date)
        | historical_sessions["sleep_date"].isin(fallback_dates)
    ]
    combined_sessions = pd.concat(
        [historical_sessions, new_garmin_sessions], ignore_index=True
    ).sort_values(["wake_time", "sleep_start"], ignore_index=True)

    accepted_garmin_ids = set(new_garmin_sessions.get("session_id", pd.Series(dtype=str)))
    accepted_garmin_ids.update(matched_session_ids)
    accepted_intervals = garmin_intervals[
        garmin_intervals["session_id"].isin(accepted_garmin_ids)
    ].copy()
    accepted_stages = garmin_stages[
        garmin_stages["session_id"].isin(
            accepted_garmin_ids | set(subset_stage_session_ids)
        )
    ].copy()
    for garmin_id, historical_id in matched_session_ids.items():
        historical_row = historical_sessions[
            historical_sessions["session_id"] == historical_id
        ].iloc[0]
        interval_mask = accepted_intervals["session_id"] == garmin_id
        if interval_mask.any():
            historical_intervals = historical_intervals[
                historical_intervals["session_id"] != historical_id
            ]
        time_shift = matched_time_shifts.get(garmin_id, timedelta(0))
        if time_shift:
            accepted_intervals.loc[interval_mask, "interval_start"] = (
                accepted_intervals.loc[interval_mask, "interval_start"] + time_shift
            )
            accepted_intervals.loc[interval_mask, "interval_end"] = (
                accepted_intervals.loc[interval_mask, "interval_end"] + time_shift
            )
        accepted_intervals.loc[interval_mask, "session_id"] = historical_id
        accepted_intervals.loc[interval_mask, "aid_name"] = historical_row["aid_name"]
        accepted_intervals.loc[
            interval_mask, "legacy_garmin_score_10"
        ] = historical_row["legacy_garmin_score_10"]
        accepted_intervals.loc[interval_mask, "source"] = "sleepmeter+garmin"
        stage_mask = accepted_stages["session_id"] == garmin_id
        if time_shift:
            accepted_stages.loc[stage_mask, "interval_start"] = (
                accepted_stages.loc[stage_mask, "interval_start"] + time_shift
            )
            accepted_stages.loc[stage_mask, "interval_end"] = (
                accepted_stages.loc[stage_mask, "interval_end"] + time_shift
            )
        accepted_stages.loc[stage_mask, "session_id"] = historical_id
        accepted_stages.loc[stage_mask, "source"] = "sleepmeter+garmin"

    unavailable_stage_rows: list[dict[str, object]] = []
    for garmin_id, historical_id in subset_stage_session_ids.items():
        time_shift = subset_time_shifts.get(garmin_id, timedelta(0))
        stage_mask = accepted_stages["session_id"] == garmin_id
        if time_shift:
            accepted_stages.loc[stage_mask, "interval_start"] = (
                accepted_stages.loc[stage_mask, "interval_start"] + time_shift
            )
            accepted_stages.loc[stage_mask, "interval_end"] = (
                accepted_stages.loc[stage_mask, "interval_end"] + time_shift
            )
        accepted_stages.loc[stage_mask, "session_id"] = historical_id
        accepted_stages.loc[stage_mask, "source"] = "sleepmeter+garmin-stages"
        subset_start, subset_wake = subset_windows[garmin_id]
        historical_row = historical_sessions[
            historical_sessions["session_id"] == historical_id
        ].iloc[0]
        for interval in historical_intervals[
            historical_intervals["session_id"] == historical_id
        ].itertuples(index=False):
            if interval.interval_start < subset_start:
                unavailable_end = min(interval.interval_end, subset_start)
                if unavailable_end > interval.interval_start:
                    unavailable_stage_rows.append(
                        {
                            "session_id": historical_id,
                            "sleep_date": historical_row["sleep_date"],
                            "interval_start": interval.interval_start,
                            "interval_end": unavailable_end,
                            "stage": "Stage unavailable",
                            "source": "sleepmeter",
                        }
                    )
            if interval.interval_end > subset_wake:
                unavailable_start = max(interval.interval_start, subset_wake)
                if interval.interval_end > unavailable_start:
                    unavailable_stage_rows.append(
                        {
                            "session_id": historical_id,
                            "sleep_date": historical_row["sleep_date"],
                            "interval_start": unavailable_start,
                            "interval_end": interval.interval_end,
                            "stage": "Stage unavailable",
                            "source": "sleepmeter",
                        }
                    )
    if unavailable_stage_rows:
        accepted_stages = pd.concat(
            [accepted_stages, pd.DataFrame(unavailable_stage_rows)],
            ignore_index=True,
        )

    historical_intervals = historical_intervals[
        (historical_intervals["sleep_date"] < garmin_start_date)
        | historical_intervals["sleep_date"].isin(fallback_dates)
    ]
    combined_intervals = pd.concat(
        [historical_intervals, accepted_intervals], ignore_index=True
    )
    if not combined_intervals.empty:
        combined_intervals = combined_intervals.sort_values(
            ["interval_start", "interval_end", "session_id"], ignore_index=True
        )
    for session_id, score_100 in score_overrides.items():
        session_mask = combined_sessions["session_id"] == session_id
        session_ids = combined_sessions.loc[session_mask, "session_id"]
        combined_sessions.loc[
            session_mask, "legacy_garmin_score_10"
        ] = score_100 / 10
        combined_intervals.loc[
            combined_intervals["session_id"].isin(session_ids),
            "legacy_garmin_score_10",
        ] = score_100 / 10

    for root in roots:
        activities_root = root / "activities"
        if not activities_root.exists():
            continue
        for activity_path in activities_root.glob("*.json"):
            activity_payload = json.loads(
                activity_path.read_text(encoding="utf-8")
            ) or []
            if isinstance(activity_payload, dict):
                activity_payload = [activity_payload]
            for activity in activity_payload:
                activity_id = str(activity.get("activityId") or "")
                start_text = activity.get("startTimeGMT")
                end_text = activity.get("endTimeGMT")
                if not activity_id or not start_text:
                    continue
                activity_start = _garmin_timestamp(start_text, local_timezone)
                duration_seconds = float(activity.get("elapsedDuration") or 0)
                activity_end = (
                    _garmin_timestamp(end_text, local_timezone)
                    if end_text
                    else activity_start + timedelta(seconds=duration_seconds)
                )
                activity_type = activity.get("activityType") or {}
                activities_by_id[activity_id] = {
                    "activity_id": activity_id,
                    "activity_date": activity_start.date(),
                    "activity_start": activity_start,
                    "activity_end": activity_end,
                    "activity_name": activity.get("activityName") or "Activity",
                    "activity_type": activity_type.get("typeKey") or "activity",
                    "duration_seconds": duration_seconds,
                    "calories": activity.get("calories"),
                    "average_hr": activity.get("averageHR"),
                    "max_hr": activity.get("maxHR"),
                    "moderate_intensity_minutes": activity.get(
                        "moderateIntensityMinutes"
                    ),
                    "vigorous_intensity_minutes": activity.get(
                        "vigorousIntensityMinutes"
                    ),
                    "hr_zone_1_seconds": activity.get("hrTimeInZone_1"),
                    "hr_zone_2_seconds": activity.get("hrTimeInZone_2"),
                    "hr_zone_3_seconds": activity.get("hrTimeInZone_3"),
                    "hr_zone_4_seconds": activity.get("hrTimeInZone_4"),
                    "hr_zone_5_seconds": activity.get("hrTimeInZone_5"),
                    "source": "garmin",
                }

    for root in roots:
        heart_rate_root = root / "heart_rate"
        for heart_rate_path in (
            heart_rate_root.glob("*.json") if heart_rate_root.exists() else []
        ):
            payload = json.loads(heart_rate_path.read_text(encoding="utf-8")) or {}
            for timestamp_ms, value in payload.get("heartRateValues") or []:
                timestamp = datetime.fromtimestamp(
                    timestamp_ms / 1000, tz=UTC
                ).astimezone(local_timezone)
                row = intraday_rows.setdefault(
                    timestamp,
                    {"timestamp": timestamp, "date": timestamp.date()},
                )
                row["heart_rate"] = value if value and value > 0 else None

        stress_root = root / "stress"
        for stress_path in stress_root.glob("*.json") if stress_root.exists() else []:
            payload = json.loads(stress_path.read_text(encoding="utf-8")) or {}
            for timestamp_ms, value in payload.get("stressValuesArray") or []:
                timestamp = datetime.fromtimestamp(
                    timestamp_ms / 1000, tz=UTC
                ).astimezone(local_timezone)
                row = intraday_rows.setdefault(
                    timestamp,
                    {"timestamp": timestamp, "date": timestamp.date()},
                )
                row["stress"] = value if value is not None and value >= 0 else None
            for sample in payload.get("bodyBatteryValuesArray") or []:
                if len(sample) < 3:
                    continue
                timestamp = datetime.fromtimestamp(
                    sample[0] / 1000, tz=UTC
                ).astimezone(local_timezone)
                row = intraday_rows.setdefault(
                    timestamp,
                    {"timestamp": timestamp, "date": timestamp.date()},
                )
                row["body_battery"] = sample[2]

    for root in roots:
        menstrual_daily_root = root / "menstrual" / "daily"
        if not menstrual_daily_root.exists():
            continue
        for menstrual_path in menstrual_daily_root.glob("*.json"):
            try:
                menstrual_date = date.fromisoformat(menstrual_path.stem)
            except ValueError:
                continue
            menstrual_payload = json.loads(
                menstrual_path.read_text(encoding="utf-8")
            ) or {}
            summary = menstrual_payload.get("daySummary") or {}
            cycle_day = summary.get("dayInCycle")
            if cycle_day is None:
                continue
            menstrual_by_date[menstrual_date] = {
                "date": menstrual_date,
                "cycle_day": int(cycle_day),
                "cycle_start_date": summary.get("startDate"),
                "current_phase": summary.get("currentPhase"),
                "predicted_cycle_length": summary.get("predictedCycleLength"),
                "predicted_cycle": summary.get("predictedCycle"),
                "source": "garmin",
            }

    coverage_end = max(historical.coverage_end, max(garmin_dates))
    historical_daily = {
        row.date: row._asdict() for row in historical.daily.itertuples(index=False)
    }
    daily_rows: list[dict[str, object]] = []
    for current_date in pd.date_range(
        historical.coverage_start, coverage_end, freq="D"
    ).date:
        if current_date < garmin_start_date or current_date in fallback_dates:
            day_sessions = combined_sessions[
                combined_sessions["sleep_date"] == current_date
            ]
            night_minutes = day_sessions.loc[
                day_sessions["session_type"] == "NIGHT_SLEEP", "sleep_minutes"
            ].sum()
            nap_minutes = day_sessions.loc[
                day_sessions["session_type"] == "NAP", "sleep_minutes"
            ].sum()
            total_minutes = float(night_minutes + nap_minutes)
            if day_sessions.empty:
                source_state = "sleepmeter_no_session_zero"
            elif total_minutes == 0:
                source_state = "sleepmeter_zero_placeholder"
            elif day_sessions["source"].str.startswith("sleepmeter+garmin").any():
                source_state = "sleepmeter_garmin_matched"
            else:
                source_state = "sleepmeter_recorded"
            if current_date >= garmin_start_date:
                source_state = "sleepmeter_fallback"
            daily_rows.append(
                {
                    "date": current_date,
                    "night_minutes": float(night_minutes),
                    "nap_minutes": float(nap_minutes),
                    "total_sleep_minutes": total_minutes,
                    "session_count": int(len(day_sessions)),
                    "zero_placeholder_count": int(day_sessions["zero_duration"].sum()),
                    "zero_sleep": total_minutes == 0,
                    "source_state": source_state,
                    "source": "sleepmeter+garmin" if source_state == "sleepmeter_garmin_matched" else "sleepmeter",
                }
            )
        elif current_date in daily_by_date:
            daily_rows.append(daily_by_date[current_date])
        else:
            daily_rows.append(
                {
                    "date": current_date,
                    "night_minutes": float("nan"),
                    "nap_minutes": float("nan"),
                    "total_sleep_minutes": float("nan"),
                    "session_count": 0,
                    "zero_placeholder_count": 0,
                    "zero_sleep": False,
                    "source_state": "garmin_not_fetched",
                    "source": "garmin",
                }
            )

    return SleepDataset(
        source_path=historical.source_path,
        coverage_start=historical.coverage_start,
        coverage_end=coverage_end,
        aids=historical.aids,
        sessions=combined_sessions,
        intervals=combined_intervals,
        sleep_stages=accepted_stages,
        activities=pd.DataFrame(
            activities_by_id.values(),
            columns=[
                "activity_id",
                "activity_date",
                "activity_start",
                "activity_end",
                "activity_name",
                "activity_type",
                "duration_seconds",
                "calories",
                "average_hr",
                "max_hr",
                "moderate_intensity_minutes",
                "vigorous_intensity_minutes",
                "hr_zone_1_seconds",
                "hr_zone_2_seconds",
                "hr_zone_3_seconds",
                "hr_zone_4_seconds",
                "hr_zone_5_seconds",
                "source",
            ],
        ),
        menstrual=pd.DataFrame(
            menstrual_by_date.values(),
            columns=[
                "date",
                "cycle_day",
                "cycle_start_date",
                "current_phase",
                "predicted_cycle_length",
                "predicted_cycle",
                "source",
            ],
        ),
        intraday=pd.DataFrame(
            intraday_rows.values(),
            columns=["timestamp", "date", "heart_rate", "stress", "body_battery"],
        ).sort_values("timestamp", ignore_index=True),
        daily=pd.DataFrame(daily_rows),
        garmin_metrics=pd.DataFrame(metrics),
    )


def split_intervals_by_calendar_day(intervals: pd.DataFrame) -> pd.DataFrame:
    """Split only the chart representation at midnight; sessions stay intact."""
    segments: list[dict[str, object]] = []
    for row in intervals.itertuples(index=False):
        cursor = row.interval_start
        while cursor < row.interval_end:
            next_midnight = datetime.combine(
                cursor.date() + timedelta(days=1), time.min, tzinfo=cursor.tzinfo
            )
            segment_end = min(row.interval_end, next_midnight)
            start_minute = cursor.hour * 60 + cursor.minute + cursor.second / 60
            end_minute = (
                1440.0
                if segment_end == next_midnight
                else segment_end.hour * 60
                + segment_end.minute
                + segment_end.second / 60
            )
            segments.append(
                {
                    "session_id": row.session_id,
                    "calendar_date": cursor.date(),
                    "start_minute": start_minute,
                    "end_minute": end_minute,
                    "interval_start": row.interval_start,
                    "interval_end": row.interval_end,
                    "session_type": row.session_type,
                    "aid_name": row.aid_name,
                    "legacy_garmin_score_10": row.legacy_garmin_score_10,
                    "source": row.source,
                }
            )
            cursor = segment_end
    return pd.DataFrame(segments)
