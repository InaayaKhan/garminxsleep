"""Which saved Garmin files the daily sync re-downloads."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

from dashboard.garmin_sync import _needs_refresh


def _saved(tmp_path, relative_path, value, saved_at):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.utime(path, (saved_at.timestamp(), saved_at.timestamp()))
    return path


DAY = date(2026, 5, 11)
DURING_DAY = datetime(2026, 5, 11, 7, 30)
AFTER_DAY = datetime(2026, 5, 12, 7, 30)
SLEEP = {"dailySleepDTO": {"sleepTimeSeconds": 28800}}


def test_file_saved_after_its_day_ended_is_final(tmp_path):
    path = _saved(tmp_path, f"sleep/{DAY}.json", SLEEP, AFTER_DAY)
    assert not _needs_refresh(f"sleep/{DAY}.json", path)


def test_file_saved_during_its_day_is_fetched_again(tmp_path):
    # Morning sleep file: a nap later that day isn't in it yet.
    path = _saved(tmp_path, f"sleep/{DAY}.json", SLEEP, DURING_DAY)
    assert _needs_refresh(f"sleep/{DAY}.json", path)


def test_empty_file_is_fetched_again_even_after_its_day(tmp_path):
    # The watch hadn't synced to Garmin yet when this was saved.
    empty_sleep = {"dailySleepDTO": {"sleepTimeSeconds": None}}
    for relative_path, value in (
        (f"sleep/{DAY}.json", empty_sleep),
        (f"hrv/{DAY}.json", {}),
        (f"heart_rate/{DAY}.json", {"heartRateValues": []}),
        (f"steps/{DAY}.json", []),
        (f"spo2/{DAY}.json", None),
    ):
        path = _saved(tmp_path, relative_path, value, AFTER_DAY)
        assert _needs_refresh(relative_path, path), relative_path


def test_range_file_is_final_once_its_last_day_ended(tmp_path):
    relative_path = f"activities/{DAY - timedelta(days=2)}_{DAY}.json"
    activities = [{"activityId": 1}]
    assert _needs_refresh(relative_path, _saved(tmp_path, relative_path, activities, DURING_DAY))
    assert not _needs_refresh(relative_path, _saved(tmp_path, relative_path, activities, AFTER_DAY))


def test_undated_file_is_kept_when_not_empty(tmp_path):
    relative_path = "activity_details/123.json"
    path = _saved(tmp_path, relative_path, {"activityId": 123}, DURING_DAY)
    assert not _needs_refresh(relative_path, path)
