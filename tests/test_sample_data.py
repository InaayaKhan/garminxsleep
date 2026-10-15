"""Invariants of the merged dataset and key charts, checked on sample_data/.

These test rules that must hold whatever the data, such as "session totals
equal daily totals", rather than chart details like labels or colours. The
dated cases (watch-off nights, zero-sleep days) are set up deliberately in
scripts/generate_sample_data.py.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from dashboard.charts import (
    aid_sleep_stage_composition_figure,
    daily_duration_figure,
    hour_px,
    sleep_stage_schedule_figure,
)
from dashboard.model import load_combined_sleep_data

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"
SETTINGS = json.loads((SAMPLE / "settings.json").read_text(encoding="utf-8"))
GARMIN_START = date.fromisoformat(SETTINGS["garmin_start_date"])


@pytest.fixture(scope="module")
def dataset():
    return load_combined_sleep_data(
        next(SAMPLE.glob("sleep_history_*.csv")),
        [SAMPLE / "raw"],
        GARMIN_START,
        aid_overrides_path=SAMPLE / "garmin_aid_overrides.json",
        timezone_name=SETTINGS["timezone"],
    )


def _minutes_by_date(sessions: pd.DataFrame, session_type: str) -> pd.Series:
    return (
        sessions[sessions["session_type"] == session_type]
        .groupby("sleep_date")["sleep_minutes"]
        .sum()
    )


def test_session_totals_match_daily_totals(dataset):
    daily = dataset.daily.set_index("date")
    for session_type, column in (("NIGHT_SLEEP", "night_minutes"), ("NAP", "nap_minutes")):
        from_sessions = _minutes_by_date(dataset.sessions, session_type)
        compared = pd.DataFrame(
            {"daily": daily[column], "sessions": from_sessions}
        ).fillna(0)
        mismatched = compared[(compared["daily"] - compared["sessions"]).abs() > 0.5]
        assert mismatched.empty, f"{session_type} totals differ on {list(mismatched.index)}"


def test_no_night_is_counted_twice(dataset):
    nights = dataset.sessions[dataset.sessions["session_type"] == "NIGHT_SLEEP"]
    after_handover = nights[nights["sleep_date"] >= GARMIN_START]
    assert not after_handover["sleep_date"].duplicated().any()


def test_sleepmeter_is_only_used_after_handover_as_a_fallback(dataset):
    daily = dataset.daily.set_index("date")
    sessions = dataset.sessions
    later_sleepmeter = sessions[
        (sessions["sleep_date"] >= GARMIN_START)
        & sessions["source"].str.startswith("sleepmeter")
    ]
    for sleep_date in later_sleepmeter["sleep_date"].unique():
        assert daily.loc[sleep_date, "source_state"] == "sleepmeter_fallback"


def test_garmin_naps_become_sessions_with_intervals(dataset):
    sessions = dataset.sessions
    garmin_naps = sessions[(sessions["session_type"] == "NAP") & (sessions["source"] == "garmin")]
    assert not garmin_naps.empty
    assert (garmin_naps["sleep_date"] >= GARMIN_START).all()
    nap_intervals = dataset.intervals[dataset.intervals["session_id"].isin(garmin_naps["session_id"])]
    assert set(nap_intervals["session_id"]) == set(garmin_naps["session_id"])


def test_watch_off_night_falls_back_to_sleepmeter(dataset):
    day = dataset.daily.set_index("date").loc[date(2026, 6, 3)]
    assert day["source_state"] == "sleepmeter_fallback"
    assert day["night_minutes"] > 0
    assert not day["zero_sleep"]


def test_night_with_no_record_is_unknown_not_zero(dataset):
    day = dataset.daily.set_index("date").loc[date(2026, 7, 14)]
    assert math.isnan(day["total_sleep_minutes"])
    assert not day["zero_sleep"]


def test_missing_and_placeholder_sleepmeter_days_are_zero(dataset):
    daily = dataset.daily.set_index("date")
    assert daily.loc[date(2026, 4, 17), "source_state"] == "sleepmeter_no_session_zero"
    assert daily.loc[date(2026, 3, 28), "source_state"] == "sleepmeter_zero_placeholder"
    assert daily.loc[date(2026, 4, 17), "zero_sleep"]
    assert daily.loc[date(2026, 3, 28), "zero_sleep"]


def test_overlapping_nights_are_matched_before_handover(dataset):
    matched = dataset.daily[dataset.daily["source_state"] == "sleepmeter_garmin_matched"]
    assert len(matched) > 0
    assert (matched["date"] < GARMIN_START).all()


def test_interruptions_reduce_asleep_time_only(dataset):
    sleepmeter = dataset.sessions[dataset.sessions["source"] == "sleepmeter"]
    asleep_plus_awake = sleepmeter["sleep_minutes"] + sleepmeter["interruption_minutes"]
    assert (asleep_plus_awake - sleepmeter["gross_minutes"]).abs().max() < 1e-6
    assert (sleepmeter["interruption_minutes"] >= 0).all()


def test_aids_without_stage_data_are_labelled_not_dropped(dataset):
    # April-June: Melatonin was only used in April, before Garmin recorded
    # stages, while other aids in the range do have staged nights.
    sessions = dataset.sessions
    in_range = sessions[
        (sessions["sleep_date"] >= date(2026, 4, 1))
        & (sessions["sleep_date"] <= date(2026, 6, 30))
    ]
    stages = dataset.sleep_stages[
        dataset.sleep_stages["session_id"].isin(in_range["session_id"])
    ]
    figure = aid_sleep_stage_composition_figure(in_range, stages)
    labels = list(figure.data[0].x)
    assert any(label.startswith("Melatonin<br>0 of ") for label in labels)
    assert "No stage data" in [annotation.text for annotation in figure.layout.annotations]


def _overview_charts(dataset, view_start, view_end):
    duration = daily_duration_figure(
        dataset.daily, dataset.sessions, dataset.garmin_metrics, dataset.menstrual,
        dataset.sleep_stages, dataset.activities, show_sleep_stages=True,
        view_start=view_start, view_end=view_end,
    )
    stages = sleep_stage_schedule_figure(
        dataset.sessions, dataset.intervals, dataset.sleep_stages, dataset.activities,
        dataset.menstrual, dataset.daily, dataset.coverage_start, dataset.coverage_end,
        garmin_metrics=dataset.garmin_metrics, match_hour_px=hour_px(duration),
        view_start=view_start, view_end=view_end,
    )
    return duration, stages


def test_overview_charts_open_on_view_and_pan_over_all_data(dataset):
    view_start, view_end = date(2026, 6, 1), date(2026, 6, 30)
    for figure in _overview_charts(dataset, view_start, view_end):
        start, end = figure.layout.xaxis.range
        assert pd.Timestamp(start) == pd.Timestamp(view_start)
        assert pd.Timestamp(end) == pd.Timestamp(view_end) + pd.Timedelta(days=1)
        assert pd.Timestamp(figure.layout.xaxis.minallowed) == pd.Timestamp(dataset.coverage_start)
        assert figure.layout.meta["pan"] is True


def test_duration_and_stage_charts_share_an_hour_scale(dataset):
    duration, stages = _overview_charts(dataset, None, None)
    assert hour_px(stages) == pytest.approx(hour_px(duration), rel=0.01)


def test_hidden_hrv_lines_still_get_a_real_axis(dataset):
    _, stages = _overview_charts(dataset, None, None)
    low, high = stages.layout.yaxis2.range
    top_hrv = dataset.garmin_metrics["hrv_last_night_average"].max()
    assert low == 0 and high >= top_hrv
