"""Streamlit entry point: load the data, set up the sidebar, render the tabs."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from dashboard.config import DATABASE_PATH
from dashboard.data import (
    cached_dataset,
    export_database,
    find_export,
    garmin_archive_signature,
)
from dashboard.tabs import (
    ViewData,
    activity,
    aids,
    frames_between,
    garmin,
    overview,
    patterns,
    records,
    sleep_quality,
    timing,
)
from dashboard.ui import install_chart_defaults

# Quick date ranges: label -> number of most recent days (None = everything).
DATE_PRESETS = {"30 days": 30, "90 days": 90, "180 days": 180, "All": None}
DEFAULT_PRESET = "180 days"

st.set_page_config(
    page_title="garminxsleep",
    page_icon="🌙",
    layout="wide",
    # Charts get the full width; the date range and summary are one click away.
    initial_sidebar_state="collapsed",
)
install_chart_defaults()
st.title("garminxsleep")
st.sidebar.caption("Local Garmin and Sleepmeter dashboard")

try:
    export_path = find_export()
    data_version = (
        str(export_path),
        export_path.stat().st_mtime_ns,
        garmin_archive_signature(),
    )
    dataset = cached_dataset(*data_version)
except Exception as error:
    st.error(f"Unable to load the Sleepmeter export: {error}")
    st.stop()

# The DuckDB copy is only for SQL analysis; the charts never read it, so a
# failed write is reported but does not stop the dashboard.
try:
    export_database(*data_version, dataset)
except Exception as error:
    st.sidebar.warning(f"Could not update {DATABASE_PATH.name}: {error}")

# Date range above the tabs, since it applies to all of them. Presets cover the
# common cases; editing the dates by hand switches the preset to "Custom".
coverage = (dataset.coverage_start, dataset.coverage_end)


def _preset_range(preset: str) -> tuple[date, date]:
    days = DATE_PRESETS[preset]
    if days is None:
        return coverage
    return max(coverage[0], coverage[1] - timedelta(days=days - 1)), coverage[1]


def _apply_preset() -> None:
    preset = st.session_state.date_preset
    if preset in DATE_PRESETS:
        st.session_state.date_range = _preset_range(preset)


def _mark_custom() -> None:
    st.session_state.date_preset = "Custom"


stored = st.session_state.get("date_range")
if (
    not isinstance(stored, tuple)
    or not stored
    or not all(coverage[0] <= day <= coverage[1] for day in stored)
):
    # First load, or the data's coverage changed: start from the default preset.
    st.session_state.date_preset = DEFAULT_PRESET
    st.session_state.date_range = _preset_range(DEFAULT_PRESET)

preset_column, range_column = st.columns([3, 2])
with preset_column:
    st.segmented_control(
        "Date range",
        options=[*DATE_PRESETS, "Custom"],
        key="date_preset",
        on_change=_apply_preset,
    )
with range_column:
    st.date_input(
        "Dates",
        key="date_range",
        min_value=coverage[0],
        max_value=coverage[1],
        on_change=_mark_custom,
        label_visibility="hidden",
    )
selected_range = st.session_state.date_range
if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
elif isinstance(selected_range, tuple) and len(selected_range) == 1:
    # While picking a custom range only the first date is set.
    start_date = end_date = selected_range[0]
else:
    start_date, end_date = coverage

# The Overview date charts get all history up to the end date and open on the
# selected range; earlier days are reached by panning.
view = ViewData(
    dataset=dataset,
    start_date=start_date,
    end_date=end_date,
    selected=frames_between(dataset, start_date, end_date),
    history=frames_between(dataset, None, end_date),
)


# Core sleep views first, then what may influence sleep, then Garmin-only
# measurements; the raw records table comes last.
overview_tab, patterns_tab, timing_tab, sleep_quality_tab, aids_tab, activity_tab, garmin_tab, records_tab = st.tabs(
    [
        "Overview",
        "Patterns",
        "Timing",
        "Sleep quality",
        "Aids",
        "Activity & sleep",
        "Garmin",
        "Records",
    ]
)

with overview_tab:
    overview.render(view)

with patterns_tab:
    patterns.render(view)

with timing_tab:
    timing.render(view)

with sleep_quality_tab:
    sleep_quality.render(view)

with aids_tab:
    aids.render(view)

with activity_tab:
    activity.render(view)

with garmin_tab:
    garmin.render(view)

with records_tab:
    records.render(view)

st.sidebar.divider()
st.sidebar.caption(f"Source: {export_path.name}")
st.sidebar.caption(
    f"Coverage: {dataset.coverage_start.isoformat()} to {dataset.coverage_end.isoformat()}"
)
st.sidebar.caption(f"Local database: {DATABASE_PATH.name}")
