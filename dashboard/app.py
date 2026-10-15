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

DEFAULT_RANGE_DAYS = 180

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

st.sidebar.header("Date range")
# Open on the most recent DEFAULT_RANGE_DAYS; earlier dates stay selectable.
default_start = max(
    dataset.coverage_start,
    dataset.coverage_end - timedelta(days=DEFAULT_RANGE_DAYS - 1),
)
selected_range = st.sidebar.date_input(
    "Sleep dates",
    value=(default_start, dataset.coverage_end),
    min_value=dataset.coverage_start,
    max_value=dataset.coverage_end,
)
if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date = end_date = selected_range if isinstance(selected_range, date) else dataset.coverage_end

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
