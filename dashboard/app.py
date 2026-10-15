"""Streamlit entry point: load the data, set up the sidebar, render the tabs."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from dashboard.config import DATABASE_PATH
from dashboard.data import cached_dataset, find_export, garmin_archive_signature
from dashboard.storage import persist_dataset
from dashboard.tabs import (
    ViewData,
    aids,
    frames_between,
    garmin,
    overview,
    records,
    relationships,
    subsequent_sleep,
    timing,
    trends,
)
from dashboard.ui import format_duration, install_chart_defaults

DEFAULT_RANGE_DAYS = 180

st.set_page_config(page_title="garminxsleep", page_icon="🌙", layout="wide")
install_chart_defaults()
st.title("garminxsleep")
st.sidebar.caption("Local Garmin and Sleepmeter dashboard")

try:
    export_path = find_export()
    dataset = cached_dataset(
        str(export_path),
        export_path.stat().st_mtime_ns,
        garmin_archive_signature(),
    )
    persist_dataset(dataset, DATABASE_PATH)
except Exception as error:
    st.error(f"Unable to load the Sleepmeter export: {error}")
    st.stop()

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

days = len(view.selected.daily)
average_minutes = view.selected.daily["total_sleep_minutes"].mean() if days else 0
zero_days = int(view.selected.daily["zero_sleep"].sum()) if days else 0
session_count = len(view.selected.sessions)

st.sidebar.divider()
st.sidebar.subheader("Summary")
st.sidebar.metric("Calendar days", days)
st.sidebar.metric("Average daily sleep", format_duration(float(average_minutes)))
st.sidebar.metric("Zero-sleep days", zero_days)
st.sidebar.metric("Recorded sessions", session_count)

# Core sleep views first, then what may influence sleep, then Garmin-only
# measurements; the raw records table comes last.
overview_tab, distributions_tab, timing_tab, aids_tab, subsequent_sleep_tab, relationships_tab, garmin_tab, records_tab = st.tabs(
    [
        "Overview",
        "Trends & distributions",
        "Timing",
        "Aids",
        "Subsequent sleep",
        "Relationships",
        "Garmin",
        "Records",
    ]
)

with overview_tab:
    overview.render(view)

with distributions_tab:
    trends.render(view)

with timing_tab:
    timing.render(view)

with aids_tab:
    aids.render(view)

with subsequent_sleep_tab:
    subsequent_sleep.render(view)

with relationships_tab:
    relationships.render(view)

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
