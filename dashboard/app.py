"""Streamlit entry point: load the data, set up the sidebar, render the tabs."""

from __future__ import annotations

import base64
from datetime import date, timedelta
from pathlib import Path

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
from dashboard.ui import format_duration, install_chart_defaults

# Quick date ranges: label -> number of most recent days (None = everything).
DATE_PRESETS = {"30 days": 30, "90 days": 90, "180 days": 180, "All": None}
DEFAULT_PRESET = "180 days"

ASSETS = Path(__file__).resolve().parent / "assets"
# Brand colours of the "GarminxSleep" wordmark, matching the logo.
BRAND_X_COLOR = "#FFB547"  # ovulation yellow
BRAND_WORD_COLOR = "#EE55C0"  # REM sleep pink


def render_header() -> None:
    """Logo with "GarminxSleep" beside it, in Inter (system sans-serif offline)."""
    logo = base64.b64encode((ASSETS / "logo_header.png").read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500&display=swap');
        .gxs-header {{ display: flex; align-items: center; gap: 0.8rem; margin: 0 0 1.5rem; }}
        .gxs-header img {{ height: 2.3rem; }}
        .gxs-wordmark {{
            font-family: 'Inter', system-ui, sans-serif; font-weight: 500;
            font-size: 1.35rem; line-height: 1;
            color: {BRAND_WORD_COLOR};
        }}
        .gxs-wordmark .x {{ color: {BRAND_X_COLOR}; }}
        </style>
        <div class="gxs-header">
          <img src="data:image/png;base64,{logo}" alt="GarminxSleep logo">
          <span class="gxs-wordmark">Garmin<span class="x">x</span>Sleep</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(
    page_title="GarminxSleep",
    page_icon=str(ASSETS / "logo_icon.png"),
    layout="wide",
    # Charts get the full width; the date range and summary are one click away.
    initial_sidebar_state="collapsed",
)
install_chart_defaults()
render_header()
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

# Date controls on the left (presets, then the exact dates below them), and
# the summary of the selected range on the right.
controls_column, summary_column = st.columns([2, 3], gap="large", vertical_alignment="center")
with controls_column:
    st.segmented_control(
        "Date range",
        options=[*DATE_PRESETS, "Custom"],
        key="date_preset",
        on_change=_apply_preset,
        width="stretch",
    )
    st.date_input(
        "Dates",
        key="date_range",
        min_value=coverage[0],
        max_value=coverage[1],
        on_change=_mark_custom,
        label_visibility="collapsed",
        # Day first, matching the charts' date axes.
        format="DD.MM.YYYY",
        width="stretch",
    )
selected_range = st.session_state.date_range
if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
elif isinstance(selected_range, tuple) and len(selected_range) == 1:
    # While picking a custom range only the first date is set.
    start_date = end_date = selected_range[0]
else:
    start_date, end_date = coverage

# A preset like "30 days" already says how long the range is; for a custom
# range or "All", show it in brackets after the dates in the picker. Streamlit
# can't add text to the field itself, so CSS draws it there.
if st.session_state.date_preset not in DATE_PRESETS or DATE_PRESETS[st.session_state.date_preset] is None:
    day_count = (end_date - start_date).days + 1
    st.markdown(
        f"""
        <style>
        .st-key-date_range [data-testid="stDateInputFieldsScroller"]::after {{
            content: "({day_count} {'day' if day_count == 1 else 'days'})";
            align-self: center; margin-left: 0.75rem; white-space: nowrap;
            color: rgba(250, 250, 250, 0.55); font-size: 0.875rem; pointer-events: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# The Overview date charts get all history up to the end date and open on the
# selected range; earlier days are reached by panning.
view = ViewData(
    dataset=dataset,
    start_date=start_date,
    end_date=end_date,
    selected=frames_between(dataset, start_date, end_date),
    history=frames_between(dataset, None, end_date),
)

st.markdown(
    """
    <style>
    .st-key-summary [data-testid="stMetric"] { text-align: center; }
    .st-key-summary [data-testid="stMetricLabel"] { display: flex; justify-content: center; }
    .st-key-summary [data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
with summary_column, st.container(key="summary"):
    daily = view.selected.daily
    days = len(daily)
    average_minutes = daily["total_sleep_minutes"].mean() if days else 0
    average_column, zero_column, sessions_column = st.columns(3)
    average_column.metric("Average daily sleep", format_duration(float(average_minutes)))
    zero_column.metric("Zero-sleep days", int(daily["zero_sleep"].sum()) if days else 0)
    sessions_column.metric("Recorded sessions", len(view.selected.sessions))


# Core sleep views first, then what may influence sleep, then Garmin-only
# measurements; the raw records table comes last.
overview_tab, patterns_tab, timing_tab, sleep_quality_tab, aids_tab, activity_tab, garmin_tab, records_tab = st.tabs(
    [
        "Overview",
        "Patterns",
        "Timing",
        "Sleep quality",
        "Aids & hindrances",
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
    f"Coverage: {dataset.coverage_start:%d.%m.%Y} to {dataset.coverage_end:%d.%m.%Y}"
)
st.sidebar.caption(f"Local database: {DATABASE_PATH.name}")
