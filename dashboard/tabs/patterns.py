"""Patterns: how sleep is distributed across nights and days."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    calendar_sleep_heatmap,
    calendar_sleep_score_heatmap,
    daily_duration_histogram,
    nightly_duration_histogram,
    sleep_nap_pie_figure,
    sleep_state_pie_figure,
)
from dashboard.tabs import ViewData


def render(view: ViewData) -> None:
    st.markdown(
        'How your sleep is distributed: typical durations, sleep versus naps, and day-by-day calendars.'
    )
    st.subheader("Duration")
    nightly_histogram_column, daily_histogram_column = st.columns(2)
    with nightly_histogram_column:
        st.plotly_chart(
            nightly_duration_histogram(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with daily_histogram_column:
        st.plotly_chart(
            daily_duration_histogram(view.selected.daily),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Histograms use one-hour bins. Nightly duration excludes naps; daily "
        "duration includes naps. Zero-sleep observations remain included."
    )
    st.subheader("Sleep and naps")
    sleep_nap_column, time_breakdown_column = st.columns(2)
    with sleep_nap_column:
        st.plotly_chart(
            sleep_nap_pie_figure(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with time_breakdown_column:
        st.plotly_chart(
            sleep_state_pie_figure(view.selected.sessions, view.selected.daily),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Sleep and naps use recorded asleep time. Awake in holes uses Sleepmeter "
        "interruptions; Awake is the remaining time across the selected days. The "
        "first pie compares night sleep with naps only."
    )
    st.subheader("Calendars")
    duration_heatmap_column, score_heatmap_column = st.columns(2)
    with duration_heatmap_column:
        st.plotly_chart(
            calendar_sleep_heatmap(view.selected.daily),
            width="stretch",
            config={"displaylogo": False},
        )
    with score_heatmap_column:
        st.plotly_chart(
            calendar_sleep_score_heatmap(view.selected.daily, view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
