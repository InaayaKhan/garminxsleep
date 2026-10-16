"""Patterns: how sleep is distributed across nights and days."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    daily_duration_histogram,
    nightly_duration_histogram,
    sleep_nap_pie_figure,
    sleep_state_pie_figure,
)
from dashboard.tabs import ViewData


DURATION_INFO = (
    "Histograms use one-hour bins. Nightly duration excludes naps; daily "
    "duration includes naps. Zero-sleep observations remain included."
)


SLEEP_AND_NAPS_INFO = (
    "Sleep and naps use recorded asleep time. Awake in holes uses Sleepmeter "
    "interruptions; Awake is the remaining time across the selected days. The "
    "first pie compares night sleep with naps only."
)


def render(view: ViewData) -> None:
    st.markdown(
        'How your sleep is distributed: typical durations and sleep versus naps.'
    )
    st.subheader("Duration", help=DURATION_INFO)
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
    st.subheader("Sleep and naps", help=SLEEP_AND_NAPS_INFO)
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
