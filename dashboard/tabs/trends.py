"""Trends & distributions: histograms, pies, credit/debit and calendars."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    TARGET_SLEEP_HOURS,
    calendar_activity_heatmap,
    calendar_intensity_minutes_heatmap,
    calendar_sleep_heatmap,
    calendar_sleep_score_heatmap,
    credit_debit_figure,
    daily_duration_histogram,
    nightly_duration_histogram,
    sleep_nap_pie_figure,
    sleep_state_pie_figure,
)
from dashboard.tabs import ViewData
from dashboard.ui import format_duration


def render(view: ViewData) -> None:
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
    credit_target_minutes = st.select_slider(
        "Daily sleep target",
        options=range(5 * 60, 10 * 60 + 1, 15),
        value=int(TARGET_SLEEP_HOURS * 60),
        format_func=format_duration,
        key="credit_debit_target_minutes",
    )
    st.plotly_chart(
        credit_debit_figure(view.selected.daily, credit_target_minutes / 60),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        f"Sleepmeter-style credit/debit uses a {format_duration(credit_target_minutes)} "
        "daily target and a 24-hour "
        "balance scale: meeting the target is neutral and zero sleep contributes "
        "−24:00. The shaded region is the mean cumulative balance ± one standard "
        "deviation; the solid diagonal line is the overall trend."
    )

    activity_heatmap_column, intensity_heatmap_column = st.columns(2)
    with activity_heatmap_column:
        st.plotly_chart(
            calendar_activity_heatmap(view.selected.daily, view.selected.activities),
            width="stretch",
            config={"displaylogo": False},
        )
    with intensity_heatmap_column:
        st.plotly_chart(
            calendar_intensity_minutes_heatmap(
                view.selected.daily, view.selected.garmin_metrics
            ),
            width="stretch",
            config={"displaylogo": False},
        )
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
