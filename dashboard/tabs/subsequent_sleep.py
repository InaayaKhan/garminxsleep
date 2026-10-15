"""Subsequent sleep: activity and daily load vs the following night."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    activity_effect_on_sleep_figure,
    activity_intensity_effect_figure,
    activity_to_sleep_gap_figure,
    activity_type_comparison_figure,
    intensity_minutes_vs_subsequent_sleep_figure,
    steps_floors_vs_subsequent_sleep_figure,
)
from dashboard.tabs import ViewData


def render(view: ViewData) -> None:
    activity_predictor = st.radio(
        "Activity predictor",
        ["Activity duration", "Time from activity to sleep"],
        horizontal=True,
    )
    st.plotly_chart(
        activity_effect_on_sleep_figure(
            view.selected.activities,
            view.selected.sessions,
            view.selected.garmin_metrics,
            predictor=(
                "gap"
                if activity_predictor == "Time from activity to sleep"
                else "duration"
            ),
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Each activity day is paired with the first night sleep starting after that "
        "day's final recorded activity. Lines show simple overall trends, not causation."
    )
    st.plotly_chart(
        activity_type_comparison_figure(
            view.selected.activities, view.selected.sessions, view.selected.garmin_metrics
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        activity_intensity_effect_figure(
            view.selected.activities, view.selected.sessions, view.selected.garmin_metrics
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        intensity_minutes_vs_subsequent_sleep_figure(
            view.selected.garmin_metrics, view.selected.sessions
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    st.plotly_chart(
        steps_floors_vs_subsequent_sleep_figure(
            view.selected.garmin_metrics, view.selected.sessions
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    st.plotly_chart(
        activity_to_sleep_gap_figure(view.selected.activities, view.selected.sessions),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    st.plotly_chart(
        activity_to_sleep_gap_figure(
            view.selected.activities,
            view.selected.sessions,
            exclude_daytime_activities=True,
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
