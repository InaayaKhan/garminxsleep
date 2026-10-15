"""Activity & sleep: activity calendars and the sleep that follows."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    activity_effect_on_sleep_figure,
    activity_intensity_effect_figure,
    activity_to_sleep_gap_figure,
    activity_type_comparison_figure,
    calendar_activity_heatmap,
    calendar_intensity_minutes_heatmap,
    intensity_minutes_vs_subsequent_sleep_figure,
    steps_floors_vs_subsequent_sleep_figure,
)
from dashboard.tabs import ViewData


def render(view: ViewData) -> None:
    st.markdown(
        "Your activity, and how the following night compares. Each activity or day is paired with the next night's sleep."
    )
    st.subheader("Activity calendars")
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
    st.subheader("Activities vs the following night")
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
    st.subheader("Daily load vs the following night")
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
    st.subheader("Time from activity to sleep")
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
