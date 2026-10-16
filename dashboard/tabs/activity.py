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


# Shown behind the info icon on the section heading.
PAIRING_INFO = (
    "Activity charts pair each activity day with the first night sleep starting "
    "after that day's final recorded activity. Intensity minutes, steps and floors "
    "pair each day with the first night sleep starting after noon that day. "
    "Dotted lines show simple overall trends, not causation."
)


def _plot(figure) -> None:
    st.plotly_chart(
        figure, width="stretch", config={"displaylogo": False, "scrollZoom": False}
    )


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
    st.subheader("Activity and the following night", help=PAIRING_INFO)
    st.plotly_chart(
        activity_intensity_effect_figure(
            view.selected.activities, view.selected.sessions, view.selected.garmin_metrics
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    # Breathing room between the scatter row above and the box plots.
    st.markdown('<div style="height:2.5rem"></div>', unsafe_allow_html=True)
    st.plotly_chart(
        activity_type_comparison_figure(
            view.selected.activities, view.selected.sessions, view.selected.garmin_metrics
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    # Both predictors shown together: duration, then time from activity to sleep.
    for predictor in ("duration", "gap"):
        _plot(
            activity_effect_on_sleep_figure(
                view.selected.activities,
                view.selected.sessions,
                view.selected.garmin_metrics,
                predictor=predictor,
            ),
        )
    _plot(
        intensity_minutes_vs_subsequent_sleep_figure(
            view.selected.garmin_metrics, view.selected.sessions
        ),
    )
    for predictor in ("steps", "floors"):
        _plot(
            steps_floors_vs_subsequent_sleep_figure(
                view.selected.garmin_metrics, view.selected.sessions, predictor
            ),
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
