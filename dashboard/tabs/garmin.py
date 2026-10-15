"""Garmin: measurements that come only from the watch."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    awakenings_efficiency_figure,
    daily_calories_stacked_figure,
    daily_physiological_load_figure,
    heart_rate_24_hour_heatmap,
    intraday_recovery_timeline_figure,
    overnight_recovery_figure,
    sleep_score_components_figure,
)
from dashboard.tabs import ViewData


def render(view: ViewData) -> None:
    st.markdown(
        'Measurements recorded only by the Garmin watch.'
    )
    st.subheader("Sleep score")
    st.plotly_chart(
        sleep_score_components_figure(view.selected.garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Component colors are Garmin's Poor, Fair, Good, and Excellent qualifiers. "
        "Hover for the underlying percentage or measurement where Garmin provides it."
    )
    st.subheader("Overnight recovery")
    st.plotly_chart(
        overnight_recovery_figure(view.selected.garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Aligned overnight recovery measures from Garmin. Missing values mean the "
        "watch or Garmin response did not provide that measurement."
    )
    st.subheader("Daily load")
    load_column, calories_column = st.columns(2)
    with load_column:
        st.plotly_chart(
            daily_physiological_load_figure(view.selected.garmin_metrics),
            width="stretch",
            config={"displaylogo": False},
        )
    with calories_column:
        st.plotly_chart(
            daily_calories_stacked_figure(view.selected.garmin_metrics),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )
    st.subheader("Awakenings")
    st.plotly_chart(
        awakenings_efficiency_figure(view.selected.garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Efficiency is asleep stage time divided by asleep plus awake stage time. "
        "Garmin did not provide numeric restless-moment counts, so its qualitative "
        "Restlessness rating is shown instead."
    )
    st.subheader("Through the day")
    intraday_dates = sorted(view.selected.intraday["date"].dropna().unique(), reverse=True)
    if intraday_dates:
        selected_intraday_date = st.selectbox(
            "Intraday recovery date", intraday_dates, key="intraday_recovery_date"
        )
        st.plotly_chart(
            intraday_recovery_timeline_figure(
                view.selected.intraday,
                view.selected.activities,
                view.selected.sessions,
                selected_intraday_date,
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    else:
        st.info("No intraday recovery samples in the selected date range.")
    st.plotly_chart(
        heart_rate_24_hour_heatmap(
            view.selected.intraday, view.selected.activities, view.start_date, view.end_date
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
