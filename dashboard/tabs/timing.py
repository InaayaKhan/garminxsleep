"""Timing: bed and wake times, weekday patterns, and timing vs sleep."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    asleep_time_vs_sleep_figure,
    asleep_time_weekend_highlight_figure,
    daily_duration_by_weekday_boxplot,
    hourly_sleep_punch_card,
    nightly_duration_by_weekday_boxplot,
    score_by_weekday_boxplot,
    wake_up_time_weekend_highlight_figure,
)
from dashboard.tabs import ViewData


ASLEEP_AND_WAKE_UP_TIMES_INFO = (
    "Each chart uses one continuous line. Lighter segments mark Saturday and "
    "Sunday; darker segments mark weekdays. Dashed lines are observed means, "
    "shaded regions are ± one standard deviation, and solid straight lines "
    "show overall trends."
)


BY_DAY_OF_WEEK_INFO = (
    "Boxplots show the weekday distributions and individual observations; "
    "yellow diamonds mark weekday means. Daily duration includes naps, while "
    "nightly duration excludes naps."
)


HOUR_BY_HOUR_INFO = (
    "Dot size and color intensity show the number of days asleep during each "
    "hour. Purple excludes naps; blue includes night sleep and naps."
)


TIMING_VS_SLEEP_INFO = (
    "Night sleep only. Zero-sleep placeholders are excluded because they do "
    "not have a genuine asleep time. Each night's asleep time is set against "
    "that night's duration, Sleep score, HRV and stress. ρ is the Spearman "
    "correlation, n the number of nights; dotted lines show simple overall "
    "trends, not causation."
)


def render(view: ViewData) -> None:
    st.markdown(
        'When you fall asleep and wake up, and how timing relates to how long and how well you sleep.'
    )
    st.subheader("Asleep and wake-up times", help=ASLEEP_AND_WAKE_UP_TIMES_INFO)
    asleep_highlight_column, wake_highlight_column = st.columns(2)
    with asleep_highlight_column:
        st.plotly_chart(
            asleep_time_weekend_highlight_figure(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with wake_highlight_column:
        st.plotly_chart(
            wake_up_time_weekend_highlight_figure(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    st.subheader("By day of week", help=BY_DAY_OF_WEEK_INFO)
    st.plotly_chart(
        score_by_weekday_boxplot(view.selected.sessions),
        width="stretch",
        config={"displaylogo": False},
    )
    nightly_boxplot_column, daily_boxplot_column = st.columns(2)
    with nightly_boxplot_column:
        st.plotly_chart(
            nightly_duration_by_weekday_boxplot(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with daily_boxplot_column:
        st.plotly_chart(
            daily_duration_by_weekday_boxplot(view.selected.daily),
            width="stretch",
            config={"displaylogo": False},
        )
    st.subheader("Hour by hour", help=HOUR_BY_HOUR_INFO)
    night_punch_card_column, all_sleep_punch_card_column = st.columns(2)
    with night_punch_card_column:
        st.plotly_chart(
            hourly_sleep_punch_card(
                view.selected.night_intervals,
                title="Hourly sleep punch card (naps excluded)",
                color="#5B32D6",
                light_color="#B8A7F4",
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    with all_sleep_punch_card_column:
        st.plotly_chart(
            hourly_sleep_punch_card(
                view.selected.intervals,
                title="Hourly sleep punch card (night sleep + naps)",
                color="#087CFF",
                light_color="#8FC5FF",
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    st.subheader("Timing vs sleep", help=TIMING_VS_SLEEP_INFO)
    st.plotly_chart(
        asleep_time_vs_sleep_figure(view.selected.sessions, view.selected.garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
