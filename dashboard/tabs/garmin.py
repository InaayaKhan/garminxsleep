"""Garmin: measurements that come only from the watch."""

from __future__ import annotations

from datetime import date

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


INTRADAY_DATE_KEY = "intraday_recovery_date"


def _step_intraday_date(dates: list[date], step: int) -> None:
    """Move to the previous or next day that has intraday samples."""
    current = st.session_state[INTRADAY_DATE_KEY]
    if step < 0:
        earlier = [day for day in dates if day < current]
        st.session_state[INTRADAY_DATE_KEY] = earlier[-1] if earlier else dates[0]
    else:
        later = [day for day in dates if day > current]
        st.session_state[INTRADAY_DATE_KEY] = later[0] if later else dates[-1]


def _intraday_date_picker(dates: list[date]) -> date:
    """Date picker with previous/next buttons; opens on the latest day with data."""
    current = st.session_state.get(INTRADAY_DATE_KEY)
    if not isinstance(current, date) or not dates[0] <= current <= dates[-1]:
        st.session_state[INTRADAY_DATE_KEY] = dates[-1]
    current = st.session_state[INTRADAY_DATE_KEY]
    previous_column, picker_column, next_column, _ = st.columns(
        [0.6, 2.4, 0.6, 8], vertical_alignment="bottom"
    )
    previous_column.button(
        "◀",
        key="intraday_previous",
        help="Previous day with data",
        disabled=current <= dates[0],
        on_click=_step_intraday_date,
        args=(dates, -1),
        width="stretch",
    )
    picker_column.date_input(
        "Intraday recovery date",
        key=INTRADAY_DATE_KEY,
        min_value=dates[0],
        max_value=dates[-1],
        format="DD.MM.YYYY",
    )
    next_column.button(
        "▶",
        key="intraday_next",
        help="Next day with data",
        disabled=current >= dates[-1],
        on_click=_step_intraday_date,
        args=(dates, 1),
        width="stretch",
    )
    return st.session_state[INTRADAY_DATE_KEY]


SLEEP_SCORE_INFO = (
    "Component colors are Garmin's Poor, Fair, Good, and Excellent qualifiers. "
    "Hover for the underlying percentage or measurement where Garmin provides it."
)


OVERNIGHT_RECOVERY_INFO = (
    "Aligned overnight recovery measures from Garmin. Missing values mean the "
    "watch or Garmin response did not provide that measurement."
)


AWAKENINGS_INFO = (
    "Efficiency is asleep stage time divided by asleep plus awake stage time. "
    "Garmin did not provide numeric restless-moment counts, so its qualitative "
    "Restlessness rating is shown instead."
)


def render(view: ViewData) -> None:
    st.markdown(
        'Measurements recorded only by the Garmin watch.'
    )
    st.subheader("Sleep score", help=SLEEP_SCORE_INFO)
    st.plotly_chart(
        sleep_score_components_figure(view.selected.garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.subheader("Overnight recovery", help=OVERNIGHT_RECOVERY_INFO)
    st.plotly_chart(
        overnight_recovery_figure(view.selected.garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
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
    st.subheader("Awakenings", help=AWAKENINGS_INFO)
    st.plotly_chart(
        awakenings_efficiency_figure(view.selected.garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.subheader("Through the day")
    intraday_dates = sorted(view.selected.intraday["date"].dropna().unique())
    if intraday_dates:
        selected_intraday_date = _intraday_date_picker(intraday_dates)
    if intraday_dates and selected_intraday_date not in intraday_dates:
        st.info(f"No intraday recovery samples on {selected_intraday_date:%d.%m.%Y}.")
    elif intraday_dates:
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
