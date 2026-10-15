"""Overview: daily duration, stages, composition and sleep balance."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    TARGET_SLEEP_HOURS,
    credit_debit_figure,
    daily_duration_figure,
    hour_px,
    sleep_composition_figure,
    sleep_stage_schedule_figure,
)
from dashboard.tabs import ViewData
from dashboard.ui import format_duration, link_pannable_charts


def render(view: ViewData) -> None:
    st.markdown(
        'How much and how well you slept each day. Drag the first three charts sideways to pan back in time; they move together.'
    )
    show_duration_stages = st.toggle(
        "Show sleep stages in duration chart", value=True
    )
    duration_chart = daily_duration_figure(
        view.history.daily,
        view.history.sessions,
        view.history.garmin_metrics,
        view.history.menstrual,
        view.history.sleep_stages,
        view.history.activities,
        show_sleep_stages=show_duration_stages,
        view_start=view.start_date,
        view_end=view.end_date,
    )
    st.plotly_chart(
        duration_chart,
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Daily totals use the wake date. Rolling averages include missing historical "
        "dates and zero-duration placeholders as zero sleep. The 7- and 28-day "
        "averages start hidden; click their legend entries to show them."
        " The colored strip below the x-axis shows Garmin's menstrual phase; hover "
        "over it to see the phase, cycle day, and cycle start date. Drag sideways to "
        "pan back through earlier dates; the 24-hour chart below follows."
    )
    show_sleep_stages = st.toggle(
        "Show sleep stages in 24-hour chart", value=True
    )
    st.plotly_chart(
        sleep_stage_schedule_figure(
            view.history.sessions,
            view.history.intervals,
            (
                view.history.sleep_stages
                if show_sleep_stages
                else view.history.sleep_stages.iloc[0:0]
            ),
            view.history.activities,
            view.history.menstrual,
            view.history.daily,
            view.dataset.coverage_start,
            view.end_date,
            show_sleep_stages=show_sleep_stages,
            garmin_metrics=view.history.garmin_metrics,
            # An hour of sleep is drawn the same height as in the duration chart.
            match_hour_px=hour_px(duration_chart),
            view_start=view.start_date,
            view_end=view.end_date,
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    link_pannable_charts()
    st.caption(
        "Garmin nights show Deep, Light, REM, and Unmeasurable stages; recorded awake "
        "periods are left as unfilled gaps. Completed Garmin workouts and other "
        "recorded activities appear as green time blocks. Sleepmeter history and any "
        "records without stage samples use the striped unavailable pattern; zero-sleep days remain red. The "
        "menstrual phase strip uses the same colors and cycle boundaries as the "
        "duration chart."
    )
    st.plotly_chart(
        sleep_composition_figure(
            view.history.garmin_metrics,
            view_start=view.start_date,
            view_end=view.end_date,
            extent_start=view.dataset.coverage_start,
            extent_end=view.end_date,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Each recorded Garmin night is normalized to 100%, showing the relative "
        "share of Deep, REM, Light, and Awake time."
    )
    st.subheader("Sleep balance")
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
