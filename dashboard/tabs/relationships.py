"""Relationships: duration, timing and efficiency vs score; multifactor and cycle."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    TARGET_SLEEP_HOURS,
    asleep_time_vs_duration_figure,
    asleep_time_vs_score_figure,
    menstrual_phase_comparisons_figure,
    multifactor_relationships_figure,
    nightly_duration_quality_figure,
    sleep_efficiency_figure,
    sleep_efficiency_vs_score_figure,
)
from dashboard.tabs import ViewData
from dashboard.ui import format_duration


def render(view: ViewData) -> None:
    st.plotly_chart(
        nightly_duration_quality_figure(view.selected.sessions),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Night sleeps only. The red vertical line marks the "
        f"{format_duration(TARGET_SLEEP_HOURS * 60)} target; the "
        "dashed horizontal line marks mean historical Garmin score, and the shaded "
        "region is ± one standard deviation."
    )
    asleep_duration_column, asleep_score_column = st.columns(2)
    with asleep_duration_column:
        st.plotly_chart(
            asleep_time_vs_duration_figure(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with asleep_score_column:
        st.plotly_chart(
            asleep_time_vs_score_figure(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Night sleep only. Zero-sleep placeholders are excluded because they do "
        "not have a genuine asleep time."
    )
    efficiency_timeline_column, efficiency_score_column = st.columns(2)
    with efficiency_timeline_column:
        st.plotly_chart(
            sleep_efficiency_figure(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with efficiency_score_column:
        st.plotly_chart(
            sleep_efficiency_vs_score_figure(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Sleep efficiency excludes zero-duration placeholders because they have no "
        "sleep window. Garmin Sleep scores use the 0–100 scale."
    )
    st.plotly_chart(
        multifactor_relationships_figure(
            view.selected.daily,
            view.selected.sessions,
            view.selected.activities,
            view.selected.garmin_metrics,
            view.selected.menstrual,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Multifactor coefficients are standardized associations, not causal effects. "
        "Positive bars move with the outcome after accounting for the other displayed predictors."
    )
    st.plotly_chart(
        menstrual_phase_comparisons_figure(
            view.selected.daily,
            view.selected.sessions,
            view.selected.garmin_metrics,
            view.selected.menstrual,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Phase comparisons include only dates where Garmin supplied both the cycle "
        "phase and the corresponding sleep or recovery measurement."
    )
