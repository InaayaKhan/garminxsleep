"""Sleep quality: duration, efficiency and other factors vs score."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    TARGET_SLEEP_HOURS,
    menstrual_phase_comparisons_figure,
    multifactor_relationships_figure,
    nightly_duration_quality_figure,
    sleep_efficiency_figure,
    sleep_efficiency_vs_score_figure,
)
from dashboard.tabs import ViewData
from dashboard.ui import format_duration


def render(view: ViewData) -> None:
    st.markdown(
        'What goes with better or worse nights: duration, efficiency, several factors together, and the menstrual cycle.'
    )
    st.subheader("Duration and score")
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
    st.subheader("Efficiency")
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
    st.subheader("Several factors together")
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
    st.subheader("Menstrual cycle")
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
