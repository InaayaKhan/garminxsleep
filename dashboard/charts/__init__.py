"""Plotly figures for the dashboard, one module per tab.

Everything public is re-exported here, so callers use ``from dashboard.charts import ...``.
"""

from dashboard.charts.common import (
    TARGET_SLEEP_HOURS,
)
from dashboard.charts.overview import (
    credit_debit_figure,
    daily_duration_figure,
    hour_px,
    sleep_composition_figure,
    sleep_stage_schedule_figure,
)
from dashboard.charts.patterns import (
    calendar_sleep_heatmap,
    calendar_sleep_score_heatmap,
    daily_duration_histogram,
    nightly_duration_histogram,
    sleep_nap_pie_figure,
    sleep_state_pie_figure,
)
from dashboard.charts.timing import (
    asleep_time_vs_duration_figure,
    asleep_time_vs_score_figure,
    asleep_time_weekend_highlight_figure,
    daily_duration_by_weekday_boxplot,
    hourly_sleep_punch_card,
    nightly_duration_by_weekday_boxplot,
    score_by_weekday_boxplot,
    wake_up_time_weekend_highlight_figure,
)
from dashboard.charts.aids import (
    aid_duration_timeline_figure,
    aid_figure,
    aid_score_figure,
    aid_sleep_stage_composition_figure,
)
from dashboard.charts.activity import (
    activity_effect_on_sleep_figure,
    activity_intensity_effect_figure,
    activity_to_sleep_gap_figure,
    activity_type_comparison_figure,
    calendar_activity_heatmap,
    calendar_intensity_minutes_heatmap,
    intensity_minutes_vs_subsequent_sleep_figure,
    steps_floors_vs_subsequent_sleep_figure,
)
from dashboard.charts.sleep_quality import (
    menstrual_phase_comparisons_figure,
    multifactor_relationships_figure,
    nightly_duration_quality_figure,
    sleep_efficiency_figure,
    sleep_efficiency_vs_score_figure,
)
from dashboard.charts.garmin import (
    awakenings_efficiency_figure,
    daily_calories_stacked_figure,
    daily_physiological_load_figure,
    heart_rate_24_hour_heatmap,
    intraday_recovery_timeline_figure,
    overnight_recovery_figure,
    sleep_score_components_figure,
)

__all__ = [
    "TARGET_SLEEP_HOURS",
    "activity_effect_on_sleep_figure",
    "activity_intensity_effect_figure",
    "activity_to_sleep_gap_figure",
    "activity_type_comparison_figure",
    "aid_duration_timeline_figure",
    "aid_figure",
    "aid_score_figure",
    "aid_sleep_stage_composition_figure",
    "asleep_time_vs_duration_figure",
    "asleep_time_vs_score_figure",
    "asleep_time_weekend_highlight_figure",
    "awakenings_efficiency_figure",
    "calendar_activity_heatmap",
    "calendar_intensity_minutes_heatmap",
    "calendar_sleep_heatmap",
    "calendar_sleep_score_heatmap",
    "credit_debit_figure",
    "daily_calories_stacked_figure",
    "daily_duration_by_weekday_boxplot",
    "daily_duration_figure",
    "daily_duration_histogram",
    "daily_physiological_load_figure",
    "heart_rate_24_hour_heatmap",
    "hour_px",
    "hourly_sleep_punch_card",
    "intensity_minutes_vs_subsequent_sleep_figure",
    "intraday_recovery_timeline_figure",
    "menstrual_phase_comparisons_figure",
    "multifactor_relationships_figure",
    "nightly_duration_by_weekday_boxplot",
    "nightly_duration_histogram",
    "nightly_duration_quality_figure",
    "overnight_recovery_figure",
    "score_by_weekday_boxplot",
    "sleep_composition_figure",
    "sleep_efficiency_figure",
    "sleep_efficiency_vs_score_figure",
    "sleep_nap_pie_figure",
    "sleep_score_components_figure",
    "sleep_stage_schedule_figure",
    "sleep_state_pie_figure",
    "steps_floors_vs_subsequent_sleep_figure",
    "wake_up_time_weekend_highlight_figure",
]
