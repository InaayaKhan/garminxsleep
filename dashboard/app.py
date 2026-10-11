from __future__ import annotations

import os
import json
import math
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from dashboard.charts import (
    TARGET_SLEEP_HOURS,
    hour_px,
    aid_figure,
    aid_duration_timeline_figure,
    aid_sleep_stage_composition_figure,
    aid_score_figure,
    activity_effect_on_sleep_figure,
    activity_intensity_effect_figure,
    activity_to_sleep_gap_figure,
    activity_type_comparison_figure,
    calendar_activity_heatmap,
    calendar_intensity_minutes_heatmap,
    asleep_time_weekend_highlight_figure,
    asleep_time_vs_score_figure,
    asleep_time_vs_duration_figure,
    calendar_sleep_heatmap,
    calendar_sleep_score_heatmap,
    credit_debit_figure,
    daily_duration_by_weekday_boxplot,
    daily_duration_histogram,
    daily_duration_figure,
    daily_calories_stacked_figure,
    daily_physiological_load_figure,
    hourly_sleep_punch_card,
    nightly_duration_histogram,
    nightly_duration_by_weekday_boxplot,
    nightly_duration_quality_figure,
    awakenings_efficiency_figure,
    score_by_weekday_boxplot,
    sleep_efficiency_figure,
    sleep_efficiency_vs_score_figure,
    sleep_score_components_figure,
    sleep_nap_pie_figure,
    sleep_state_pie_figure,
    sleep_stage_schedule_figure,
    overnight_recovery_figure,
    menstrual_phase_comparisons_figure,
    multifactor_relationships_figure,
    intraday_recovery_timeline_figure,
    intensity_minutes_vs_subsequent_sleep_figure,
    steps_floors_vs_subsequent_sleep_figure,
    heart_rate_24_hour_heatmap,
    sleep_regularity_figure,
    sleep_composition_figure,
    wake_up_time_weekend_highlight_figure,
)
from dashboard.config import (
    DATA_ROOT,
    DATABASE_PATH,
    RAW_ROOT,
    TIMEZONE_NAME,
    garmin_start_date,
)
from dashboard.model import SleepDataset, load_combined_sleep_data
from dashboard.storage import persist_dataset


GARMIN_RAW_ROOTS = [RAW_ROOT]
GARMIN_AID_OVERRIDES_PATH = DATA_ROOT / "garmin_aid_overrides.json"
GARMIN_SCORE_OVERRIDES_PATH = DATA_ROOT / "garmin_score_overrides.json"
GARMIN_MERGE_OVERRIDES_PATH = DATA_ROOT / "garmin_merge_overrides.json"


def find_export() -> Path:
    configured = os.getenv("SLEEPMETER_EXPORT")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configured export does not exist: {path}")
        return path
    exports = sorted(
        DATA_ROOT.glob("sleep_history_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not exports:
        raise FileNotFoundError(f"No sleep_history_*.csv export found in {DATA_ROOT}")
    return exports[0]


def garmin_archive_signature() -> tuple[tuple[str, int, int], ...]:
    paths = [
        path
        for root in GARMIN_RAW_ROOTS
        if root.exists()
        for path in root.glob("*/*.json")
    ]
    if GARMIN_AID_OVERRIDES_PATH.exists():
        paths.append(GARMIN_AID_OVERRIDES_PATH)
    if GARMIN_SCORE_OVERRIDES_PATH.exists():
        paths.append(GARMIN_SCORE_OVERRIDES_PATH)
    if GARMIN_MERGE_OVERRIDES_PATH.exists():
        paths.append(GARMIN_MERGE_OVERRIDES_PATH)
    return tuple(
        sorted((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    )


def save_garmin_aid_override(sleep_date: date, aid_name: str | None) -> None:
    overrides: dict[str, str] = {}
    if GARMIN_AID_OVERRIDES_PATH.exists():
        overrides = json.loads(GARMIN_AID_OVERRIDES_PATH.read_text(encoding="utf-8"))
    key = sleep_date.isoformat()
    overrides[key] = aid_name or "None"
    GARMIN_AID_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = GARMIN_AID_OVERRIDES_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(overrides, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(GARMIN_AID_OVERRIDES_PATH)


@st.cache_data(show_spinner=False)
def cached_dataset(
    path_text: str,
    modified_ns: int,
    raw_signature: tuple[tuple[str, int, int], ...],
) -> SleepDataset:
    del modified_ns, raw_signature
    return load_combined_sleep_data(
        path_text,
        GARMIN_RAW_ROOTS,
        garmin_start_date=garmin_start_date(),
        aid_overrides_path=GARMIN_AID_OVERRIDES_PATH,
        score_overrides_path=GARMIN_SCORE_OVERRIDES_PATH,
        merge_overrides_path=GARMIN_MERGE_OVERRIDES_PATH,
        timezone_name=TIMEZONE_NAME,
    )


def format_duration(minutes: float) -> str:
    if not math.isfinite(minutes):
        return "No data"
    hours = int(minutes // 60)
    remainder = int(round(minutes % 60))
    return f"{hours}h {remainder:02d}m"


st.set_page_config(page_title="garminxsleep", page_icon="🌙", layout="wide")

_streamlit_plotly_chart = st.plotly_chart
LEGEND_ENTRIES_PER_ROW = 3
LEGEND_ROW_HEIGHT = 24
DEFAULT_RANGE_DAYS = 180


def _toolbar_only_plotly_chart(figure, *args, config=None, **kwargs):
    """Disable accidental chart gestures while retaining Plotly's toolbar tools.

    Legends sit along the bottom edge of every chart so wrapped rows grow into
    the bottom margin instead of colliding with the title. The chart grows by
    roughly the legend's height so the plot itself is not squeezed, unless the
    figure opts out with ``meta=dict(fixed_height=True)``.
    """
    fixed_height = (figure.layout.meta or {}).get("fixed_height", False)
    if (
        figure.layout.height
        and figure.layout.showlegend is not False
        and not fixed_height
    ):
        entries = {
            trace.legendgroup or trace.name
            for trace in figure.data
            if trace.showlegend is not False and trace.name
        }
        legend_rows = math.ceil(len(entries) / LEGEND_ENTRIES_PER_ROW)
        figure.update_layout(
            height=figure.layout.height + legend_rows * LEGEND_ROW_HEIGHT
        )
    figure.update_layout(
        # Charts that hold more history than they show opt into panning.
        dragmode="pan" if (figure.layout.meta or {}).get("pan") else False,
        legend=dict(
            orientation="h",
            xref="paper",
            x=0,
            xanchor="left",
            yref="container",
            y=0,
            yanchor="bottom",
            entrywidth=180,
        ),
    )
    if (figure.layout.meta or {}).get("legend_position") == "right":
        # A vertical legend in the figure's reserved right margin never
        # changes the plot size, so charts sharing a scale stay in step.
        figure.update_layout(
            legend=dict(
                orientation="v",
                xref="container",
                x=1,
                xanchor="right",
                yref="paper",
                y=1,
                yanchor="top",
            )
        )
    chart_config = dict(config or {})
    chart_config["scrollZoom"] = False
    return _streamlit_plotly_chart(
        figure,
        *args,
        config=chart_config,
        **kwargs,
    )


st.plotly_chart = _toolbar_only_plotly_chart

# Plotly charts render in the main page, so a script in a same-origin
# component can reach them: panning one pannable chart pans the others too.
_LINK_PANNING_SCRIPT = """
<script>
const page = window.parent;
function linkPannableCharts() {
  const charts = [...page.document.querySelectorAll(".js-plotly-plot")]
    .filter((chart) => chart.layout && chart.layout.meta && chart.layout.meta.pan);
  for (const chart of charts) {
    if (chart._panLinked) continue;
    chart._panLinked = true;
    chart.on("plotly_relayout", (change) => {
      if (page._panSyncing) return;
      const start = change["xaxis.range[0]"] ?? (change["xaxis.range"] || [])[0];
      const end = change["xaxis.range[1]"] ?? (change["xaxis.range"] || [])[1];
      if (start === undefined || end === undefined) return;
      page._panSyncing = true;
      Promise.all(
        charts
          .filter((other) => other !== chart && other.isConnected)
          .map((other) => page.Plotly.relayout(other, {"xaxis.range": [start, end]}))
      ).finally(() => { page._panSyncing = false; });
    });
  }
}
// Streamlit re-creates charts on every rerun; re-link whatever is new.
linkPannableCharts();
new MutationObserver(linkPannableCharts).observe(page.document.body, {childList: true, subtree: true});
</script>
"""


def link_pannable_charts() -> None:
    components.html(_LINK_PANNING_SCRIPT, height=0)
st.title("garminxsleep")
st.sidebar.caption("Local Garmin and Sleepmeter dashboard")

try:
    export_path = find_export()
    dataset = cached_dataset(
        str(export_path),
        export_path.stat().st_mtime_ns,
        garmin_archive_signature(),
    )
    persist_dataset(dataset, DATABASE_PATH)
except Exception as error:
    st.error(f"Unable to load the Sleepmeter export: {error}")
    st.stop()

st.sidebar.header("Date range")
# Open on the most recent DEFAULT_RANGE_DAYS; earlier dates stay selectable.
default_start = max(
    dataset.coverage_start,
    dataset.coverage_end - timedelta(days=DEFAULT_RANGE_DAYS - 1),
)
selected_range = st.sidebar.date_input(
    "Sleep dates",
    value=(default_start, dataset.coverage_end),
    min_value=dataset.coverage_start,
    max_value=dataset.coverage_end,
)
if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date = end_date = selected_range if isinstance(selected_range, date) else dataset.coverage_end

filtered_daily = dataset.daily[
    (dataset.daily["date"] >= start_date) & (dataset.daily["date"] <= end_date)
].copy()
filtered_sessions = dataset.sessions[
    (dataset.sessions["sleep_date"] >= start_date)
    & (dataset.sessions["sleep_date"] <= end_date)
].copy()
filtered_intervals = dataset.intervals[
    dataset.intervals["session_id"].isin(filtered_sessions["session_id"])
].copy()
filtered_sleep_stages = dataset.sleep_stages[
    dataset.sleep_stages["session_id"].isin(filtered_sessions["session_id"])
].copy()
filtered_activities = dataset.activities[
    (dataset.activities["activity_date"] >= start_date)
    & (dataset.activities["activity_date"] <= end_date)
].copy()
filtered_menstrual = dataset.menstrual[
    (dataset.menstrual["date"] >= start_date)
    & (dataset.menstrual["date"] <= end_date)
].copy()
filtered_night_intervals = filtered_intervals[
    filtered_intervals["session_id"].isin(
        filtered_sessions.loc[
            filtered_sessions["session_type"] == "NIGHT_SLEEP", "session_id"
        ]
    )
].copy()
filtered_garmin_metrics = dataset.garmin_metrics[
    (dataset.garmin_metrics["date"] >= start_date)
    & (dataset.garmin_metrics["date"] <= end_date)
].copy()
filtered_intraday = dataset.intraday[
    (dataset.intraday["date"] >= start_date)
    & (dataset.intraday["date"] <= end_date)
].copy()

# The Overview date charts get all history up to the end date and open on the
# selected range; earlier days are reached by panning.
history_daily = dataset.daily[dataset.daily["date"] <= end_date].copy()
history_sessions = dataset.sessions[dataset.sessions["sleep_date"] <= end_date].copy()
history_intervals = dataset.intervals[
    dataset.intervals["session_id"].isin(history_sessions["session_id"])
].copy()
history_sleep_stages = dataset.sleep_stages[
    dataset.sleep_stages["session_id"].isin(history_sessions["session_id"])
].copy()
history_activities = dataset.activities[
    dataset.activities["activity_date"] <= end_date
].copy()
history_menstrual = dataset.menstrual[dataset.menstrual["date"] <= end_date].copy()
history_garmin_metrics = dataset.garmin_metrics[
    dataset.garmin_metrics["date"] <= end_date
].copy()

days = len(filtered_daily)
average_minutes = filtered_daily["total_sleep_minutes"].mean() if days else 0
zero_days = int(filtered_daily["zero_sleep"].sum()) if days else 0
session_count = len(filtered_sessions)

st.sidebar.divider()
st.sidebar.subheader("Summary")
st.sidebar.metric("Calendar days", days)
st.sidebar.metric("Average daily sleep", format_duration(float(average_minutes)))
st.sidebar.metric("Zero-sleep days", zero_days)
st.sidebar.metric("Recorded sessions", session_count)

overview_tab, timing_tab, relationships_tab, experimental_tab, experimental_2_tab, experimental_3_tab, aids_tab, distributions_tab, records_tab = st.tabs(
    [
        "Overview",
        "Timing",
        "Relationships",
        "Experimental",
        "Experimental 2",
        "Experimental 3",
        "Aids",
        "Trends & distributions",
        "Records",
    ]
)

with overview_tab:
    show_duration_stages = st.toggle(
        "Show sleep stages in duration chart", value=True
    )
    duration_chart = daily_duration_figure(
        history_daily,
        history_sessions,
        history_garmin_metrics,
        history_menstrual,
        history_sleep_stages,
        history_activities,
        show_sleep_stages=show_duration_stages,
        view_start=start_date,
        view_end=end_date,
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
            history_sessions,
            history_intervals,
            (
                history_sleep_stages
                if show_sleep_stages
                else history_sleep_stages.iloc[0:0]
            ),
            history_activities,
            history_menstrual,
            history_daily,
            dataset.coverage_start,
            end_date,
            show_sleep_stages=show_sleep_stages,
            garmin_metrics=history_garmin_metrics,
            # An hour of sleep is drawn the same height as in the duration chart.
            match_hour_px=hour_px(duration_chart),
            view_start=start_date,
            view_end=end_date,
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
    activity_heatmap_column, intensity_heatmap_column = st.columns(2)
    with activity_heatmap_column:
        st.plotly_chart(
            calendar_activity_heatmap(filtered_daily, filtered_activities),
            width="stretch",
            config={"displaylogo": False},
        )
    with intensity_heatmap_column:
        st.plotly_chart(
            calendar_intensity_minutes_heatmap(
                filtered_daily, filtered_garmin_metrics
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    duration_heatmap_column, score_heatmap_column = st.columns(2)
    with duration_heatmap_column:
        st.plotly_chart(
            calendar_sleep_heatmap(filtered_daily),
            width="stretch",
            config={"displaylogo": False},
        )
    with score_heatmap_column:
        st.plotly_chart(
            calendar_sleep_score_heatmap(filtered_daily, filtered_sessions),
            width="stretch",
            config={"displaylogo": False},
        )

with timing_tab:
    asleep_highlight_column, wake_highlight_column = st.columns(2)
    with asleep_highlight_column:
        st.plotly_chart(
            asleep_time_weekend_highlight_figure(filtered_sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with wake_highlight_column:
        st.plotly_chart(
            wake_up_time_weekend_highlight_figure(filtered_sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Each chart uses one continuous line. Lighter segments mark Saturday and "
        "Sunday; darker segments mark weekdays. Dashed lines are observed means, "
        "shaded regions are ± one standard deviation, and solid straight lines "
        "show overall trends."
    )
    st.plotly_chart(
        score_by_weekday_boxplot(filtered_sessions),
        width="stretch",
        config={"displaylogo": False},
    )
    nightly_boxplot_column, daily_boxplot_column = st.columns(2)
    with nightly_boxplot_column:
        st.plotly_chart(
            nightly_duration_by_weekday_boxplot(filtered_sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with daily_boxplot_column:
        st.plotly_chart(
            daily_duration_by_weekday_boxplot(filtered_daily),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Boxplots show the weekday distributions and individual observations; "
        "yellow diamonds mark weekday means. Daily duration includes naps, while "
        "nightly duration excludes naps."
    )
    night_punch_card_column, all_sleep_punch_card_column = st.columns(2)
    with night_punch_card_column:
        st.plotly_chart(
            hourly_sleep_punch_card(
                filtered_night_intervals,
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
                filtered_intervals,
                title="Hourly sleep punch card (night sleep + naps)",
                color="#087CFF",
                light_color="#8FC5FF",
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Dot size and color intensity show the number of days asleep during each "
        "hour. Purple excludes naps; blue includes night sleep and naps."
    )

with relationships_tab:
    st.plotly_chart(
        nightly_duration_quality_figure(filtered_sessions),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Night sleeps only. The red vertical line marks the 7h 30m target; the "
        "dashed horizontal line marks mean historical Garmin score, and the shaded "
        "region is ± one standard deviation."
    )
    asleep_duration_column, asleep_score_column = st.columns(2)
    with asleep_duration_column:
        st.plotly_chart(
            asleep_time_vs_duration_figure(filtered_sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with asleep_score_column:
        st.plotly_chart(
            asleep_time_vs_score_figure(filtered_sessions),
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
            sleep_efficiency_figure(filtered_sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with efficiency_score_column:
        st.plotly_chart(
            sleep_efficiency_vs_score_figure(filtered_sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Sleep efficiency excludes zero-duration placeholders because they have no "
        "sleep window. Garmin Sleep scores use the 0–100 scale."
    )

with experimental_tab:
    st.plotly_chart(
        sleep_score_components_figure(filtered_garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Component colors are Garmin's Poor, Fair, Good, and Excellent qualifiers. "
        "Hover for the underlying percentage or measurement where Garmin provides it."
    )
    st.plotly_chart(
        sleep_composition_figure(filtered_garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Each recorded Garmin night is normalized to 100%, showing the relative "
        "share of Deep, REM, Light, and Awake time."
    )
    st.plotly_chart(
        overnight_recovery_figure(filtered_garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Aligned overnight recovery measures from Garmin. Missing values mean the "
        "watch or Garmin response did not provide that measurement."
    )
    st.plotly_chart(
        daily_physiological_load_figure(filtered_garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        menstrual_phase_comparisons_figure(
            filtered_daily,
            filtered_sessions,
            filtered_garmin_metrics,
            filtered_menstrual,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Phase comparisons include only dates where Garmin supplied both the cycle "
        "phase and the corresponding sleep or recovery measurement."
    )
    st.plotly_chart(
        awakenings_efficiency_figure(filtered_garmin_metrics),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Efficiency is asleep stage time divided by asleep plus awake stage time. "
        "Garmin did not provide numeric restless-moment counts, so its qualitative "
        "Restlessness rating is shown instead."
    )
    activity_predictor = st.radio(
        "Activity predictor",
        ["Activity duration", "Time from activity to sleep"],
        horizontal=True,
    )
    st.plotly_chart(
        activity_effect_on_sleep_figure(
            filtered_activities,
            filtered_sessions,
            filtered_garmin_metrics,
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

with experimental_2_tab:
    st.plotly_chart(
        activity_type_comparison_figure(
            filtered_activities, filtered_sessions, filtered_garmin_metrics
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        activity_intensity_effect_figure(
            filtered_activities, filtered_sessions, filtered_garmin_metrics
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        sleep_regularity_figure(filtered_sessions),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "This is a custom 0–100 timing-consistency score based on changes in "
        "sleep onset and wake time between adjacent nights; it is not the formal "
        "research Sleep Regularity Index."
    )
    intraday_dates = sorted(filtered_intraday["date"].dropna().unique(), reverse=True)
    if intraday_dates:
        selected_intraday_date = st.selectbox(
            "Intraday recovery date", intraday_dates, key="intraday_recovery_date"
        )
        st.plotly_chart(
            intraday_recovery_timeline_figure(
                filtered_intraday,
                filtered_activities,
                filtered_sessions,
                selected_intraday_date,
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    else:
        st.info("No intraday recovery samples in the selected date range.")
    st.plotly_chart(
        multifactor_relationships_figure(
            filtered_daily,
            filtered_sessions,
            filtered_activities,
            filtered_garmin_metrics,
            filtered_menstrual,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Multifactor coefficients are standardized associations, not causal effects. "
        "Positive bars move with the outcome after accounting for the other displayed predictors."
    )

with experimental_3_tab:
    st.plotly_chart(
        heart_rate_24_hour_heatmap(
            filtered_intraday, filtered_activities, start_date, end_date
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    st.plotly_chart(
        daily_calories_stacked_figure(filtered_garmin_metrics),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    st.plotly_chart(
        intensity_minutes_vs_subsequent_sleep_figure(
            filtered_garmin_metrics, filtered_sessions
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    st.plotly_chart(
        steps_floors_vs_subsequent_sleep_figure(
            filtered_garmin_metrics, filtered_sessions
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    st.plotly_chart(
        activity_to_sleep_gap_figure(filtered_activities, filtered_sessions),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    st.plotly_chart(
        activity_to_sleep_gap_figure(
            filtered_activities,
            filtered_sessions,
            exclude_daytime_activities=True,
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )

with aids_tab:
    garmin_aid_sessions = filtered_sessions[
        (filtered_sessions["source"] == "garmin")
        & (filtered_sessions["session_type"] == "NIGHT_SLEEP")
    ].sort_values("sleep_date", ascending=False)
    with st.expander("Set aid for Garmin sleep"):
        if garmin_aid_sessions.empty:
            st.info("No Garmin night-sleep records are available in this date range.")
        else:
            session_dates = garmin_aid_sessions["sleep_date"].tolist()
            selected_aid_date = st.selectbox(
                "Garmin sleep date",
                session_dates,
                format_func=lambda value: value.isoformat(),
            )
            selected_aid_session = garmin_aid_sessions[
                garmin_aid_sessions["sleep_date"] == selected_aid_date
            ].iloc[0]
            current_aid = str(selected_aid_session["aid_name"])
            known_aids = sorted(
                {
                    "None",
                    *dataset.aids["aid_name"].astype(str).tolist(),
                    *dataset.sessions["aid_name"].dropna().astype(str).tolist(),
                },
                key=lambda value: (value != "None", value.casefold()),
            )
            aid_choices = known_aids + ["Custom…"]
            selected_aid = st.selectbox(
                "Recorded sleep aid",
                aid_choices,
                index=(
                    aid_choices.index(current_aid)
                    if current_aid in aid_choices
                    else aid_choices.index("Custom…")
                ),
                key=f"garmin_aid_choice_{selected_aid_date.isoformat()}",
            )
            custom_aid = ""
            if selected_aid == "Custom…":
                custom_aid = st.text_input(
                    "Custom aid name",
                    value=current_aid if current_aid not in known_aids else "",
                    key=f"garmin_custom_aid_{selected_aid_date.isoformat()}",
                ).strip()
            save_column, clear_column = st.columns(2)
            with save_column:
                if st.button("Save aid", type="primary", width="stretch"):
                    aid_to_save = custom_aid if selected_aid == "Custom…" else selected_aid
                    if not aid_to_save:
                        st.error("Enter a custom aid name before saving.")
                    else:
                        save_garmin_aid_override(selected_aid_date, aid_to_save)
                        cached_dataset.clear()
                        st.rerun()
            with clear_column:
                if st.button("Clear aid", width="stretch"):
                    save_garmin_aid_override(selected_aid_date, None)
                    cached_dataset.clear()
                    st.rerun()
            st.caption(
                "The selected aid applies from this date forward until another aid "
                "change is saved. Clear aid applies None from this date forward. "
                "Nothing is written back to Garmin Connect."
            )
    st.plotly_chart(
        aid_score_figure(filtered_sessions),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        aid_figure(filtered_sessions),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        aid_sleep_stage_composition_figure(
            filtered_sessions, filtered_sleep_stages
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        aid_sleep_stage_composition_figure(
            filtered_sessions,
            filtered_sleep_stages,
            use_stage_duration=True,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        aid_duration_timeline_figure(
            history_daily,
            history_sessions,
            view_start=start_date,
            view_end=end_date,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Garmin Sleep scores are shown for night sleep only. Point colors indicate "
        "the recorded sleep aid. Aid comparisons are descriptive, not causal."
    )

with distributions_tab:
    nightly_histogram_column, daily_histogram_column = st.columns(2)
    with nightly_histogram_column:
        st.plotly_chart(
            nightly_duration_histogram(filtered_sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with daily_histogram_column:
        st.plotly_chart(
            daily_duration_histogram(filtered_daily),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Histograms use one-hour bins. Nightly duration excludes naps; daily "
        "duration includes naps. Zero-sleep observations remain included."
    )
    sleep_nap_column, time_breakdown_column = st.columns(2)
    with sleep_nap_column:
        st.plotly_chart(
            sleep_nap_pie_figure(filtered_sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with time_breakdown_column:
        st.plotly_chart(
            sleep_state_pie_figure(filtered_sessions, filtered_daily),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Sleep and naps use recorded asleep time. Awake in holes uses Sleepmeter "
        "interruptions; Awake is the remaining time across the selected days. The "
        "first pie compares night sleep with naps only."
    )
    credit_target_minutes = st.select_slider(
        "Daily sleep target",
        options=range(5 * 60, 10 * 60 + 1, 15),
        value=int(TARGET_SLEEP_HOURS * 60),
        format_func=format_duration,
        key="credit_debit_target_minutes",
    )
    st.plotly_chart(
        credit_debit_figure(filtered_daily, credit_target_minutes / 60),
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

with records_tab:
    display = filtered_sessions[
        [
            "sleep_date",
            "sleep_start",
            "wake_time",
            "session_type",
            "sleep_minutes",
            "interruption_minutes",
            "aid_name",
            "legacy_garmin_score_10",
            "zero_duration",
            "holes",
            "dreams",
            "hindrances",
            "notes",
        ]
    ].copy()
    display["sleep_hours"] = (display.pop("sleep_minutes") / 60).round(2)
    st.dataframe(display, width="stretch", hide_index=True)

st.sidebar.divider()
st.sidebar.caption(f"Source: {export_path.name}")
st.sidebar.caption(
    f"Coverage: {dataset.coverage_start.isoformat()} to {dataset.coverage_end.isoformat()}"
)
st.sidebar.caption(f"Local database: {DATABASE_PATH.name}")
