from __future__ import annotations

import json
import math
import zlib
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.config import DATA_ROOT, TIMEZONE_NAME

NIGHT_BAR_COLOR = "#5B32D6"
NAP_BAR_COLOR = "#087CFF"
ZERO_COLOR = "#ff3030"
NO_DATA_COLOR = "#6B7280"
AVERAGE_3_COLOR = "#E8793F"
AVERAGE_7_COLOR = "#E6B84F"
AVERAGE_28_COLOR = "#4DB88A"
SLEEP_SCORE_PINK = "#FF7AB6"
HRV_GREEN = "#4DB88A"
HRV_WEEKLY_GREEN = "#8ED8B8"
STATS_GREEN = "#00ee28"
STATS_RED = "#ff2020"
CREDIT_COLOR = "#E6B84F"
ASLEEP_TIME_COLOR = "#5B32D6"
WAKE_TIME_COLOR = "#E8793F"
TARGET_SLEEP_HOURS = 7.5
TARGET_ASLEEP_HOUR = 25.0  # 01:00 on a noon-to-noon clock
TARGET_WAKE_HOUR = 8.5
MAXIMUM_MARKER_COLOR = "#22c55e"
AID_COLORS_PATH = DATA_ROOT / "aid_colors.json"
# No aid uses the regular night/nap colours, so unaided nights look the same
# here as in the daily duration chart.
NO_AID_COLORS = {"color": NIGHT_BAR_COLOR, "nap": NAP_BAR_COLOR}
# Aids missing from aid_colors.json get a stable colour from this palette.
FALLBACK_AID_PALETTE = [
    {"color": "#ef4444", "nap": "#f87171"},
    {"color": "#facc15", "nap": "#fde047"},
    {"color": "#22c55e", "nap": "#4ade80"},
    {"color": "#087CFF", "nap": "#60a5fa"},
    {"color": "#a855f7", "nap": "#c084fc"},
    {"color": "#f97316", "nap": "#fb923c"},
]
CLOCK_BOUNDARY_MINUTE = 0
MENSTRUAL_PHASE_COLORS = {
    1: "#E14E5C",
    2: "#2FD6A3",
    3: "#FFB547",
    4: "#F29BDA",
}
MENSTRUAL_STRIP_PX = 7
# The daily duration and 24-hour stage charts share a day width and an hour
# scale. Their legends sit in a fixed-width column on the right, so legend
# wrapping never changes the plot size; the right margin holds the secondary
# y-axes (160px) plus that legend column.
RIGHT_LEGEND_PX = 180
DATE_CHART_SIDE_MARGINS = dict(l=120, r=160 + RIGHT_LEGEND_PX)
DATE_CHART_RIGHT_LEGEND = dict(fixed_height=True, legend_position="right")


def _pannable_date_view(
    figure: go.Figure,
    data_start: date,
    data_end: date,
    view_start: date | None,
    view_end: date | None,
) -> None:
    """Show ``view_start``..``view_end`` but let the user pan across all data.

    Only the date axis moves: y-axes are fixed and panning stops at the first
    and last day with data.
    """
    lower = pd.Timestamp(data_start)
    upper = pd.Timestamp(data_end) + pd.Timedelta(days=1)
    visible = [
        max(lower, pd.Timestamp(view_start)) if view_start else lower,
        min(upper, pd.Timestamp(view_end) + pd.Timedelta(days=1)) if view_end else upper,
    ]
    figure.update_xaxes(range=visible, minallowed=lower, maxallowed=upper)
    figure.update_yaxes(fixedrange=True)
    figure.update_layout(meta={**DATE_CHART_RIGHT_LEGEND, "pan": True})


def hour_px(figure: go.Figure) -> float:
    """Pixels per hour on the primary y-axis of a figure with fixed geometry."""
    layout = figure.layout
    plot_height = layout.height - layout.margin.t - layout.margin.b
    domain_start, domain_end = layout.yaxis.domain or (0.0, 1.0)
    range_start, range_end = layout.yaxis.range
    return plot_height * (domain_end - domain_start) / (range_end - range_start)


def _add_no_data_markers(figure: go.Figure, frame: pd.DataFrame) -> None:
    """Mark days with no sleep record from any source in grey.

    Unlike red zero-sleep markers, these say nothing about how long the person
    slept: the watch was not worn or not synced and Sleepmeter has no record.
    """
    no_data = frame[frame["total_sleep_minutes"].isna()]
    if no_data.empty:
        return
    figure.add_trace(
        go.Bar(
            name="No data",
            x=no_data["date_plot"],
            y=[0.08] * len(no_data),
            base=[0] * len(no_data),
            marker=dict(color=NO_DATA_COLOR, line=dict(width=0)),
            hovertemplate=(
                "<b>No data</b><br>%{x|%d/%m/%Y}<br>"
                "No Garmin or Sleepmeter record<extra></extra>"
            ),
        )
    )


def _stacked_bar_top(figure: go.Figure) -> float:
    """Highest point reached by bars on the primary y-axis."""
    stacked: dict[object, float] = {}
    top = 0.0
    for trace in figure.data:
        if trace.type != "bar" or trace.yaxis not in (None, "y"):
            continue
        for x_value, y_value in zip(trace.x, trace.y):
            if y_value is None or not math.isfinite(y_value):
                continue
            if trace.base is not None:
                top = max(top, float(y_value))
            else:
                stacked[x_value] = stacked.get(x_value, 0.0) + float(y_value)
    return max([top, *stacked.values()])


def _add_menstrual_strip(
    figure: go.Figure, dates: pd.Series, phases: pd.Series
) -> None:
    """Draw one phase-coloured day block per date in a fixed-height strip.

    The strip is sized in pixels and sits directly above the date axis at the
    bottom of the plot area, so it looks the same in every chart and height.
    """
    for day, phase in zip(dates, phases):
        color = MENSTRUAL_PHASE_COLORS.get(phase)
        if color is None:
            continue
        day_start = pd.Timestamp(day)
        figure.add_shape(
            type="rect",
            x0=day_start,
            x1=day_start + pd.Timedelta(days=1),
            xref="x",
            yref="paper",
            ysizemode="pixel",
            yanchor=0,
            y0=0,
            y1=MENSTRUAL_STRIP_PX,
            fillcolor=color,
            line=dict(width=0),
            layer="above",
        )


def _load_aid_colors(path: Path) -> dict[str, dict[str, str]]:
    """Read per-aid colours kept outside Git, keyed by case-folded aid name."""
    if not path.exists():
        return {}
    configured = json.loads(path.read_text(encoding="utf-8"))
    return {str(name).casefold(): colors for name, colors in configured.items()}


AID_COLORS = _load_aid_colors(AID_COLORS_PATH)


def _aid_colors(aid: object) -> dict[str, str]:
    aid_key = str(aid).casefold()
    if aid_key == "none":
        return NO_AID_COLORS
    if aid_key in AID_COLORS:
        return AID_COLORS[aid_key]
    index = zlib.crc32(aid_key.encode("utf-8")) % len(FALLBACK_AID_PALETTE)
    return FALLBACK_AID_PALETTE[index]


def _aid_color(aid: object) -> str:
    return _aid_colors(aid)["color"]


def _aid_nap_color(aid: object) -> str:
    colors = _aid_colors(aid)
    return colors.get("nap", colors["color"])


def _aid_fill_color(aid: object) -> str:
    hex_color = _aid_color(aid).lstrip("#")
    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, 0.5)"


def _linear_trend(values: pd.Series) -> pd.Series:
    """Return a least-squares trend over equally spaced observations."""
    positions = pd.Series(range(len(values)), dtype="float64")
    numeric = pd.Series(values, dtype="float64").reset_index(drop=True)
    if len(numeric) < 2:
        return numeric
    centered = positions - positions.mean()
    denominator = float((centered**2).sum())
    if denominator == 0:
        return pd.Series([numeric.mean()] * len(numeric), dtype="float64")
    slope = float((centered * (numeric - numeric.mean())).sum() / denominator)
    return numeric.mean() + slope * centered


def _spearman_summary(
    x_values: pd.Series, y_values: pd.Series
) -> tuple[float, int] | None:
    paired = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
    if len(paired) < 3 or paired["x"].nunique() < 2 or paired["y"].nunique() < 2:
        return None
    ranked = paired.rank(method="average")
    correlation = ranked["x"].corr(ranked["y"])
    if pd.isna(correlation):
        return None
    return float(correlation), int(len(paired))


def _add_spearman_annotation(
    figure: go.Figure, x_values: pd.Series, y_values: pd.Series
) -> None:
    summary = _spearman_summary(x_values, y_values)
    if summary is None:
        return
    correlation, count = summary
    figure.add_annotation(
        x=0.99,
        y=0.99,
        xref="paper",
        yref="paper",
        text=f"Spearman ρ={correlation:+.2f} · n={count}",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(14,17,23,0.72)",
        borderpad=4,
        font=dict(size=11, color="#d1d5db"),
    )


def _format_signed_duration(hours: float) -> str:
    total_minutes = int(round(abs(float(hours)) * 60))
    sign = "-" if hours < 0 else "+" if hours > 0 else ""
    return f"{sign}{total_minutes // 60}:{total_minutes % 60:02d}"


def _format_clock_hour(hours: float) -> str:
    total_minutes = int(round(float(hours) * 60)) % (24 * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def _format_duration_hour(hours: float) -> str:
    total_minutes = max(0, int(round(float(hours) * 60)))
    return f"{total_minutes // 60}:{total_minutes % 60:02d}"


def _clock_hour(value: datetime, boundary_hour: int = 0) -> float:
    hour = value.hour + value.minute / 60 + value.second / 3600
    return hour + 24 if hour < boundary_hour else hour


def _date_axis_limits(values: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return day-aligned bounds that contain every timestamp without empty tails."""
    start = pd.Timestamp(values.min()).normalize()
    end = pd.Timestamp(values.max()).normalize() + pd.Timedelta(days=1)
    return start, end


def _nice_tick_step(span: float, target_ticks: int = 5) -> float:
    rough_step = max(span / target_ticks, 1.0)
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    for candidate in (1, 2, 5, 10):
        if normalized <= candidate:
            return candidate * magnitude
    return 10 * magnitude


def _stats_timeline_figure(
    dates: pd.Series,
    values: pd.Series,
    *,
    title: str,
    yaxis_title: str,
    target: float,
    y_range: tuple[float, float],
    tick_values: list[float],
    tick_text: list[str],
    formatted_values: list[str],
    color: str = STATS_GREEN,
) -> go.Figure:
    numeric = pd.Series(values, dtype="float64").reset_index(drop=True)
    timeline = pd.Series(dates).reset_index(drop=True)
    if numeric.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No matching sleep records in the selected date range",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        figure.update_layout(
            title=title,
            height=540,
            margin=dict(l=90, r=24, t=64, b=64),
            xaxis=dict(title="Date", tickformat="%d/%m", gridcolor="#303030"),
            yaxis=dict(
                title=yaxis_title,
                range=list(y_range),
                tickmode="array",
                tickvals=tick_values,
                ticktext=tick_text,
                gridcolor="#303030",
            ),
        )
        return figure
    trend = _linear_trend(numeric)
    x_start, x_end = _date_axis_limits(timeline)
    mean = float(numeric.mean())
    std_value = numeric.std(ddof=1)
    std = float(std_value) if pd.notna(std_value) else 0.0

    figure = go.Figure()
    figure.add_hrect(
        y0=max(y_range[0], mean - std),
        y1=min(y_range[1], mean + std),
        fillcolor=color,
        opacity=0.14,
        line_width=0,
        layer="below",
    )
    figure.add_hline(
        y=mean,
        line=dict(color=color, width=1.5, dash="dash"),
    )
    figure.add_hline(y=target, line=dict(color=STATS_RED, width=2))
    figure.add_trace(
        go.Scatter(
            name="Observed",
            x=timeline,
            y=numeric,
            mode="lines",
            line=dict(color=color, width=1.8),
            customdata=[[item] for item in formatted_values],
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Value: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Trend",
            x=timeline,
            y=trend,
            mode="lines",
            line=dict(color=color, width=2.2),
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title=title,
        showlegend=False,
        height=540,
        margin=dict(l=90, r=24, t=64, b=64),
        hovermode="x",
        xaxis=dict(
            title="Date",
            type="date",
            range=[x_start, x_end],
            minallowed=x_start,
            maxallowed=x_end,
            tickformat="%d/%m",
            gridcolor="#303030",
        ),
        yaxis=dict(
            title=yaxis_title,
            range=list(y_range),
            minallowed=y_range[0],
            maxallowed=y_range[1],
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            gridcolor="#303030",
            zeroline=False,
        ),
    )
    return figure


def _clock_phase_minute(value: datetime) -> float:
    minute = value.hour * 60 + value.minute + value.second / 60
    return (minute - CLOCK_BOUNDARY_MINUTE) % 1440


def _next_clock_boundary(value: datetime) -> datetime:
    return datetime.combine(
        value.date() + timedelta(days=1), time.min, tzinfo=value.tzinfo
    )


def _sleepmeter_clock_segments(
    sessions: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    """Represent intervals on an ordinary repeating 00:00-to-24:00 clock."""
    if intervals.empty:
        return pd.DataFrame()
    session_details = sessions[
        [
            "session_id",
            "sleep_start",
            "wake_time",
            "session_type",
            "sleep_minutes",
            "aid_name",
            "legacy_garmin_score_10",
        ]
    ].rename(
        columns={
            "sleep_start": "session_start",
            "wake_time": "session_wake",
            "session_type": "session_kind",
            "aid_name": "session_aid",
            "legacy_garmin_score_10": "session_score",
        }
    )
    joined = intervals.merge(session_details, on="session_id", how="inner")
    segments: list[dict[str, object]] = []
    for row in joined.itertuples(index=False):
        cursor = row.interval_start
        while cursor < row.interval_end:
            boundary = _next_clock_boundary(cursor)
            segment_end = min(row.interval_end, boundary)
            start_phase = _clock_phase_minute(cursor)
            end_phase = (
                1440.0
                if segment_end == boundary
                else _clock_phase_minute(segment_end)
            )
            if end_phase > start_phase:
                segments.append(
                    {
                        "session_id": row.session_id,
                        "x": datetime.combine(
                            row.session_wake.date(),
                            time(12, 0),
                            tzinfo=row.session_wake.tzinfo,
                        ),
                        "start_phase": start_phase,
                        "end_phase": end_phase,
                        "session_start": row.session_start,
                        "session_wake": row.session_wake,
                        "session_type": row.session_kind,
                        "sleep_minutes": row.sleep_minutes,
                        "aid_name": row.session_aid,
                        "legacy_garmin_score_10": row.session_score,
                    }
                )
            cursor = segment_end
    return pd.DataFrame(segments)


def sleep_schedule_figure(
    sessions: pd.DataFrame,
    intervals: pd.DataFrame,
    daily: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> go.Figure:
    segments = _sleepmeter_clock_segments(sessions, intervals)
    figure = go.Figure()

    if not segments.empty:
        segment_colors = segments["session_type"].map(
            {"NIGHT_SLEEP": NIGHT_BAR_COLOR, "NAP": NAP_BAR_COLOR}
        )
        customdata = [
            [
                row.session_start.strftime("%d/%m/%Y %H:%M"),
                row.session_wake.strftime("%d/%m/%Y %H:%M"),
                row.session_type.replace("_", " ").title(),
                row.aid_name,
                (
                    f"{int(row.legacy_garmin_score_10 * 10)}/100"
                    if pd.notna(row.legacy_garmin_score_10)
                    else "Not recorded"
                ),
                row.sleep_minutes / 60,
                row.session_id,
            ]
            for row in segments.itertuples(index=False)
        ]
        figure.add_trace(
            go.Bar(
                name="Sleep record",
                x=segments["x"],
                y=(segments["end_phase"] - segments["start_phase"]) / 60,
                base=segments["start_phase"] / 60,
                marker=dict(color=segment_colors),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[2]}</b><br>"
                    "%{customdata[0]} → %{customdata[1]}<br>"
                    "Asleep: %{customdata[5]:.2f} h<br>"
                    "Aid: %{customdata[3]}<br>"
                    "Garmin Sleep score: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    zero_days = daily[daily["zero_sleep"]].copy()
    if not zero_days.empty:
        zero_days["x"] = pd.to_datetime(zero_days["date"]) + pd.Timedelta(hours=12)
        zero_days["session_ids"] = zero_days["date"].map(
            lambda day: ",".join(
                sessions.loc[
                    (sessions["sleep_date"] == day) & sessions["zero_duration"],
                    "session_id",
                ]
            )
        )
        figure.add_trace(
            go.Bar(
                name="Zero sleep",
                x=zero_days["x"],
                y=[0.08] * len(zero_days),
                base=[0] * len(zero_days),
                marker=dict(
                    color=ZERO_COLOR,
                    line=dict(width=0),
                ),
                customdata=zero_days[["session_ids", "source_state"]],
                hovertemplate=(
                    "<b>Zero sleep</b><br>%{x|%d/%m/%Y}<br>"
                    "%{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_traces(width=0.8 * 24 * 60 * 60 * 1000)  # ms
    chart_start = pd.Timestamp(start_date)
    chart_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    figure.update_layout(
        title="Daily Sleep Bar Graph",
        showlegend=False,
        barmode="overlay",
        bargap=0.2,
        dragmode=False,
        hovermode="closest",
        height=650,
        margin=dict(l=88, r=22, t=64, b=72),
        xaxis=dict(
            title="Date",
            type="date",
            range=[chart_start, chart_end],
            minallowed=chart_start,
            maxallowed=chart_end,
            tickformat="%d/%m",
            nticks=10,
            showgrid=False,
            zeroline=False,
            linecolor="#dddddd",
            tickfont=dict(size=14),
            title_font=dict(size=18),
        ),
        yaxis=dict(
            title="Time of day",
            range=[-0.5, 24.5],
            minallowed=-0.5,
            maxallowed=24.5,
            tickmode="array",
            tickvals=[0, 4, 8, 12, 16, 20, 24],
            ticktext=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
            showgrid=True,
            minor=dict(showgrid=True, dtick=1, gridcolor="#202020"),
            gridcolor="#444444",
            zeroline=False,
            linecolor="#dddddd",
            tickfont=dict(size=14),
            title_font=dict(size=18),
        ),
    )
    return figure


def sleep_stage_schedule_figure(
    sessions: pd.DataFrame,
    intervals: pd.DataFrame,
    sleep_stages: pd.DataFrame,
    activities: pd.DataFrame,
    menstrual: pd.DataFrame,
    daily: pd.DataFrame,
    start_date: date,
    end_date: date,
    previous_colors: bool = False,
    show_sleep_stages: bool = True,
    garmin_metrics: pd.DataFrame | None = None,
    match_hour_px: float | None = None,
    view_start: date | None = None,
    view_end: date | None = None,
) -> go.Figure:
    """Show Garmin sleep stages on the same 00:00–24:00 layout as the schedule.

    ``match_hour_px`` sizes the plot so an hour is that many pixels tall, e.g.
    ``hour_px(daily_duration_figure(...))`` to match the duration chart.
    ``start_date``..``end_date`` is the pannable extent; ``view_start`` and
    ``view_end`` pick the window shown first.
    """
    figure = go.Figure()
    session_details = sessions[
        ["session_id", "wake_time", "sleep_start", "session_type"]
    ].rename(columns={"wake_time": "session_wake"})

    stage_segments: list[dict[str, object]] = []
    if not sleep_stages.empty:
        joined = sleep_stages.merge(session_details, on="session_id", how="inner")
        for row in joined.itertuples(index=False):
            cursor = row.interval_start
            while cursor < row.interval_end:
                boundary = _next_clock_boundary(cursor)
                segment_end = min(row.interval_end, boundary)
                start_phase = _clock_phase_minute(cursor)
                end_phase = (
                    1440.0
                    if segment_end == boundary
                    else _clock_phase_minute(segment_end)
                )
                if end_phase > start_phase:
                    stage_segments.append(
                        {
                            "session_id": row.session_id,
                            "x": datetime.combine(
                                row.session_wake.date(),
                                time(12, 0),
                                tzinfo=row.session_wake.tzinfo,
                            ),
                            "start_phase": start_phase,
                            "end_phase": end_phase,
                            "interval_start": cursor,
                            "interval_end": segment_end,
                            "stage": row.stage,
                        }
                    )
                cursor = segment_end

    stage_frame = pd.DataFrame(stage_segments)
    stage_colors = {
        "Deep": "#3A1F99" if previous_colors else "#331B8C",
        "Light": "#8F6BFF" if previous_colors else "#5B32D6",
        "REM": "#EE55C0",
        "Unmeasurable": "#6B7280",
        "Stage unavailable": "#6345CC" if previous_colors else "#4B29B8",
    }
    unavailable_marker = dict(color=stage_colors["Stage unavailable"])
    if not previous_colors:
        unavailable_marker = dict(
            color="#5B32D6",
            line=dict(width=0),
            pattern=dict(
                shape="/",
                bgcolor="#5B32D6",
                fgcolor="#331B8C",
                fillmode="replace",
                size=6,
                solidity=0.33,
            ),
        )
    stage_order = ["Deep", "Light", "REM", "Unmeasurable", "Stage unavailable"]
    if not stage_frame.empty:
        for stage_name in stage_order:
            stage_rows = stage_frame[stage_frame["stage"] == stage_name]
            if stage_rows.empty:
                continue
            customdata = [
                [
                    row.interval_start.strftime("%d/%m/%Y %H:%M"),
                    row.interval_end.strftime("%d/%m/%Y %H:%M"),
                ]
                for row in stage_rows.itertuples(index=False)
            ]
            figure.add_trace(
                go.Bar(
                    name=stage_name,
                    x=stage_rows["x"],
                    y=(stage_rows["end_phase"] - stage_rows["start_phase"]) / 60,
                    base=stage_rows["start_phase"] / 60,
                    marker=(
                        unavailable_marker
                        if not previous_colors
                        and stage_name in ("Unmeasurable", "Stage unavailable")
                        else dict(color=stage_colors[stage_name])
                    ),
                    legendgroup=(
                        "unavailable"
                        if stage_name == "Stage unavailable"
                        else stage_name
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        f"<b>{stage_name}</b><br>"
                        "%{customdata[0]} → %{customdata[1]}<extra></extra>"
                    ),
                )
            )

    staged_session_ids = (
        set(sleep_stages["session_id"].unique()) if not sleep_stages.empty else set()
    )
    unavailable_intervals = intervals[
        ~intervals["session_id"].isin(staged_session_ids)
    ]
    unavailable = _sleepmeter_clock_segments(sessions, unavailable_intervals)
    if not unavailable.empty:
        for session_type, trace_name, color in (
            (
                "NIGHT_SLEEP",
                "Stage unavailable" if show_sleep_stages else "Sleep",
                stage_colors["Stage unavailable"] if show_sleep_stages else NIGHT_BAR_COLOR,
            ),
            ("NAP", "Nap", NAP_BAR_COLOR),
        ):
            unavailable_rows = unavailable[
                unavailable["session_type"] == session_type
            ]
            if unavailable_rows.empty:
                continue
            figure.add_trace(
                go.Bar(
                    name=trace_name,
                    x=unavailable_rows["x"],
                    y=(
                        unavailable_rows["end_phase"]
                        - unavailable_rows["start_phase"]
                    )
                    / 60,
                    base=unavailable_rows["start_phase"] / 60,
                    marker=(
                        unavailable_marker
                        if session_type == "NIGHT_SLEEP" and show_sleep_stages
                        else dict(color=color)
                    ),
                    legendgroup=(
                        ("unavailable" if show_sleep_stages else "sleep")
                        if session_type == "NIGHT_SLEEP"
                        else "nap"
                    ),
                    showlegend=(
                        not show_sleep_stages
                        or stage_frame.empty
                        or "Stage unavailable" not in set(stage_frame["stage"])
                        if session_type == "NIGHT_SLEEP"
                        else True
                    ),
                    customdata=[
                        [
                            row.session_start.strftime("%d/%m/%Y %H:%M"),
                            row.session_wake.strftime("%d/%m/%Y %H:%M"),
                        ]
                        for row in unavailable_rows.itertuples(index=False)
                    ],
                    hovertemplate=(
                        f"<b>{trace_name}</b><br>"
                        "%{customdata[0]} → %{customdata[1]}<extra></extra>"
                    ),
                )
            )

    if not previous_colors:
        # Unmeasurable and unavailable share one striped fill, so give them a
        # single legend entry; hover text still names the specific stage.
        unavailable_traces = [
            trace
            for trace in figure.data
            if trace.name in ("Unmeasurable", "Stage unavailable")
        ]
        for index, trace in enumerate(unavailable_traces):
            trace.update(
                name="Unavailable / unmeasurable",
                legendgroup="unavailable",
                showlegend=index == 0,
            )

    activity_segments: list[dict[str, object]] = []
    if not activities.empty:
        for row in activities.itertuples(index=False):
            cursor = row.activity_start
            while cursor < row.activity_end:
                boundary = _next_clock_boundary(cursor)
                segment_end = min(row.activity_end, boundary)
                start_phase = _clock_phase_minute(cursor)
                end_phase = (
                    1440.0
                    if segment_end == boundary
                    else _clock_phase_minute(segment_end)
                )
                if end_phase > start_phase:
                    activity_segments.append(
                        {
                            "x": datetime.combine(
                                cursor.date(), time(12, 0), tzinfo=cursor.tzinfo
                            ),
                            "start_phase": start_phase,
                            "end_phase": end_phase,
                            "activity_start": row.activity_start,
                            "activity_end": row.activity_end,
                            "activity_name": row.activity_name,
                            "activity_type": row.activity_type,
                        }
                    )
                cursor = segment_end
    activity_frame = pd.DataFrame(activity_segments)
    if not activity_frame.empty:
        figure.add_trace(
            go.Bar(
                name="Garmin activity",
                x=activity_frame["x"],
                y=(
                    activity_frame["end_phase"] - activity_frame["start_phase"]
                )
                / 60,
                base=activity_frame["start_phase"] / 60,
                marker=dict(color="#4DB88A", line=dict(width=0)),
                customdata=[
                    [
                        row.activity_name,
                        row.activity_type.replace("_", " ").title(),
                        row.activity_start.strftime("%d/%m/%Y %H:%M"),
                        row.activity_end.strftime("%d/%m/%Y %H:%M"),
                    ]
                    for row in activity_frame.itertuples(index=False)
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "%{customdata[2]} → %{customdata[3]}<extra></extra>"
                ),
            )
        )

    hrv_top = None
    if garmin_metrics is not None and not garmin_metrics.empty:
        hrv_frame = garmin_metrics.copy()
        hrv_frame["date_plot"] = pd.to_datetime(hrv_frame["date"]) + pd.Timedelta(
            hours=12
        )
        hrv_frame["nightly_hrv"] = hrv_frame[
            "hrv_last_night_average"
        ].fillna(hrv_frame["overnight_hrv"])
        hrv_top = pd.concat(
            [hrv_frame["nightly_hrv"], hrv_frame["hrv_weekly_average"]]
        ).max()
        figure.add_trace(
            go.Scatter(
                name="Nightly average HRV",
                x=hrv_frame["date_plot"],
                y=hrv_frame["nightly_hrv"],
                yaxis="y2",
                mode="lines",
                visible="legendonly",
                connectgaps=False,
                line=dict(color=HRV_GREEN, width=1),
                hovertemplate=(
                    "%{x|%d/%m/%Y}<br>Nightly average HRV: %{y:.0f} ms<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                name="7-day average HRV",
                x=hrv_frame["date_plot"],
                y=hrv_frame["hrv_weekly_average"],
                yaxis="y2",
                mode="lines",
                visible="legendonly",
                connectgaps=False,
                line=dict(color=HRV_WEEKLY_GREEN, width=1, dash="dot"),
                hovertemplate=(
                    "%{x|%d/%m/%Y}<br>7-day average HRV: %{y:.0f} ms<extra></extra>"
                ),
            )
        )

    scored_nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    if not scored_nights.empty:
        scores_by_date = (
            scored_nights.groupby("sleep_date", as_index=False)[
                "legacy_garmin_score_10"
            ].mean()
        )
        score_frame = daily[["date"]].copy().rename(columns={"date": "sleep_date"})
        score_frame = score_frame.merge(scores_by_date, on="sleep_date", how="left")
        score_frame["date_plot"] = pd.to_datetime(
            score_frame["sleep_date"]
        ) + pd.Timedelta(hours=12)
        score_frame["score_100"] = score_frame["legacy_garmin_score_10"] * 10
        score_frame["score_7_day"] = score_frame["score_100"].rolling(
            7, min_periods=1
        ).mean()
        figure.add_trace(
            go.Scatter(
                name="Garmin Sleep score",
                x=score_frame["date_plot"],
                y=score_frame["score_100"],
                yaxis="y3",
                mode="lines",
                visible="legendonly",
                connectgaps=False,
                line=dict(color=SLEEP_SCORE_PINK, width=1),
                hovertemplate=(
                    "%{x|%d/%m/%Y}<br>Garmin Sleep score: %{y:.0f}/100<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                name="7-day average Sleep score",
                x=score_frame["date_plot"],
                y=score_frame["score_7_day"],
                yaxis="y3",
                mode="lines",
                visible="legendonly",
                connectgaps=False,
                line=dict(color=SLEEP_SCORE_PINK, width=1, dash="dot"),
                hovertemplate=(
                    "%{x|%d/%m/%Y}<br>7-day average Sleep score: %{y:.1f}/100<extra></extra>"
                ),
            )
        )

    if not menstrual.empty:
        _add_menstrual_strip(figure, menstrual["date"], menstrual["current_phase"])
        for cycle_start in menstrual.loc[menstrual["cycle_day"] == 1, "date"]:
            cycle_boundary = pd.Timestamp(cycle_start)
            figure.add_shape(
                type="line",
                x0=cycle_boundary,
                x1=cycle_boundary,
                xref="x",
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="rgba(255,255,255,0.20)", width=1),
                layer="below",
            )
        # Invisible point that gives the strip a "Menstrual phase" axis label,
        # matching how the duration chart labels its strip.
        figure.add_trace(
            go.Scatter(
                x=[pd.Timestamp(start_date)],
                y=["Menstrual phase"],
                yaxis="y4",
                mode="markers",
                marker=dict(opacity=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    zero_days = daily[daily["zero_sleep"]].copy()
    if not zero_days.empty:
        zero_days["x"] = pd.to_datetime(zero_days["date"]) + pd.Timedelta(hours=12)
        figure.add_trace(
            go.Bar(
                name="Zero sleep",
                x=zero_days["x"],
                y=[0.08] * len(zero_days),
                base=[0] * len(zero_days),
                marker_color=ZERO_COLOR,
                hovertemplate="<b>Zero sleep</b><br>%{x|%d/%m/%Y}<extra></extra>",
            )
        )

    figure.update_traces(
        width=0.8 * 24 * 60 * 60 * 1000,
        selector=dict(type="bar"),
    )
    chart_start = pd.Timestamp(start_date)
    chart_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    figure.update_layout(
        title=(
            "Daily Sleep Stages — Previous Colors"
            if previous_colors
            else "Daily Sleep Stages"
        ),
        barmode="overlay",
        bargap=0.2,
        dragmode=False,
        hovermode="closest",
        height=650,
        # Shared side margins keep each day the same width as in the daily
        # duration chart.
        margin=dict(**DATE_CHART_SIDE_MARGINS, t=76, b=96),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        xaxis=dict(
            title="Date",
            type="date",
            range=[chart_start, chart_end],
            minallowed=chart_start,
            maxallowed=chart_end,
            tickformat="%d/%m",
            # Never rotate: rotated labels would need more bottom margin.
            tickangle=0,
            showgrid=False,
            zeroline=False,
            # No axis line: it would run directly under the menstrual strip.
            showline=False,
            tickfont=dict(size=14),
            title_font=dict(size=18),
            # Dates sit below the menstrual strip band rather than under it.
            anchor="free",
            position=0,
        ),
        yaxis=dict(
            title="Time of day",
            domain=[0.11 if not menstrual.empty else 0.0, 1.0],
            range=[-0.5, 24.5],
            minallowed=-0.5,
            maxallowed=24.5,
            tickmode="array",
            tickvals=[0, 4, 8, 12, 16, 20, 24],
            ticktext=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
            showgrid=True,
            minor=dict(showgrid=True, dtick=1, gridcolor="#202020"),
            gridcolor="#444444",
            zeroline=False,
            linecolor="#dddddd",
            tickfont=dict(size=14),
            title_font=dict(size=18),
        ),
        yaxis2=dict(
            title="HRV (ms)",
            overlaying="y",
            side="right",
            # The HRV lines start hidden, so autorange would label an empty
            # 0-4 axis; size it from the data instead.
            range=[0, hrv_top * 1.1] if pd.notna(hrv_top) and hrv_top else None,
            # Own round ticks rather than ones synced to the time-of-day grid.
            tickmode="linear",
            dtick=10,
            rangemode="tozero",
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=HRV_GREEN),
            title_font=dict(color=HRV_GREEN),
        ),
        yaxis3=dict(
            title="Sleep score (0–100)",
            overlaying="y",
            side="right",
            anchor="free",
            position=1.0,
            autoshift=True,
            range=[0, 100],
            dtick=20,
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=SLEEP_SCORE_PINK),
            title_font=dict(color=SLEEP_SCORE_PINK),
        ),
        yaxis4=dict(
            domain=[0.0, 0.02],
            anchor="x",
            type="category",
            categoryarray=["Menstrual phase"],
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#cbd5e1", size=11),
        ),
    )
    _pannable_date_view(figure, start_date, end_date, view_start, view_end)
    if match_hour_px:
        domain_start, domain_end = figure.layout.yaxis.domain
        range_start, range_end = figure.layout.yaxis.range
        plot_height = (
            match_hour_px * (range_end - range_start) / (domain_end - domain_start)
        )
        figure.update_layout(
            height=round(
                figure.layout.margin.t + figure.layout.margin.b + plot_height
            )
        )
    return figure


def sleep_score_components_figure(metrics: pd.DataFrame) -> go.Figure:
    """Show Garmin sleep-score component qualifiers across time."""
    if metrics.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No Garmin sleep-score components in the selected date range",
            showarrow=False,
        )
        figure.update_layout(title="Garmin Sleep score components", height=360)
        return figure

    frame = metrics.sort_values("date").copy()
    qualifier_rank = {"POOR": 1, "FAIR": 2, "GOOD": 3, "EXCELLENT": 4}
    components = (
        ("Total duration", "score_duration_qualifier", None, ""),
        ("Sleep stress", "score_stress_qualifier", "average_sleep_stress", ""),
        (
            "Restlessness",
            "score_restlessness_qualifier",
            "restless_moment_count",
            " moments",
        ),
        ("Awake count", "score_awake_count_qualifier", "awake_count", ""),
        ("Deep sleep", "score_deep_qualifier", "score_deep_percentage", "%"),
        ("REM sleep", "score_rem_qualifier", "score_rem_percentage", "%"),
        ("Light sleep", "score_light_qualifier", "score_light_percentage", "%"),
    )
    z_rows: list[list[float | None]] = []
    text_rows: list[list[str]] = []
    for _, qualifier_column, value_column, suffix in components:
        qualifiers = (
            frame[qualifier_column]
            if qualifier_column in frame
            else pd.Series([None] * len(frame), index=frame.index)
        )
        values = (
            frame[value_column]
            if value_column is not None and value_column in frame
            else pd.Series([None] * len(frame), index=frame.index)
        )
        z_rows.append(
            [qualifier_rank.get(str(value).upper()) if pd.notna(value) else None for value in qualifiers]
        )
        row_text: list[str] = []
        for qualifier, value in zip(qualifiers, values):
            if pd.isna(qualifier):
                row_text.append("Unavailable")
                continue
            label = str(qualifier).replace("_", " ").title()
            if pd.notna(value):
                label += f" · {float(value):.0f}{suffix}"
            row_text.append(label)
        text_rows.append(row_text)

    colorscale = [
        [0.0, "#FF5A4D"],
        [0.2499, "#FF5A4D"],
        [0.25, "#FF9F3F"],
        [0.4999, "#FF9F3F"],
        [0.5, "#C5DC4A"],
        [0.7499, "#C5DC4A"],
        [0.75, "#3FD67A"],
        [1.0, "#3FD67A"],
    ]
    figure = go.Figure(
        go.Heatmap(
            x=pd.to_datetime(frame["date"]),
            y=[component[0] for component in components],
            z=z_rows,
            text=text_rows,
            zmin=0.5,
            zmax=4.5,
            colorscale=colorscale,
            colorbar=dict(
                title="Qualifier",
                tickmode="array",
                tickvals=[1, 2, 3, 4],
                ticktext=["Poor", "Fair", "Good", "Excellent"],
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}<br><b>%{y}</b><br>%{text}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        )
    )
    figure.update_layout(
        title="Garmin Sleep score components",
        height=470,
        margin=dict(l=125, r=80, t=70, b=60),
        xaxis=dict(title="Sleep date", tickformat="%d/%m", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    return figure


def overnight_recovery_figure(metrics: pd.DataFrame) -> go.Figure:
    """Show key overnight recovery measurements as aligned small multiples."""
    panels = (
        ("Overnight HRV", "hrv_last_night_average", "ms", "#4DB88A"),
        ("Sleeping heart rate", "average_sleep_hr", "bpm", "#E8793F"),
        ("Sleep stress", "average_sleep_stress", "", "#E6B84F"),
        ("Body Battery gained", "body_battery_change", "", "#2FD6A3"),
        ("Sleeping respiration", "average_respiration", "brpm", "#087CFF"),
        ("Average SpO₂", "average_spo2", "%", "#EE55C0"),
    )
    figure = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        subplot_titles=[panel[0] for panel in panels],
    )
    if metrics.empty:
        figure.add_annotation(
            text="No overnight recovery measurements in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        frame = metrics.sort_values("date").copy()
        frame["overnight_hrv_display"] = frame[
            "hrv_last_night_average"
        ].fillna(frame["overnight_hrv"])
        for index, (title, column, unit, color) in enumerate(panels):
            row, col = divmod(index, 2)
            row += 1
            col += 1
            values = (
                frame["overnight_hrv_display"]
                if column == "hrv_last_night_average"
                else frame[column]
            )
            figure.add_trace(
                go.Scatter(
                    name=title,
                    x=pd.to_datetime(frame["date"]),
                    y=values,
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=color, width=1.4),
                    showlegend=False,
                    hovertemplate=(
                        f"%{{x|%d/%m/%Y}}<br>{title}: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            figure.update_yaxes(title_text=unit, showgrid=True, row=row, col=col)
    figure.update_xaxes(tickformat="%d/%m", showgrid=False)
    figure.update_layout(
        title="Overnight recovery panel",
        height=760,
        margin=dict(l=72, r=32, t=90, b=55),
        hovermode="x unified",
    )
    return figure


def sleep_composition_figure(
    metrics: pd.DataFrame,
    view_start: date | None = None,
    view_end: date | None = None,
    extent_start: date | None = None,
    extent_end: date | None = None,
) -> go.Figure:
    """Show each Garmin night's sleep-stage composition normalized to 100%.

    Opens on ``view_start``..``view_end`` and pans across ``extent_start``..
    ``extent_end`` (default: the metrics' own dates), matching the other
    Overview date charts when given their extent.
    """
    figure = go.Figure()
    if metrics.empty:
        figure.add_annotation(
            text="No Garmin sleep composition in the selected date range",
            showarrow=False,
        )
        figure.update_layout(title="Nightly sleep composition", height=390)
        return figure

    frame = metrics.sort_values("date").copy()
    stage_columns = {
        "Deep": "deep_sleep_minutes",
        "REM": "rem_sleep_minutes",
        "Light": "light_sleep_minutes",
        "Awake": "awake_minutes",
    }
    total = frame[list(stage_columns.values())].fillna(0).sum(axis=1)
    frame = frame[total > 0].copy()
    total = total[total > 0]
    colors = {
        "Deep": "#331B8C",
        "REM": "#EE55C0",
        "Light": "#5B32D6",
        "Awake": "#9CA3AF",
    }
    for stage_name, column in stage_columns.items():
        percentages = frame[column].fillna(0) / total * 100
        figure.add_trace(
            go.Bar(
                name=stage_name,
                x=pd.to_datetime(frame["date"]) + pd.Timedelta(hours=12),
                y=percentages,
                marker=dict(color=colors[stage_name], line=dict(width=0)),
                customdata=frame[[column]],
                hovertemplate=(
                    f"%{{x|%d/%m/%Y}}<br>{stage_name}: %{{y:.1f}}%"
                    " (%{customdata[0]:.0f} min)<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Nightly sleep composition",
        barmode="stack",
        bargap=0.2,
        height=470,
        margin=dict(**DATE_CHART_SIDE_MARGINS, t=64, b=72),
        xaxis=dict(
            title="Sleep date", type="date", tickformat="%d/%m", tickangle=0, showgrid=False
        ),
        yaxis=dict(title="Share of sleep window", range=[0, 100], ticksuffix="%"),
        hovermode="x unified",
    )
    _pannable_date_view(
        figure,
        extent_start or frame["date"].min(),
        extent_end or frame["date"].max(),
        view_start,
        view_end,
    )
    return figure


def daily_physiological_load_figure(metrics: pd.DataFrame) -> go.Figure:
    """Show daily activity and physiological load as aligned small multiples."""
    panels = (
        ("Steps", "daily_steps", "steps", "#087CFF"),
        ("Intensity minutes", "daily_intensity_minutes", "min", "#5B32D6"),
        ("Resting heart rate", "daily_resting_heart_rate", "bpm", "#E8793F"),
        ("Average stress", "daily_average_stress", "", "#E6B84F"),
        ("Body Battery drained", "daily_body_battery_drained", "", "#EE55C0"),
        ("Active calories", "daily_active_calories", "kcal", "#4DB88A"),
    )
    figure = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        subplot_titles=[panel[0] for panel in panels],
    )
    if metrics.empty:
        figure.add_annotation(
            text="No daily physiological-load data in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        frame = metrics.sort_values("date")
        for index, (title, column, unit, color) in enumerate(panels):
            row, col = divmod(index, 2)
            row += 1
            col += 1
            values = (
                frame[column]
                if column in frame
                else pd.Series([float("nan")] * len(frame), index=frame.index)
            )
            figure.add_trace(
                go.Scatter(
                    name=title,
                    x=pd.to_datetime(frame["date"]),
                    y=values,
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=color, width=1.4),
                    showlegend=False,
                    hovertemplate=(
                        f"%{{x|%d/%m/%Y}}<br>{title}: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            figure.update_yaxes(title_text=unit, rangemode="tozero", row=row, col=col)
    figure.update_xaxes(tickformat="%d/%m", showgrid=False)
    figure.update_layout(
        title="Daily physiological load",
        height=760,
        margin=dict(l=72, r=32, t=90, b=55),
        hovermode="x unified",
    )
    return figure


def menstrual_phase_comparisons_figure(
    daily: pd.DataFrame,
    sessions: pd.DataFrame,
    metrics: pd.DataFrame,
    menstrual: pd.DataFrame,
) -> go.Figure:
    """Compare sleep and recovery measurements across Garmin cycle phases."""
    panels = (
        ("Sleep duration", "sleep_hours", "hours"),
        ("Garmin Sleep score", "sleep_score", "score"),
        ("Overnight HRV", "overnight_hrv_display", "ms"),
        ("Resting heart rate", "daily_resting_heart_rate", "bpm"),
        ("Sleep stress", "average_sleep_stress", ""),
        ("REM share", "score_rem_percentage", "%"),
    )
    figure = make_subplots(
        rows=2,
        cols=3,
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
        subplot_titles=[panel[0] for panel in panels],
    )
    if menstrual.empty:
        figure.add_annotation(
            text="No menstrual phase data in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        frame = menstrual[["date", "current_phase"]].copy()
        frame = frame.merge(
            daily[["date", "total_sleep_minutes"]], on="date", how="left"
        )
        frame["sleep_hours"] = frame["total_sleep_minutes"] / 60
        scored = sessions[
            (sessions["session_type"] == "NIGHT_SLEEP")
            & sessions["legacy_garmin_score_10"].notna()
        ].groupby("sleep_date")["legacy_garmin_score_10"].mean() * 10
        frame["sleep_score"] = frame["date"].map(scored)
        if not metrics.empty:
            metric_frame = metrics.copy()
            metric_frame["overnight_hrv_display"] = metric_frame[
                "hrv_last_night_average"
            ].fillna(metric_frame["overnight_hrv"])
            metric_columns = [
                "date",
                "overnight_hrv_display",
                "daily_resting_heart_rate",
                "average_sleep_stress",
                "score_rem_percentage",
            ]
            frame = frame.merge(metric_frame[metric_columns], on="date", how="left")
        phase_names = {1: "Menstrual", 2: "Follicular", 3: "Ovulation", 4: "Luteal"}
        phase_colors = {
            "Menstrual": "#E14E5C",
            "Follicular": "#2FD6A3",
            "Ovulation": "#FFB547",
            "Luteal": "#F29BDA",
        }
        frame["phase"] = frame["current_phase"].map(phase_names)
        for panel_index, (title, column, unit) in enumerate(panels):
            row, col = divmod(panel_index, 3)
            row += 1
            col += 1
            if column not in frame:
                continue
            for phase in phase_names.values():
                values = frame.loc[frame["phase"] == phase, column].dropna()
                if values.empty:
                    continue
                figure.add_trace(
                    go.Box(
                        name=phase,
                        x=[phase] * len(values),
                        y=values,
                        marker_color=phase_colors[phase],
                        fillcolor="rgba(0,0,0,0)",
                        line=dict(color=phase_colors[phase], width=1.3),
                        boxmean=True,
                        showlegend=panel_index == 0,
                        legendgroup=phase,
                        hovertemplate=(
                            f"{phase}<br>{title}: %{{y:.1f}} {unit}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            figure.update_yaxes(title_text=unit, row=row, col=col)
    figure.update_layout(
        title="Sleep and recovery by menstrual phase",
        height=690,
        margin=dict(l=62, r=28, t=92, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        boxmode="group",
    )
    figure.update_xaxes(showgrid=False)
    return figure


def awakenings_efficiency_figure(metrics: pd.DataFrame) -> go.Figure:
    """Show Garmin awakening measures and stage-derived sleep efficiency."""
    panels = (
        ("Awake time", "awake_minutes", "min", "#E14E5C"),
        ("Awake count", "awake_count", "count", "#FFB547"),
        ("Sleep efficiency", "sleep_efficiency", "%", "#4DB88A"),
        ("Restlessness rating", "restlessness_rating", "rating", "#5B32D6"),
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.17,
        horizontal_spacing=0.10,
        subplot_titles=[panel[0] for panel in panels],
    )
    if metrics.empty:
        figure.add_annotation(
            text="No Garmin awakening data in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        frame = metrics.sort_values("date").copy()
        asleep_minutes = frame[
            ["deep_sleep_minutes", "light_sleep_minutes", "rem_sleep_minutes"]
        ].fillna(0).sum(axis=1)
        measured_minutes = asleep_minutes + frame["awake_minutes"].fillna(0)
        frame["sleep_efficiency"] = (
            asleep_minutes / measured_minutes.where(measured_minutes > 0) * 100
        )
        rating_values = {"POOR": 1, "FAIR": 2, "GOOD": 3, "EXCELLENT": 4}
        frame["restlessness_rating"] = frame[
            "score_restlessness_qualifier"
        ].map(rating_values)
        for index, (title, column, unit, color) in enumerate(panels):
            row, col = divmod(index, 2)
            row += 1
            col += 1
            figure.add_trace(
                go.Scatter(
                    name=title,
                    x=pd.to_datetime(frame["date"]),
                    y=frame[column],
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=color, width=1.4),
                    showlegend=False,
                    hovertemplate=(
                        f"%{{x|%d/%m/%Y}}<br>{title}: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            if column == "restlessness_rating":
                figure.update_yaxes(
                    tickmode="array",
                    tickvals=[1, 2, 3, 4],
                    ticktext=["Poor", "Fair", "Good", "Excellent"],
                    range=[0.5, 4.5],
                    row=row,
                    col=col,
                )
            elif column == "sleep_efficiency":
                figure.update_yaxes(title_text=unit, range=[0, 100], row=row, col=col)
            else:
                figure.update_yaxes(title_text=unit, rangemode="tozero", row=row, col=col)
    figure.update_xaxes(tickformat="%d/%m", showgrid=False)
    figure.update_layout(
        title="Awakenings and sleep efficiency",
        height=610,
        margin=dict(l=74, r=32, t=90, b=55),
        hovermode="x unified",
    )
    return figure


def activity_effect_on_sleep_figure(
    activities: pd.DataFrame,
    sessions: pd.DataFrame,
    metrics: pd.DataFrame,
    predictor: str = "duration",
) -> go.Figure:
    """Relate each activity day's load to its next recorded night sleep."""
    panels = (
        ("Subsequent sleep duration", "sleep_hours", "hours", "#5B32D6"),
        ("Subsequent Sleep score", "sleep_score", "score", "#FF7AB6"),
        ("Subsequent overnight HRV", "overnight_hrv_display", "ms", "#4DB88A"),
        ("Subsequent sleep stress", "average_sleep_stress", "", "#E6B84F"),
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.17,
        horizontal_spacing=0.10,
        subplot_titles=[panel[0] for panel in panels],
    )
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    if activities.empty or nights.empty:
        figure.add_annotation(
            text="No activity-to-subsequent-sleep pairs in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        daily_activity = (
            activities.groupby("activity_date", as_index=False)
            .agg(
                activity_seconds=("duration_seconds", "sum"),
                last_activity_end=("activity_end", "max"),
                activity_types=(
                    "activity_type",
                    lambda values: ", ".join(
                        sorted({str(value).replace("_", " ").title() for value in values})
                    ),
                ),
            )
            .sort_values("activity_date")
        )
        metric_frame = metrics.copy()
        if not metric_frame.empty:
            metric_frame["overnight_hrv_display"] = metric_frame[
                "hrv_last_night_average"
            ].fillna(metric_frame["overnight_hrv"])
            metric_frame = metric_frame.set_index("date")
        pairs: list[dict[str, object]] = []
        for activity in daily_activity.itertuples(index=False):
            next_nights = nights[nights["sleep_start"] >= activity.last_activity_end]
            if next_nights.empty:
                continue
            night = next_nights.iloc[0]
            sleep_date = night["sleep_date"]
            metric = (
                metric_frame.loc[sleep_date]
                if not metric_frame.empty and sleep_date in metric_frame.index
                else None
            )
            if isinstance(metric, pd.DataFrame):
                metric = metric.iloc[0]
            pairs.append(
                {
                    "activity_date": activity.activity_date,
                    "activity_hours": activity.activity_seconds / 3600,
                    "activity_types": activity.activity_types,
                    "sleep_date": sleep_date,
                    "hours_to_sleep": (
                        night["sleep_start"] - activity.last_activity_end
                    ).total_seconds()
                    / 3600,
                    "sleep_hours": float(night["sleep_minutes"]) / 60,
                    "sleep_score": (
                        float(night["legacy_garmin_score_10"]) * 10
                        if pd.notna(night["legacy_garmin_score_10"])
                        else float("nan")
                    ),
                    "overnight_hrv_display": (
                        metric["overnight_hrv_display"]
                        if metric is not None
                        else float("nan")
                    ),
                    "average_sleep_stress": (
                        metric["average_sleep_stress"]
                        if metric is not None
                        else float("nan")
                    ),
                }
            )
        pair_frame = pd.DataFrame(pairs)
        if pair_frame.empty:
            figure.add_annotation(
                text="No activity-to-subsequent-sleep pairs in the selected date range",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
        else:
            predictor_column = (
                "hours_to_sleep" if predictor == "gap" else "activity_hours"
            )
            predictor_label = (
                "Time from activity end to sleep (hours)"
                if predictor == "gap"
                else "Activity duration (hours)"
            )
            customdata = pair_frame[
                [
                    "activity_date",
                    "activity_types",
                    "sleep_date",
                    "activity_hours",
                    "hours_to_sleep",
                ]
            ]
            for index, (title, column, unit, color) in enumerate(panels):
                row, col = divmod(index, 2)
                row += 1
                col += 1
                figure.add_trace(
                    go.Scatter(
                        name=title,
                        x=pair_frame[predictor_column],
                        y=pair_frame[column],
                        mode="markers",
                        marker=dict(color=color, size=7, opacity=0.78),
                        customdata=customdata,
                        showlegend=False,
                        hovertemplate=(
                            "Activity: %{customdata[1]}<br>"
                            "Activity date: %{customdata[0]}<br>"
                            "Following sleep: %{customdata[2]}<br>"
                            "Activity duration: %{customdata[3]:.2f} h<br>"
                            "Activity end → sleep: %{customdata[4]:.2f} h<br>"
                            f"{predictor_label}: %{{x:.2f}}<br>"
                            f"{title}: %{{y:.1f}} {unit}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                valid = pair_frame[[predictor_column, column]].dropna().sort_values(
                    predictor_column
                )
                if len(valid) >= 2 and valid[predictor_column].nunique() > 1:
                    x_values = valid[predictor_column]
                    y_values = valid[column]
                    centered = x_values - x_values.mean()
                    slope = float(
                        (centered * (y_values - y_values.mean())).sum()
                        / (centered**2).sum()
                    )
                    trend_x = [float(x_values.min()), float(x_values.max())]
                    intercept = float(y_values.mean() - slope * x_values.mean())
                    figure.add_trace(
                        go.Scatter(
                            name="Trend",
                            x=trend_x,
                            y=[intercept + slope * value for value in trend_x],
                            mode="lines",
                            line=dict(color=color, width=1.5, dash="dash"),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
                summary = _spearman_summary(
                    pair_frame[predictor_column], pair_frame[column]
                )
                if summary is not None:
                    correlation, count = summary
                    figure.layout.annotations[index].text = (
                        f"{title}<br><sup>Spearman ρ={correlation:+.2f} · n={count}</sup>"
                    )
                figure.update_xaxes(title_text=predictor_label, row=row, col=col)
                figure.update_yaxes(title_text=unit, row=row, col=col)
    figure.update_layout(
        title=(
            "Activity timing vs subsequent sleep"
            if predictor == "gap"
            else "Activity duration vs subsequent sleep"
        ),
        height=650,
        margin=dict(l=76, r=32, t=90, b=62),
    )
    return figure


def _activity_sleep_pairs(
    activities: pd.DataFrame, sessions: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP") & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    if activities.empty or nights.empty:
        return pd.DataFrame()
    grouped = (
        activities.groupby("activity_date", as_index=False)
        .agg(
            activity_hours=("duration_seconds", lambda values: values.sum() / 3600),
            last_activity_end=("activity_end", "max"),
            activity_type=(
                "activity_type",
                lambda values: (
                    str(next(iter(set(values)))).replace("_", " ").title()
                    if len(set(values)) == 1
                    else "Mixed activities"
                ),
            ),
            calories=("calories", "sum"),
            average_hr=("average_hr", "mean"),
            max_hr=("max_hr", "max"),
            high_zone_seconds=(
                "hr_zone_4_seconds",
                lambda values: values.fillna(0).sum(),
            ),
            zone_5_seconds=(
                "hr_zone_5_seconds",
                lambda values: values.fillna(0).sum(),
            ),
        )
    )
    metric_index = metrics.set_index("date") if not metrics.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for activity in grouped.itertuples(index=False):
        candidates = nights[nights["sleep_start"] >= activity.last_activity_end]
        if candidates.empty:
            continue
        night = candidates.iloc[0]
        sleep_date = night["sleep_date"]
        metric = (
            metric_index.loc[sleep_date]
            if not metric_index.empty and sleep_date in metric_index.index
            else None
        )
        if isinstance(metric, pd.DataFrame):
            metric = metric.iloc[0]
        rows.append(
            {
                **activity._asdict(),
                "sleep_date": sleep_date,
                "sleep_hours": float(night["sleep_minutes"]) / 60,
                "sleep_score": (
                    float(night["legacy_garmin_score_10"]) * 10
                    if pd.notna(night["legacy_garmin_score_10"])
                    else float("nan")
                ),
                "overnight_hrv": (
                    metric.get("hrv_last_night_average")
                    if metric is not None
                    else float("nan")
                ),
                "sleep_stress": (
                    metric.get("average_sleep_stress")
                    if metric is not None
                    else float("nan")
                ),
                "high_zone_minutes": (
                    float(activity.high_zone_seconds or 0)
                    + float(activity.zone_5_seconds or 0)
                )
                / 60,
            }
        )
    return pd.DataFrame(rows)


def activity_type_comparison_figure(
    activities: pd.DataFrame, sessions: pd.DataFrame, metrics: pd.DataFrame
) -> go.Figure:
    pairs = _activity_sleep_pairs(activities, sessions, metrics)
    panels = (
        ("Following sleep duration", "sleep_hours", "hours"),
        ("Following Sleep score", "sleep_score", "score"),
        ("Following overnight HRV", "overnight_hrv", "ms"),
        ("Following sleep stress", "sleep_stress", ""),
    )
    figure = make_subplots(rows=2, cols=2, subplot_titles=[p[0] for p in panels])
    if pairs.empty:
        figure.add_annotation(text="No activity/sleep pairs", showarrow=False)
    else:
        palette = ["#087CFF", "#5B32D6", "#E8793F", "#4DB88A", "#EE55C0"]
        types = sorted(pairs["activity_type"].unique())
        for panel_index, (title, column, unit) in enumerate(panels):
            row, col = divmod(panel_index, 2)
            row += 1
            col += 1
            for type_index, activity_type in enumerate(types):
                values = pairs.loc[pairs["activity_type"] == activity_type, column].dropna()
                if values.empty:
                    continue
                figure.add_trace(
                    go.Box(
                        name=activity_type,
                        x=[activity_type] * len(values),
                        y=values,
                        boxmean=True,
                        boxpoints="all",
                        jitter=0.25,
                        marker_color=palette[type_index % len(palette)],
                        fillcolor="rgba(0,0,0,0)",
                        showlegend=panel_index == 0,
                        legendgroup=activity_type,
                        hovertemplate=f"{activity_type}<br>{title}: %{{y:.1f}} {unit}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            figure.update_yaxes(title_text=unit, row=row, col=col)
    figure.update_layout(
        title="Subsequent sleep by activity type",
        height=680,
        margin=dict(l=70, r=30, t=95, b=100),
        legend=dict(orientation="h", y=1.04),
        boxmode="group",
    )
    return figure


def activity_intensity_effect_figure(
    activities: pd.DataFrame, sessions: pd.DataFrame, metrics: pd.DataFrame
) -> go.Figure:
    pairs = _activity_sleep_pairs(activities, sessions, metrics)
    panels = (
        ("Calories", "calories", "kcal"),
        ("Average activity HR", "average_hr", "bpm"),
        ("Maximum activity HR", "max_hr", "bpm"),
        ("Time in HR zones 4–5", "high_zone_minutes", "min"),
    )
    figure = make_subplots(rows=2, cols=2, subplot_titles=[p[0] for p in panels])
    for index, (title, column, unit) in enumerate(panels):
        row, col = divmod(index, 2)
        row += 1
        col += 1
        if pairs.empty:
            continue
        valid = pairs[[column, "sleep_score"]].dropna()
        figure.add_trace(
            go.Scatter(
                x=valid[column],
                y=valid["sleep_score"],
                mode="markers",
                marker=dict(color="#4DB88A", size=7, opacity=0.8),
                showlegend=False,
                hovertemplate=f"{title}: %{{x:.1f}} {unit}<br>Following score: %{{y:.0f}}/100<extra></extra>",
            ),
            row=row,
            col=col,
        )
        summary = _spearman_summary(valid[column], valid["sleep_score"])
        if summary:
            rho, count = summary
            figure.layout.annotations[index].text = (
                f"{title}<br><sup>Spearman ρ={rho:+.2f} · n={count}</sup>"
            )
        figure.update_xaxes(title_text=unit, row=row, col=col)
        figure.update_yaxes(title_text="Following Sleep score", range=[0, 100], row=row, col=col)
    figure.update_layout(
        title="Activity intensity and heart-rate load",
        height=650,
        margin=dict(l=76, r=30, t=95, b=65),
    )
    return figure


def intraday_recovery_timeline_figure(
    intraday: pd.DataFrame,
    activities: pd.DataFrame,
    sessions: pd.DataFrame,
    selected_date: date,
) -> go.Figure:
    frame = intraday[intraday["date"] == selected_date].sort_values("timestamp")
    figure = go.Figure()
    for name, column, color in (
        ("Heart rate", "heart_rate", "#E8793F"),
        ("Stress", "stress", "#E6B84F"),
        ("Body Battery", "body_battery", "#4DB88A"),
    ):
        samples = frame.loc[frame[column].notna(), ["timestamp", column]]
        figure.add_trace(
            go.Scatter(
                name=name,
                x=samples["timestamp"],
                y=samples[column],
                mode="lines",
                connectgaps=False,
                line=dict(color=color, width=1),
            )
        )
    for activity in activities[activities["activity_date"] == selected_date].itertuples(index=False):
        figure.add_vrect(x0=activity.activity_start, x1=activity.activity_end, fillcolor="#2FD6A3", opacity=0.18, line_width=0, annotation_text=activity.activity_name)
    day_start = pd.Timestamp(selected_date).tz_localize(TIMEZONE_NAME)
    day_end = day_start + pd.Timedelta(days=1)
    for night in sessions[(sessions["wake_time"] > day_start) & (sessions["sleep_start"] < day_end)].itertuples(index=False):
        figure.add_vrect(x0=max(night.sleep_start, day_start.to_pydatetime()), x1=min(night.wake_time, day_end.to_pydatetime()), fillcolor="#5B32D6", opacity=0.12, line_width=0)
    figure.update_layout(
        title=f"Intraday recovery · {selected_date:%d/%m/%Y}",
        height=500,
        yaxis=dict(title="Value", rangemode="tozero"),
        xaxis=dict(title="Time", tickformat="%H:%M", range=[day_start, day_end]),
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
    )
    return figure


def heart_rate_24_hour_heatmap(
    intraday: pd.DataFrame,
    activities: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> go.Figure:
    frame = intraday[intraday["heart_rate"].notna()].copy()
    figure = go.Figure()
    if not frame.empty:
        frame["minute_of_day"] = (
            frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
        )
        frame["minute_bin"] = (frame["minute_of_day"] // 2) * 2
        matrix = frame.pivot_table(
            index="minute_bin",
            columns="date",
            values="heart_rate",
            aggfunc="mean",
        ).reindex(index=range(0, 1440, 2))
        all_dates = pd.date_range(start_date, end_date, freq="D")
        matrix = matrix.reindex(columns=[value.date() for value in all_dates])
        observed = frame["heart_rate"].dropna()
        color_min = max(35.0, float(observed.quantile(0.01)))
        color_max = max(color_min + 1, float(observed.quantile(0.995)))
        figure.add_trace(
            go.Heatmap(
                x=all_dates,
                y=matrix.index / 60,
                z=matrix.to_numpy(),
                zmin=color_min,
                zmax=color_max,
                colorscale=[
                    [0.0, "#172B55"],
                    [0.25, "#087CFF"],
                    [0.5, "#2FD6A3"],
                    [0.7, "#E6B84F"],
                    [0.85, "#E8793F"],
                    [1.0, "#FF5A4D"],
                ],
                colorbar=dict(title="bpm", thickness=14),
                hovertemplate=(
                    "%{x|%d/%m/%Y}<br>%{y:.2f} hours<br>"
                    "Heart rate: %{z:.0f} bpm<extra></extra>"
                ),
                xgap=0.2,
                ygap=0,
                connectgaps=False,
            )
        )
        if not activities.empty:
            activity_frame = activities.copy()
            activity_frame["start_hour"] = activity_frame["activity_start"].map(
                lambda value: value.hour
                + value.minute / 60
                + value.second / 3600
            )
            activity_frame["duration_hours"] = (
                activity_frame["duration_seconds"] / 3600
            )
            activity_frame["plot_date"] = pd.to_datetime(
                activity_frame["activity_date"]
            ) + pd.Timedelta(hours=12)
            figure.add_trace(
                go.Bar(
                    name="Recorded activity",
                    x=activity_frame["plot_date"],
                    y=activity_frame["duration_hours"],
                    base=activity_frame["start_hour"],
                    width=0.72 * 24 * 60 * 60 * 1000,
                    marker=dict(color="#FF7AB6", opacity=1.0, line_width=0),
                    customdata=activity_frame[
                        ["activity_name", "activity_start", "activity_end"]
                    ],
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        "%{customdata[1]|%d/%m %H:%M}–"
                        "%{customdata[2]|%H:%M}<extra></extra>"
                    ),
                )
            )
    else:
        figure.add_annotation(
            text="No heart-rate samples in the selected date range",
            showarrow=False,
        )
    figure.update_layout(
        title="Heart rate across 24 hours",
        height=720,
        margin=dict(l=78, r=50, t=70, b=65),
        xaxis=dict(
            title="Date",
            type="date",
            range=[pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1)],
            showgrid=False,
            minallowed=pd.Timestamp(start_date),
            maxallowed=pd.Timestamp(end_date) + pd.Timedelta(days=1),
        ),
        yaxis=dict(
            title="Time of day",
            range=[0, 24],
            tickmode="array",
            tickvals=[0, 4, 8, 12, 16, 20, 24],
            ticktext=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        barmode="overlay",
        legend=dict(orientation="h", y=1.02),
    )
    return figure


def daily_calories_stacked_figure(metrics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    required = {"daily_total_calories", "daily_active_calories"}
    if metrics.empty or not required.issubset(metrics.columns):
        figure.add_annotation(
            text="No daily calorie data in the selected date range",
            showarrow=False,
        )
    else:
        frame = metrics[
            ["date", "daily_total_calories", "daily_active_calories"]
        ].copy()
        frame = frame[frame["daily_total_calories"].notna()].sort_values("date")
        frame["active_calories"] = frame["daily_active_calories"].fillna(0).clip(
            lower=0
        )
        frame["resting_calories"] = (
            frame["daily_total_calories"] - frame["active_calories"]
        ).clip(lower=0)
        figure.add_trace(
            go.Bar(
                name="Resting calories",
                x=frame["date"],
                y=frame["resting_calories"],
                marker=dict(color="#5B32D6", line_width=0),
                hovertemplate="%{x|%d/%m/%Y}<br>Resting: %{y:.0f} kcal<extra></extra>",
            )
        )
        figure.add_trace(
            go.Bar(
                name="Active calories",
                x=frame["date"],
                y=frame["active_calories"],
                marker=dict(color="#FF7AB6", line_width=0),
                hovertemplate="%{x|%d/%m/%Y}<br>Active: %{y:.0f} kcal<extra></extra>",
            )
        )
    figure.update_layout(
        title="Daily calories burned",
        height=470,
        barmode="stack",
        bargap=0.2,
        margin=dict(l=76, r=34, t=70, b=62),
        xaxis=dict(title="Date", type="date", showgrid=False),
        yaxis=dict(title="Calories burned (kcal)", rangemode="tozero"),
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
    )
    return figure


def activity_to_sleep_gap_figure(
    activities: pd.DataFrame,
    sessions: pd.DataFrame,
    exclude_daytime_activities: bool = False,
) -> go.Figure:
    figure = go.Figure()
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    rows: list[dict[str, object]] = []
    activity_frame = activities.copy()
    if exclude_daytime_activities and not activity_frame.empty:
        start_minutes = activity_frame["activity_start"].map(
            lambda value: value.hour * 60 + value.minute
        )
        end_minutes = activity_frame["activity_end"].map(
            lambda value: value.hour * 60 + value.minute
        )
        daytime = (start_minutes >= 8 * 60) & (end_minutes <= 16 * 60 + 30)
        activity_frame = activity_frame[~daytime]
    if not activity_frame.empty and not nights.empty:
        for activity in activity_frame.sort_values("activity_end").itertuples(index=False):
            following = nights[nights["sleep_start"] >= activity.activity_end]
            if following.empty:
                continue
            night = following.iloc[0]
            gap_hours = (
                night["sleep_start"] - activity.activity_end
            ).total_seconds() / 3600
            rows.append(
                {
                    "activity_end": activity.activity_end,
                    "activity_name": activity.activity_name,
                    "activity_type": str(activity.activity_type)
                    .replace("_", " ")
                    .title(),
                    "sleep_start": night["sleep_start"],
                    "gap_hours": gap_hours,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        figure.add_annotation(
            text="No activity-to-sleep pairs in the selected date range",
            showarrow=False,
        )
    else:
        palette = [
            "#FF7AB6",
            "#087CFF",
            "#5B32D6",
            "#2FD6A3",
            "#E8793F",
            "#E6B84F",
            "#EE55C0",
        ]
        for index, activity_type in enumerate(sorted(frame["activity_type"].unique())):
            subset = frame[frame["activity_type"] == activity_type]
            figure.add_trace(
                go.Scatter(
                    name=activity_type,
                    x=subset["activity_end"],
                    y=subset["gap_hours"],
                    mode="markers",
                    marker=dict(
                        color=palette[index % len(palette)], size=8, opacity=0.9
                    ),
                    customdata=subset[
                        ["activity_name", "activity_end", "sleep_start"]
                    ],
                    hovertemplate=(
                        "%{customdata[0]}<br>Ended: %{customdata[1]|%d/%m %H:%M}"
                        "<br>Sleep began: %{customdata[2]|%d/%m %H:%M}"
                        "<br>Gap: %{y:.1f} hours<extra></extra>"
                    ),
                )
            )
        median_gap = float(frame["gap_hours"].median())
        figure.add_hline(
            y=median_gap,
            line=dict(color="#E6B84F", width=1.5, dash="dot"),
            annotation_text=f"Median {median_gap:.1f} h",
            annotation_position="top left",
        )
    figure.update_layout(
        title=(
            "Time from activity ending to sleep · excluding 08:00–16:30 activities"
            if exclude_daytime_activities
            else "Time from activity ending to sleep"
        ),
        height=470,
        margin=dict(l=76, r=34, t=75, b=62),
        xaxis=dict(title="Activity end date", type="date", showgrid=False),
        yaxis=dict(title="Hours until sleep begins", rangemode="tozero"),
        legend=dict(orientation="h", y=1.02),
    )
    return figure


def intensity_minutes_vs_subsequent_sleep_figure(
    metrics: pd.DataFrame, sessions: pd.DataFrame
) -> go.Figure:
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    hrv_by_date = pd.Series(dtype=float)
    if not metrics.empty:
        hrv_values = metrics["hrv_last_night_average"].fillna(
            metrics["overnight_hrv"]
        )
        hrv_by_date = pd.Series(
            hrv_values.values, index=metrics["date"]
        ).groupby(level=0).first()
    stress_by_date = (
        metrics.groupby("date")["average_sleep_stress"].first()
        if not metrics.empty
        else pd.Series(dtype=float)
    )
    rows: list[dict[str, object]] = []
    if (
        not metrics.empty
        and "daily_intensity_minutes" in metrics
        and not nights.empty
    ):
        for day in metrics[
            metrics["daily_intensity_minutes"].notna()
        ].itertuples(index=False):
            cutoff = pd.Timestamp(day.date).tz_localize(TIMEZONE_NAME) + pd.Timedelta(
                hours=12
            )
            following = nights[nights["sleep_start"] >= cutoff]
            if following.empty:
                continue
            night = following.iloc[0]
            rows.append(
                {
                    "date": day.date,
                    "intensity_minutes": float(day.daily_intensity_minutes),
                    "sleep_date": night["sleep_date"],
                    "sleep_hours": float(night["sleep_minutes"]) / 60,
                    "sleep_score": (
                        float(night["legacy_garmin_score_10"]) * 10
                        if pd.notna(night["legacy_garmin_score_10"])
                        else float("nan")
                    ),
                    "overnight_hrv": hrv_by_date.get(
                        night["sleep_date"], float("nan")
                    ),
                    "sleep_stress": stress_by_date.get(
                        night["sleep_date"], float("nan")
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    figure = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[
            "Following sleep duration",
            "Following Garmin Sleep score",
            "Following overnight HRV",
            "Following average sleep stress",
        ],
    )
    panels = (
        ("sleep_hours", "Hours asleep", None),
        ("sleep_score", "Garmin Sleep score", [0, 100]),
        ("overnight_hrv", "Overnight HRV (ms)", None),
        ("sleep_stress", "Average sleep stress", None),
    )
    for column_index, (outcome, y_title, y_range) in enumerate(panels, 1):
        if frame.empty:
            continue
        valid = frame[["intensity_minutes", outcome, "date", "sleep_date"]].dropna()
        figure.add_trace(
            go.Scatter(
                x=valid["intensity_minutes"],
                y=valid[outcome],
                mode="markers",
                marker=dict(color="#2FD6A3", size=8, opacity=0.82),
                customdata=valid[["date", "sleep_date"]],
                hovertemplate=(
                    "Activity day: %{customdata[0]}<br>"
                    "Intensity minutes: %{x:.0f}<br>"
                    "Following sleep date: %{customdata[1]}<br>"
                    f"{y_title}: %{{y:.1f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column_index,
        )
        summary = _spearman_summary(valid["intensity_minutes"], valid[outcome])
        if summary:
            rho, count = summary
            figure.layout.annotations[column_index - 1].text += (
                f"<br><sup>Spearman ρ={rho:+.2f} · n={count}</sup>"
            )
        figure.update_xaxes(
            title_text="Garmin intensity minutes that day",
            rangemode="tozero",
            row=1,
            col=column_index,
        )
        figure.update_yaxes(
            title_text=y_title,
            range=y_range,
            rangemode=None if y_range else "tozero",
            row=1,
            col=column_index,
        )
    if frame.empty:
        figure.add_annotation(
            text="No intensity-minute and subsequent-sleep pairs",
            showarrow=False,
        )
    figure.update_layout(
        title="Daily intensity minutes vs subsequent sleep",
        height=500,
        margin=dict(l=76, r=34, t=90, b=68),
    )
    return figure


def steps_floors_vs_subsequent_sleep_figure(
    metrics: pd.DataFrame, sessions: pd.DataFrame
) -> go.Figure:
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    hrv_by_date = pd.Series(dtype=float)
    if not metrics.empty:
        hrv_values = metrics["hrv_last_night_average"].fillna(
            metrics["overnight_hrv"]
        )
        hrv_by_date = pd.Series(
            hrv_values.values, index=metrics["date"]
        ).groupby(level=0).first()
    stress_by_date = (
        metrics.groupby("date")["average_sleep_stress"].first()
        if not metrics.empty
        else pd.Series(dtype=float)
    )
    rows: list[dict[str, object]] = []
    required = {"daily_steps", "daily_floors_ascended"}
    if not metrics.empty and required.issubset(metrics.columns) and not nights.empty:
        usable_days = metrics[
            metrics[["daily_steps", "daily_floors_ascended"]]
            .notna()
            .any(axis=1)
        ]
        for day in usable_days.itertuples(index=False):
            cutoff = pd.Timestamp(day.date).tz_localize(TIMEZONE_NAME) + pd.Timedelta(
                hours=12
            )
            following = nights[nights["sleep_start"] >= cutoff]
            if following.empty:
                continue
            night = following.iloc[0]
            rows.append(
                {
                    "date": day.date,
                    "steps": day.daily_steps,
                    "floors": day.daily_floors_ascended,
                    "sleep_date": night["sleep_date"],
                    "sleep_hours": float(night["sleep_minutes"]) / 60,
                    "sleep_score": (
                        float(night["legacy_garmin_score_10"]) * 10
                        if pd.notna(night["legacy_garmin_score_10"])
                        else float("nan")
                    ),
                    "overnight_hrv": hrv_by_date.get(
                        night["sleep_date"], float("nan")
                    ),
                    "sleep_stress": stress_by_date.get(
                        night["sleep_date"], float("nan")
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    panels = (
        ("steps", "sleep_hours", "Daily steps", "Following sleep duration", "#087CFF"),
        ("steps", "sleep_score", "Daily steps", "Following Garmin Sleep score", "#087CFF"),
        ("steps", "overnight_hrv", "Daily steps", "Following overnight HRV", "#087CFF"),
        ("steps", "sleep_stress", "Daily steps", "Following average sleep stress", "#087CFF"),
        ("floors", "sleep_hours", "Floors climbed", "Following sleep duration", "#E8793F"),
        ("floors", "sleep_score", "Floors climbed", "Following Garmin Sleep score", "#E8793F"),
        ("floors", "overnight_hrv", "Floors climbed", "Following overnight HRV", "#E8793F"),
        ("floors", "sleep_stress", "Floors climbed", "Following average sleep stress", "#E8793F"),
    )
    figure = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[panel[3] for panel in panels],
        vertical_spacing=0.16,
    )
    for index, (predictor, outcome, x_title, y_title, color) in enumerate(panels):
        row, column = divmod(index, 4)
        row += 1
        column += 1
        if frame.empty:
            continue
        valid = frame[[predictor, outcome, "date", "sleep_date"]].dropna()
        figure.add_trace(
            go.Scatter(
                x=valid[predictor],
                y=valid[outcome],
                mode="markers",
                marker=dict(color=color, size=7, opacity=0.82),
                customdata=valid[["date", "sleep_date"]],
                hovertemplate=(
                    "Activity day: %{customdata[0]}<br>"
                    f"{x_title}: %{{x:.1f}}<br>"
                    "Following sleep date: %{customdata[1]}<br>"
                    f"{y_title}: %{{y:.1f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        summary = _spearman_summary(valid[predictor], valid[outcome])
        if summary:
            rho, count = summary
            figure.layout.annotations[index].text += (
                f"<br><sup>{x_title} · Spearman ρ={rho:+.2f} · n={count}</sup>"
            )
        figure.update_xaxes(title_text=x_title, rangemode="tozero", row=row, col=column)
        figure.update_yaxes(
            title_text=(
                "Hours asleep"
                if outcome == "sleep_hours"
                else "Sleep score"
                if outcome == "sleep_score"
                else "Overnight HRV (ms)"
                if outcome == "overnight_hrv"
                else "Average sleep stress"
            ),
            range=([0, 100] if outcome == "sleep_score" else None),
            rangemode=(
                "tozero"
                if outcome in {"overnight_hrv", "sleep_stress"}
                else None
            ),
            row=row,
            col=column,
        )
    if frame.empty:
        figure.add_annotation(
            text="No steps/floors and subsequent-sleep pairs",
            showarrow=False,
        )
    figure.update_layout(
        title="Steps and floors climbed vs subsequent sleep",
        height=720,
        margin=dict(l=76, r=34, t=95, b=70),
    )
    return figure


def multifactor_relationships_figure(
    daily: pd.DataFrame,
    sessions: pd.DataFrame,
    activities: pd.DataFrame,
    metrics: pd.DataFrame,
    menstrual: pd.DataFrame,
) -> go.Figure:
    frame = daily[["date", "total_sleep_minutes"]].sort_values("date").copy()
    frame["sleep_hours"] = frame["total_sleep_minutes"] / 60
    prior_deficit = (7.5 - frame["sleep_hours"]).shift(1)
    frame["sleep_debt_3"] = prior_deficit.rolling(3, min_periods=1).sum()
    frame["sleep_debt_7"] = prior_deficit.rolling(7, min_periods=1).sum()
    frame["weekend"] = pd.to_datetime(frame["date"]).dt.dayofweek.isin([5, 6]).astype(float)
    scores = sessions[(sessions["session_type"] == "NIGHT_SLEEP") & sessions["legacy_garmin_score_10"].notna()].groupby("sleep_date")["legacy_garmin_score_10"].mean() * 10
    frame["sleep_score"] = frame["date"].map(scores)
    activity_hours = activities.groupby("activity_date")["duration_seconds"].sum() / 3600 if not activities.empty else pd.Series(dtype=float)
    frame["previous_activity"] = frame["date"].map(lambda value: activity_hours.get(value - timedelta(days=1), 0.0))
    if not menstrual.empty:
        frame = frame.merge(menstrual[["date", "current_phase"]], on="date", how="left")
    else:
        frame["current_phase"] = float("nan")
    if not metrics.empty:
        metric = metrics.copy()
        metric["hrv"] = metric["hrv_last_night_average"].fillna(metric["overnight_hrv"])
        frame = frame.merge(
            metric[["date", "hrv", "average_sleep_stress"]],
            on="date",
            how="left",
        )
        previous_metric_columns = {
            "daily_steps": "previous_steps",
            "daily_floors_ascended": "previous_floors",
            "daily_intensity_minutes": "previous_intensity",
            "daily_active_calories": "previous_active_calories",
            "daily_average_stress": "previous_daily_stress",
        }
        metric_by_date = metric.groupby("date").first()
        for source_column, target_column in previous_metric_columns.items():
            if source_column in metric_by_date:
                frame[target_column] = frame["date"].map(
                    lambda value, column=source_column: metric_by_date[column].get(
                        value - timedelta(days=1), float("nan")
                    )
                )
            else:
                frame[target_column] = float("nan")
    else:
        frame["hrv"] = float("nan")
        frame["average_sleep_stress"] = float("nan")
        for column in (
            "previous_steps",
            "previous_floors",
            "previous_intensity",
            "previous_active_calories",
            "previous_daily_stress",
        ):
            frame[column] = float("nan")
    predictors = [
        "sleep_hours",
        "sleep_debt_3",
        "sleep_debt_7",
        "previous_activity",
        "previous_steps",
        "previous_floors",
        "previous_intensity",
        "previous_active_calories",
        "previous_daily_stress",
        "weekend",
    ]
    phase_dummies = pd.get_dummies(frame["current_phase"], prefix="phase", dtype=float)
    design = pd.concat([frame[predictors], phase_dummies], axis=1)
    labels = {
        "sleep_hours": "Sleep duration",
        "sleep_debt_3": "3-day sleep debt",
        "sleep_debt_7": "7-day sleep debt",
        "previous_activity": "Previous-day activity",
        "previous_steps": "Previous-day steps",
        "previous_floors": "Previous-day floors climbed",
        "previous_intensity": "Previous-day intensity minutes",
        "previous_active_calories": "Previous-day active calories",
        "previous_daily_stress": "Previous-day stress",
        "weekend": "Weekend",
    }
    phase_labels = {
        1: "Menstrual phase",
        2: "Follicular phase",
        3: "Ovulation phase",
        4: "Luteal phase",
    }
    for column in phase_dummies:
        try:
            phase_number = int(float(column.removeprefix("phase_")))
        except ValueError:
            phase_number = 0
        labels[column] = phase_labels.get(phase_number, column)
    outcomes = (
        ("sleep_score", "Sleep score", 1, 1),
        ("hrv", "Overnight HRV", 1, 2),
        ("average_sleep_stress", "Average sleep stress", 2, 1),
        ("sleep_hours", "Sleep duration", 2, 2),
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[item[1] for item in outcomes],
        vertical_spacing=0.13,
    )
    for outcome, _title, row_index, column_index in outcomes:
        outcome_predictors = [
            column
            for column in design
            if not (outcome == "sleep_hours" and column == "sleep_hours")
        ]
        combined = pd.concat(
            [design[outcome_predictors], frame[[outcome]]], axis=1
        ).dropna()
        usable = [
            column
            for column in outcome_predictors
            if combined[column].nunique() > 1
        ]
        if len(combined) < len(usable) + 3 or not usable:
            continue
        x = combined[usable].astype(float)
        x = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
        y = combined[outcome].astype(float)
        y = (y - y.mean()) / (y.std(ddof=0) or 1)
        coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1:]
        order = np.argsort(np.abs(coefficients))
        figure.add_trace(go.Bar(x=coefficients[order], y=[labels[c] for c in np.array(usable)[order]], orientation="h", marker_color=["#4DB88A" if value >= 0 else "#E14E5C" for value in coefficients[order]], showlegend=False), row=row_index, col=column_index)
        figure.update_xaxes(title_text="Standardized coefficient", zeroline=True, row=row_index, col=column_index)
    figure.update_layout(title="Multifactor relationships", height=980, margin=dict(l=180, r=30, t=90, b=60))
    return figure


def daily_duration_figure(
    daily: pd.DataFrame,
    sessions: pd.DataFrame,
    garmin_metrics: pd.DataFrame | None = None,
    menstrual: pd.DataFrame | None = None,
    sleep_stages: pd.DataFrame | None = None,
    activities: pd.DataFrame | None = None,
    show_sleep_stages: bool = False,
    view_start: date | None = None,
    view_end: date | None = None,
) -> go.Figure:
    """Daily sleep totals; with a view, opens on it and pans across all ``daily``."""
    frame = daily.copy()
    frame["date_text"] = frame["date"].astype(str)
    frame["date_plot"] = pd.to_datetime(frame["date"]) + pd.Timedelta(hours=12)
    frame["night_hours"] = frame["night_minutes"] / 60
    frame["nap_hours"] = frame["nap_minutes"] / 60
    frame["total_hours"] = frame["total_sleep_minutes"] / 60
    frame["average_7"] = frame["total_hours"].rolling(7, min_periods=1).mean()
    frame["average_28"] = frame["total_hours"].rolling(28, min_periods=1).mean()
    scored_nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & sessions["legacy_garmin_score_10"].notna()
    ]
    daily_scores = scored_nights.groupby("sleep_date")[
        "legacy_garmin_score_10"
    ].mean()
    frame["sleep_score_100"] = frame["date"].map(daily_scores) * 10
    frame["sleep_score_7_day"] = frame["sleep_score_100"].rolling(
        7, min_periods=1
    ).mean()
    frame["overnight_hrv"] = float("nan")
    frame["weekly_hrv"] = float("nan")
    frame["cycle_day"] = float("nan")
    frame["cycle_phase"] = float("nan")
    frame["cycle_start_date"] = None
    if garmin_metrics is not None and not garmin_metrics.empty:
        metrics = garmin_metrics.copy()
        metrics["overnight_hrv_display"] = metrics[
            "hrv_last_night_average"
        ].fillna(metrics["overnight_hrv"])
        daily_hrv = metrics.groupby("date")["overnight_hrv_display"].mean()
        frame["overnight_hrv"] = frame["date"].map(daily_hrv)
        weekly_hrv = metrics.groupby("date")["hrv_weekly_average"].mean()
        frame["weekly_hrv"] = frame["date"].map(weekly_hrv)
    if menstrual is not None and not menstrual.empty:
        menstrual_by_date = menstrual.set_index("date")
        frame["cycle_day"] = frame["date"].map(menstrual_by_date["cycle_day"])
        frame["cycle_phase"] = frame["date"].map(
            menstrual_by_date["current_phase"]
        )
        frame["cycle_start_date"] = frame["date"].map(
            menstrual_by_date["cycle_start_date"]
        )
    phase_names = {
        1: "Menstrual",
        2: "Follicular",
        3: "Ovulation",
        4: "Luteal",
    }
    frame["cycle_phase_name"] = frame["cycle_phase"].map(phase_names)
    zero = frame[frame["zero_sleep"]]

    figure = go.Figure()
    if show_sleep_stages:
        stage_hours = pd.DataFrame(index=pd.Index(frame["date"], name="sleep_date"))
        if sleep_stages is not None and not sleep_stages.empty:
            stages = sleep_stages.copy()
            stages["stage_hours"] = [
                (interval_end - interval_start).total_seconds() / 3600
                for interval_start, interval_end in zip(
                    stages["interval_start"], stages["interval_end"]
                )
            ]
            grouped_stages = stages.groupby(["sleep_date", "stage"])[
                "stage_hours"
            ].sum().unstack(fill_value=0)
            stage_hours = stage_hours.join(grouped_stages, how="left").fillna(0)
        for stage_name in ("Deep", "Light", "REM"):
            if stage_name not in stage_hours:
                stage_hours[stage_name] = 0.0
        frame["deep_hours"] = frame["date"].map(stage_hours["Deep"]).fillna(0)
        frame["light_hours"] = frame["date"].map(stage_hours["Light"]).fillna(0)
        frame["rem_hours"] = frame["date"].map(stage_hours["REM"]).fillna(0)
        measured_hours = frame[["deep_hours", "light_hours", "rem_hours"]].sum(axis=1)
        frame["unavailable_stage_hours"] = (
            frame["night_hours"] - measured_hours
        ).clip(lower=0)
        duration_stage_traces = (
            ("Deep", "deep_hours", "#331B8C", None),
            ("REM", "rem_hours", "#EE55C0", None),
            ("Light", "light_hours", "#5B32D6", None),
            (
                "Unavailable / unmeasurable",
                "unavailable_stage_hours",
                "#5B32D6",
                dict(
                    shape="/",
                    bgcolor="#5B32D6",
                    fgcolor="#331B8C",
                    fillmode="replace",
                    size=6,
                    solidity=0.33,
                ),
            ),
        )
        for trace_name, column, color, pattern in duration_stage_traces:
            marker = dict(color=color, line=dict(width=0))
            if pattern is not None:
                marker["pattern"] = pattern
            figure.add_trace(
                go.Bar(
                    name=trace_name,
                    x=frame["date_plot"],
                    y=frame[column],
                    marker=marker,
                    hovertemplate=(
                        f"%{{x}}<br>{trace_name}: %{{y:.2f}} h<extra></extra>"
                    ),
                )
            )
    else:
        figure.add_trace(
            go.Bar(
                name="Night sleep",
                x=frame["date_plot"],
                y=frame["night_hours"],
                marker_color=NIGHT_BAR_COLOR,
                hovertemplate="%{x}<br>Night sleep: %{y:.2f} h<extra></extra>",
            )
        )
    night_interruptions = sessions[
        sessions["session_type"] == "NIGHT_SLEEP"
    ].groupby("sleep_date")["interruption_minutes"].sum()
    frame["awake_hours"] = (
        frame["date"].map(night_interruptions).fillna(0) / 60
    )
    figure.add_trace(
        go.Bar(
            name="Awake / interruptions",
            x=frame["date_plot"],
            y=frame["awake_hours"],
            marker=dict(color="#9CA3AF", line=dict(width=0)),
            hovertemplate=(
                "%{x}<br>Awake / interruptions: %{y:.2f} h<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            name="Naps",
            x=frame["date_plot"],
            y=frame["nap_hours"],
            marker_color=NAP_BAR_COLOR,
            hovertemplate="%{x}<br>Naps: %{y:.2f} h<extra></extra>",
        )
    )
    if activities is not None and not activities.empty:
        activity_hours = activities.groupby("activity_date")[
            "duration_seconds"
        ].sum() / 3600
        frame["activity_hours"] = frame["date"].map(activity_hours).fillna(0)
        figure.add_trace(
            go.Bar(
                name="Garmin activity duration",
                x=frame["date_plot"],
                y=frame["activity_hours"],
                base=0,
                visible="legendonly",
                opacity=0.72,
                marker=dict(color="#4DB88A", line=dict(width=0)),
                hovertemplate=(
                    "%{x}<br>Garmin activity: %{y:.2f} h<extra></extra>"
                ),
            )
        )
    figure.add_trace(
        go.Scatter(
            name="Nightly average HRV",
            x=frame["date_plot"],
            y=frame["overnight_hrv"],
            yaxis="y3",
            mode="lines",
            visible="legendonly",
            connectgaps=False,
            line=dict(color=HRV_GREEN, width=1),
            hovertemplate="%{x}<br>Nightly average HRV: %{y:.0f} ms<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="7-day average HRV",
            x=frame["date_plot"],
            y=frame["weekly_hrv"],
            yaxis="y3",
            mode="lines",
            connectgaps=False,
            line=dict(color=HRV_WEEKLY_GREEN, width=1, dash="dot"),
            hovertemplate="%{x}<br>7-day average HRV: %{y:.0f} ms<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="7-day average",
            x=frame["date_plot"],
            y=frame["average_7"],
            mode="lines",
            visible="legendonly",
            line=dict(color=AVERAGE_7_COLOR, width=1, dash="dot"),
            hovertemplate="%{x}<br>7-day average: %{y:.2f} h<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="28-day average",
            x=frame["date_plot"],
            y=frame["average_28"],
            mode="lines",
            visible="legendonly",
            line=dict(color=AVERAGE_28_COLOR, width=1, dash="dot"),
            hovertemplate="%{x}<br>28-day average: %{y:.2f} h<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Sleep score (0–100)",
            x=frame["date_plot"],
            y=frame["sleep_score_100"],
            yaxis="y2",
            mode="lines",
            visible="legendonly",
            connectgaps=False,
            line=dict(color=SLEEP_SCORE_PINK, width=1),
            hovertemplate="%{x}<br>Sleep score: %{y:.0f}/100<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="7-day average Sleep score",
            x=frame["date_plot"],
            y=frame["sleep_score_7_day"],
            yaxis="y2",
            mode="lines",
            visible="legendonly",
            connectgaps=False,
            line=dict(color=SLEEP_SCORE_PINK, width=1, dash="dot"),
            hovertemplate=(
                "%{x}<br>7-day average Sleep score: %{y:.1f}/100<extra></extra>"
            ),
        )
    )
    if frame["cycle_phase"].notna().any():
        phase_customdata = [
            [
                row.cycle_phase_name,
                row.cycle_day,
                row.cycle_start_date,
            ]
            for row in frame.itertuples(index=False)
        ]
        figure.add_trace(
            go.Heatmap(
                name="Menstrual phase",
                x=frame["date_plot"],
                y=["Menstrual phase"],
                z=[frame["cycle_phase"].tolist()],
                yaxis="y4",
                showscale=False,
                # Invisible hover target; the visible strip is drawn as shapes.
                opacity=0,
                customdata=[phase_customdata],
                hovertemplate=(
                    "%{x}<br><b>%{customdata[0]}</b><br>"
                    "Cycle day: %{customdata[1]:.0f}<br>"
                    "Cycle began: %{customdata[2]}<extra></extra>"
                ),
            )
        )
        _add_menstrual_strip(figure, frame["date"], frame["cycle_phase"])

        for cycle_start in frame.loc[frame["cycle_day"] == 1, "date"]:
            cycle_boundary = pd.Timestamp(cycle_start)
            figure.add_shape(
                type="line",
                x0=cycle_boundary,
                x1=cycle_boundary,
                xref="x",
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="rgba(255,255,255,0.20)", width=1),
                layer="below",
            )
    if not zero.empty:
        figure.add_trace(
            go.Bar(
                name="Zero sleep",
                x=zero["date_plot"],
                y=[0.08] * len(zero),
                marker=dict(color=ZERO_COLOR, line=dict(width=0)),
                customdata=zero[["source_state"]],
                hovertemplate="%{x}<br>Zero sleep<br>%{customdata[0]}<extra></extra>",
            )
        )
    _add_no_data_markers(figure, frame)

    figure.update_layout(
        title="Daily sleep duration",
        barmode="stack",
        height=570,
        margin=dict(**DATE_CHART_SIDE_MARGINS, t=64, b=80),
        xaxis=dict(
            title="Sleep date (wake date)",
            type="date",
            tickformat="%d/%m",
            # Never rotate: rotated labels would need more bottom margin.
            tickangle=0,
            # Draw the date axis under the menstrual strip so the axis title
            # never overlaps it, whatever the chart height.
            anchor="y4",
        ),
        yaxis=dict(title="Hours asleep", domain=[0.11, 1.0]),
        yaxis2=dict(
            title="Sleep score (0–100)",
            overlaying="y",
            side="right",
            range=[0, 100],
            dtick=20,
            showgrid=False,
            zeroline=False,
        ),
        yaxis3=dict(
            title="HRV (ms)",
            overlaying="y",
            side="right",
            anchor="free",
            position=1.0,
            autoshift=True,
            rangemode="tozero",
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=HRV_GREEN),
            title_font=dict(color=HRV_GREEN),
        ),
        yaxis4=dict(
            domain=[0.0, 0.02],
            anchor="x",
            type="category",
            categoryorder="array",
            categoryarray=["Menstrual phase"],
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#cbd5e1", size=11),
        ),
        hovermode="x unified",
    )
    # Fixed range and legend space give the plot a known hour scale, which the
    # 24-hour stage chart copies.
    figure.update_layout(
        yaxis_range=[0, max(_stacked_bar_top(figure) * 1.05, 1.0)],
        meta=DATE_CHART_RIGHT_LEGEND,
    )
    if not frame.empty:
        _pannable_date_view(
            figure, frame["date"].min(), frame["date"].max(), view_start, view_end
        )
    return figure


def score_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions.copy()
    frame = frame[
        (frame["session_type"] == "NIGHT_SLEEP")
        & frame["legacy_garmin_score_10"].notna()
    ].sort_values("wake_time")
    frame["garmin_score_100"] = frame["legacy_garmin_score_10"] * 10
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            name="Garmin Sleep score",
            x=frame["wake_time"],
            y=frame["garmin_score_100"],
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.35)", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for aid in frame["aid_name"].drop_duplicates():
        subset = frame[frame["aid_name"] == aid]
        color = _aid_color(aid)
        figure.add_trace(
            go.Scatter(
                name=str(aid),
                x=subset["wake_time"],
                y=subset["garmin_score_100"],
                mode="markers",
                marker=dict(color=color, symbol="circle", size=9),
                customdata=subset[["sleep_minutes"]],
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M}<br>Garmin Sleep score: %{y}/100<br>"
                    "Aid: %{fullData.name}<br>Sleep: %{customdata[0]:.0f} min"
                    "<extra></extra>"
                ),
            )
        )
    xaxis: dict[str, object] = {"title": "Wake time"}
    if not frame.empty:
        x_start, x_end = _date_axis_limits(frame["wake_time"])
        xaxis.update(
            range=[x_start, x_end],
            minallowed=x_start,
            maxallowed=x_end,
        )
    figure.update_layout(
        title="Garmin Sleep score (night sleep only)",
        height=460,
        margin=dict(l=60, r=24, t=64, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=xaxis,
        yaxis=dict(
            title="Garmin Sleep score (0–100)",
            range=[-2.5, 102.5],
            minallowed=-2.5,
            maxallowed=102.5,
            dtick=10,
        ),
    )
    return figure


def credit_debit_figure(
    daily: pd.DataFrame, target_sleep_hours: float = TARGET_SLEEP_HOURS
) -> go.Figure:
    """Reproduce Sleepmeter's cumulative credit/debit view.

    A target-duration day is neutral. Daily balance is scaled to a 24-hour
    credit/debit clock, so a zero-sleep day contributes -24:00.
    """
    frame = daily.sort_values("date").copy()
    frame["sleep_hours"] = frame["total_sleep_minutes"] / 60
    frame = frame[frame["sleep_hours"].notna()].copy()
    if frame.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No completed sleep data in the selected date range",
            showarrow=False,
        )
        figure.update_layout(title="Sleep credit/debit", height=540)
        return figure
    frame["daily_balance"] = (
        frame["sleep_hours"] / target_sleep_hours - 1
    ) * 24
    frame["cumulative_balance"] = frame["daily_balance"].cumsum()
    frame["trend"] = _linear_trend(frame["cumulative_balance"])
    x_start, x_end = _date_axis_limits(frame["date"])

    mean_balance = float(frame["cumulative_balance"].mean())
    balance_std_value = frame["cumulative_balance"].std(ddof=1)
    balance_std = float(balance_std_value) if pd.notna(balance_std_value) else 0.0
    band_low = mean_balance - balance_std
    band_high = mean_balance + balance_std

    figure = go.Figure()
    figure.add_hrect(
        y0=band_low,
        y1=band_high,
        fillcolor=CREDIT_COLOR,
        opacity=0.14,
        line_width=0,
        layer="below",
    )
    figure.add_hline(y=0, line=dict(color=STATS_RED, width=2))
    figure.add_hline(
        y=mean_balance,
        line=dict(color=CREDIT_COLOR, width=1.5, dash="dash"),
    )
    figure.add_trace(
        go.Scatter(
            name="Credit/debit",
            x=frame["date"],
            y=frame["cumulative_balance"],
            mode="lines",
            line=dict(color=CREDIT_COLOR, width=1.8),
            customdata=frame[["sleep_hours", "daily_balance"]],
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Sleep: %{customdata[0]:.2f} h<br>"
                "Daily change: %{customdata[1]:+.2f} h<br>"
                "Cumulative: %{y:+.2f} h<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Trend",
            x=frame["date"],
            y=frame["trend"],
            mode="lines",
            line=dict(color=CREDIT_COLOR, width=2.2),
            hoverinfo="skip",
        )
    )

    y_low = min(float(frame["cumulative_balance"].min()), band_low, 0.0)
    y_high = max(float(frame["cumulative_balance"].max()), band_high, 0.0)
    padding = max((y_high - y_low) * 0.04, 4.0)
    y_low -= padding
    y_high += padding
    tick_step = _nice_tick_step(y_high - y_low)
    tick_start = math.ceil(y_low / tick_step) * tick_step
    tick_values: list[float] = []
    value = tick_start
    while value <= y_high + tick_step * 0.01:
        tick_values.append(value)
        value += tick_step

    figure.update_layout(
        title="Sleep credit/debit",
        showlegend=False,
        height=540,
        margin=dict(l=90, r=24, t=64, b=64),
        hovermode="x",
        xaxis=dict(
            title="Date",
            type="date",
            range=[x_start, x_end],
            minallowed=x_start,
            maxallowed=x_end,
            tickformat="%d/%m",
            gridcolor="#303030",
        ),
        yaxis=dict(
            title="Credit/debit (+/− HH:MM)",
            range=[y_low, y_high],
            minallowed=y_low,
            maxallowed=y_high,
            tickmode="array",
            tickvals=tick_values,
            ticktext=[_format_signed_duration(item) for item in tick_values],
            gridcolor="#303030",
            zeroline=False,
        ),
    )
    return figure


def nightly_duration_quality_figure(
    sessions: pd.DataFrame, target_sleep_hours: float = TARGET_SLEEP_HOURS
) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    frame["sleep_hours"] = frame["sleep_minutes"] / 60
    frame["garmin_score_100"] = frame["legacy_garmin_score_10"] * 10
    frame = frame.sort_values("sleep_hours")

    if frame.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No scored night sleep in the selected date range",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        figure.update_layout(
            title="Nightly duration vs Sleep score",
            height=540,
            margin=dict(l=80, r=24, t=64, b=72),
            xaxis=dict(
                title="Nightly sleep duration (HH:MM)",
                range=[0, 12],
                minallowed=0,
                maxallowed=12,
            ),
            yaxis=dict(
                title="Sleep score (0–100)",
                range=[0, 100],
                minallowed=0,
                maxallowed=100,
                dtick=10,
            ),
        )
        return figure

    quality_mean = float(frame["garmin_score_100"].mean())
    quality_std_value = frame["garmin_score_100"].std(ddof=1)
    quality_std = float(quality_std_value) if pd.notna(quality_std_value) else 0.0
    centered = frame["sleep_hours"] - frame["sleep_hours"].mean()
    denominator = float((centered**2).sum())
    slope = (
        float(
            (
                centered
                * (frame["garmin_score_100"] - quality_mean)
            ).sum()
            / denominator
        )
        if denominator
        else 0.0
    )
    intercept = quality_mean - slope * float(frame["sleep_hours"].mean())
    x_max = max(12.0, math.ceil(float(frame["sleep_hours"].max())))
    trend_x = [0.0, x_max]
    trend_y = [intercept, intercept + slope * x_max]

    figure = go.Figure()
    figure.add_hrect(
        y0=max(0.0, quality_mean - quality_std),
        y1=min(100.0, quality_mean + quality_std),
        fillcolor=SLEEP_SCORE_PINK,
        opacity=0.14,
        line_width=0,
        layer="below",
    )
    figure.add_hline(
        y=quality_mean,
        line=dict(color=SLEEP_SCORE_PINK, width=1.5, dash="dash"),
    )
    figure.add_vline(
        x=target_sleep_hours,
        line=dict(color=STATS_RED, width=2),
    )
    figure.add_trace(
        go.Scatter(
            name="Night sleep",
            x=frame["sleep_hours"],
            y=frame["garmin_score_100"],
            mode="markers",
            marker=dict(color=SLEEP_SCORE_PINK, size=5),
            customdata=frame[["sleep_date", "aid_name"]],
            hovertemplate=(
                "%{customdata[0]}<br>Duration: %{x:.2f} h<br>"
                "Score: %{y}/100<br>Aid: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Trend",
            x=trend_x,
            y=trend_y,
            mode="lines",
            line=dict(color=SLEEP_SCORE_PINK, width=2.2),
            hoverinfo="skip",
        )
    )

    x_ticks = list(range(0, int(x_max) + 1, 2))
    figure.update_layout(
        title="Nightly duration vs Sleep score",
        showlegend=False,
        height=540,
        margin=dict(l=80, r=24, t=64, b=72),
        xaxis=dict(
            title="Nightly sleep duration (HH:MM)",
            range=[0, x_max],
            minallowed=0,
            maxallowed=x_max,
            tickmode="array",
            tickvals=x_ticks,
            ticktext=[f"{hour}:00" for hour in x_ticks],
            gridcolor="#303030",
        ),
        yaxis=dict(
            title="Sleep score (0–100)",
            range=[0, 100],
            minallowed=0,
            maxallowed=100,
            dtick=10,
            gridcolor="#303030",
            zeroline=False,
        ),
    )
    _add_spearman_annotation(
        figure, frame["sleep_hours"], frame["garmin_score_100"]
    )
    return figure


def asleep_time_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[sessions["session_type"] == "NIGHT_SLEEP"].copy()
    frame = frame.sort_values("sleep_start")
    values = frame["sleep_start"].map(lambda value: _clock_hour(value, 12))
    return _stats_timeline_figure(
        frame["sleep_start"],
        values,
        title="Asleep time",
        yaxis_title="Asleep time",
        target=TARGET_ASLEEP_HOUR,
        y_range=(12, 36),
        tick_values=[13, 17, 21, 25, 29, 33],
        tick_text=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
        formatted_values=[_format_clock_hour(item) for item in values],
        color=ASLEEP_TIME_COLOR,
    )


def wake_up_time_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[sessions["session_type"] == "NIGHT_SLEEP"].copy()
    frame = frame.sort_values("wake_time")
    values = frame["wake_time"].map(_clock_hour)
    return _stats_timeline_figure(
        frame["wake_time"],
        values,
        title="Wake-up time",
        yaxis_title="Wake-up time",
        target=TARGET_WAKE_HOUR,
        y_range=(0, 24),
        tick_values=[1, 5, 9, 13, 17, 21],
        tick_text=["01:00", "05:00", "09:00", "13:00", "17:00", "21:00"],
        formatted_values=[_format_clock_hour(item) for item in values],
        color=WAKE_TIME_COLOR,
    )


def _weekday_weekend_time_figure(
    sessions: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    yaxis_title: str,
    y_range: tuple[float, float],
    tick_values: list[float],
    tick_text: list[str],
    weekday_color: str,
    weekend_color: str,
    boundary_hour: int,
) -> go.Figure:
    frame = sessions[sessions["session_type"] == "NIGHT_SLEEP"].copy()
    frame = frame.sort_values("sleep_date")
    frame["timeline"] = pd.to_datetime(frame["sleep_date"])
    frame["clock_value"] = frame[value_column].map(
        lambda value: _clock_hour(value, boundary_hour)
    )
    frame["day_group"] = frame["timeline"].dt.dayofweek.map(
        lambda weekday: "Weekend" if weekday >= 5 else "Weekday"
    )

    figure = go.Figure()
    for group, color in [
        ("Weekday", weekday_color),
        ("Weekend", weekend_color),
    ]:
        subset = frame[frame["day_group"] == group]
        figure.add_trace(
            go.Scatter(
                name=group,
                x=subset["timeline"],
                y=subset["clock_value"],
                mode="lines",
                line=dict(color=color, width=1.8),
                customdata=[
                    [_format_clock_hour(value)] for value in subset["clock_value"]
                ],
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Time: %{customdata[0]}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    xaxis: dict[str, object] = {"title": "Sleep date", "tickformat": "%d/%m"}
    if not frame.empty:
        x_start, x_end = _date_axis_limits(frame["timeline"])
        xaxis.update(
            type="date",
            range=[x_start, x_end],
            minallowed=x_start,
            maxallowed=x_end,
        )
    figure.update_layout(
        title=title,
        height=540,
        margin=dict(l=90, r=24, t=64, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
        xaxis=xaxis,
        yaxis=dict(
            title=yaxis_title,
            range=list(y_range),
            minallowed=y_range[0],
            maxallowed=y_range[1],
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            zeroline=False,
        ),
    )
    return figure


def asleep_time_weekday_weekend_figure(sessions: pd.DataFrame) -> go.Figure:
    return _weekday_weekend_time_figure(
        sessions,
        value_column="sleep_start",
        title="Asleep time — weekdays vs weekends",
        yaxis_title="Asleep time",
        y_range=(12, 36),
        tick_values=[13, 17, 21, 25, 29, 33],
        tick_text=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
        weekday_color=ASLEEP_TIME_COLOR,
        weekend_color="#A78BFA",
        boundary_hour=12,
    )


def wake_up_time_weekday_weekend_figure(sessions: pd.DataFrame) -> go.Figure:
    return _weekday_weekend_time_figure(
        sessions,
        value_column="wake_time",
        title="Wake-up time — weekdays vs weekends",
        yaxis_title="Wake-up time",
        y_range=(0, 24),
        tick_values=[1, 5, 9, 13, 17, 21],
        tick_text=["01:00", "05:00", "09:00", "13:00", "17:00", "21:00"],
        weekday_color=WAKE_TIME_COLOR,
        weekend_color="#FDBA74",
        boundary_hour=0,
    )


def _weekend_highlight_time_figure(
    sessions: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    yaxis_title: str,
    y_range: tuple[float, float],
    tick_values: list[float],
    tick_text: list[str],
    weekday_color: str,
    weekend_color: str,
    boundary_hour: int,
) -> go.Figure:
    frame = sessions[sessions["session_type"] == "NIGHT_SLEEP"].copy()
    frame = frame.sort_values("sleep_date")
    frame["timeline"] = pd.to_datetime(frame["sleep_date"])
    frame["clock_value"] = frame[value_column].map(
        lambda value: _clock_hour(value, boundary_hour)
    )
    frame["is_weekend"] = frame["timeline"].dt.dayofweek >= 5
    frame["day_group"] = frame["is_weekend"].map(
        {False: "Weekday", True: "Weekend"}
    )
    figure = go.Figure()
    if not frame.empty:
        mean = float(frame["clock_value"].mean())
        std_value = frame["clock_value"].std(ddof=1)
        std = float(std_value) if pd.notna(std_value) else 0.0
        figure.add_hrect(
            y0=max(y_range[0], mean - std),
            y1=min(y_range[1], mean + std),
            fillcolor=weekday_color,
            opacity=0.14,
            line_width=0,
            layer="below",
        )
        figure.add_hline(
            y=mean,
            line=dict(color=weekday_color, width=1.5, dash="dash"),
        )
    for index in range(len(frame) - 1):
        current = frame.iloc[index]
        following = frame.iloc[index + 1]
        segment_color = weekend_color if following["is_weekend"] else weekday_color
        figure.add_trace(
            go.Scatter(
                x=[current["timeline"], following["timeline"]],
                y=[current["clock_value"], following["clock_value"]],
                mode="lines",
                line=dict(color=segment_color, width=1.8),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            name="Overall trend",
            x=frame["timeline"],
            y=_linear_trend(frame["clock_value"]),
            mode="lines",
            line=dict(color=weekday_color, width=2.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    xaxis: dict[str, object] = {"title": "Sleep date", "tickformat": "%d/%m"}
    if not frame.empty:
        x_start, x_end = _date_axis_limits(frame["timeline"])
        xaxis.update(
            type="date",
            range=[x_start, x_end],
            minallowed=x_start,
            maxallowed=x_end,
        )
    figure.update_layout(
        title=title,
        height=540,
        margin=dict(l=90, r=24, t=64, b=64),
        hovermode="closest",
        xaxis=xaxis,
        yaxis=dict(
            title=yaxis_title,
            range=list(y_range),
            minallowed=y_range[0],
            maxallowed=y_range[1],
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            zeroline=False,
        ),
    )
    return figure


def asleep_time_weekend_highlight_figure(sessions: pd.DataFrame) -> go.Figure:
    return _weekend_highlight_time_figure(
        sessions,
        value_column="sleep_start",
        title="Asleep time — weekends highlighted",
        yaxis_title="Asleep time",
        y_range=(12, 36),
        tick_values=[13, 17, 21, 25, 29, 33],
        tick_text=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
        weekday_color=ASLEEP_TIME_COLOR,
        weekend_color="#A78BFA",
        boundary_hour=12,
    )


def wake_up_time_weekend_highlight_figure(sessions: pd.DataFrame) -> go.Figure:
    return _weekend_highlight_time_figure(
        sessions,
        value_column="wake_time",
        title="Wake-up time — weekends highlighted",
        yaxis_title="Wake-up time",
        y_range=(0, 24),
        tick_values=[1, 5, 9, 13, 17, 21],
        tick_text=["01:00", "05:00", "09:00", "13:00", "17:00", "21:00"],
        weekday_color=WAKE_TIME_COLOR,
        weekend_color="#FDBA74",
        boundary_hour=0,
    )


def nightly_duration_timeline_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[sessions["session_type"] == "NIGHT_SLEEP"].copy()
    frame = frame.sort_values("sleep_start")
    values = frame["sleep_minutes"] / 60
    y_max = max(15.0, math.ceil(float(values.max())))
    tick_values = list(range(0, int(y_max) + 1, 3))
    return _stats_timeline_figure(
        frame["sleep_start"],
        values,
        title="Nightly sleep duration (no naps)",
        yaxis_title="Nightly sleep duration (HH:MM)",
        target=TARGET_SLEEP_HOURS,
        y_range=(0, y_max),
        tick_values=tick_values,
        tick_text=[_format_duration_hour(item) for item in tick_values],
        formatted_values=[_format_duration_hour(item) for item in values],
    )


def daily_duration_timeline_figure(daily: pd.DataFrame) -> go.Figure:
    frame = daily.sort_values("date").copy()
    values = frame["total_sleep_minutes"] / 60
    y_max = max(15.0, math.ceil(float(values.max())))
    tick_values = list(range(0, int(y_max) + 1, 3))
    return _stats_timeline_figure(
        frame["date"],
        values,
        title="Daily sleep duration (with naps)",
        yaxis_title="Daily sleep duration (HH:MM)",
        target=TARGET_SLEEP_HOURS,
        y_range=(0, y_max),
        tick_values=tick_values,
        tick_text=[_format_duration_hour(item) for item in tick_values],
        formatted_values=[_format_duration_hour(item) for item in values],
    )


def _aid_order_by_sample_count(
    frame: pd.DataFrame, sample_column: str | None = None
) -> list[object]:
    if frame.empty:
        return []
    if sample_column is None:
        counts = frame.groupby("aid_name", dropna=False).size()
    else:
        counts = frame.groupby("aid_name", dropna=False)[sample_column].nunique()
    return sorted(
        counts.index.tolist(),
        key=lambda aid: (
            str(aid).casefold() != "none",
            -int(counts.loc[aid]),
            str(aid).casefold(),
        ),
    )


def aid_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions.copy()
    frame["sleep_hours"] = frame["sleep_minutes"] / 60
    order = _aid_order_by_sample_count(frame, "session_id")
    figure = go.Figure()
    for aid in order:
        subset = frame[frame["aid_name"] == aid]
        color = _aid_color(aid)
        fill_color = _aid_fill_color(aid)
        figure.add_trace(
            go.Box(
                name=aid,
                y=subset["sleep_hours"],
                boxpoints="all",
                jitter=0.28,
                pointpos=0,
                marker_color=color,
                line_color=color,
                fillcolor=fill_color,
                customdata=subset[["sleep_date", "session_type"]],
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]}<br>"
                    "%{y:.2f} h<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        title="Session duration by recorded sleep aid",
        showlegend=False,
        height=500,
        margin=dict(l=60, r=24, t=64, b=90),
        xaxis_title="Recorded aid",
        yaxis_title="Hours asleep",
    )
    return figure


def aid_sleep_stage_composition_figure(
    sessions: pd.DataFrame,
    sleep_stages: pd.DataFrame,
    use_stage_duration: bool = False,
) -> go.Figure:
    figure = go.Figure()
    nights = sessions[sessions["session_type"] == "NIGHT_SLEEP"][
        ["session_id", "aid_name"]
    ].copy()
    if nights.empty or sleep_stages.empty:
        figure.add_annotation(
            text="No sleep-stage data by recorded aid in this date range",
            showarrow=False,
        )
    else:
        stages = sleep_stages.merge(nights, on="session_id", how="inner").copy()
        stages["stage_group"] = stages["stage"].replace(
            {
                "Stage unavailable": "Unavailable",
                "Unmeasurable": "Unavailable",
            }
        )
        stages = stages[
            stages["stage_group"].isin(
                ["Deep", "REM", "Light", "Awake", "Unavailable"]
            )
        ].copy()
        stages["minutes"] = [
            (interval_end - interval_start).total_seconds() / 60
            for interval_start, interval_end in zip(
                stages["interval_start"], stages["interval_end"]
            )
        ]
        nightly = (
            stages.groupby(
                ["session_id", "aid_name", "stage_group"], as_index=False
            )["minutes"]
            .sum()
            .pivot_table(
                index=["session_id", "aid_name"],
                columns="stage_group",
                values="minutes",
                fill_value=0,
            )
            .reset_index()
        )
        stage_order = ["Deep", "REM", "Light", "Awake", "Unavailable"]
        for stage in stage_order:
            if stage not in nightly:
                nightly[stage] = 0.0
        nightly["stage_total"] = nightly[stage_order].sum(axis=1)
        nightly = nightly[nightly["stage_total"] > 0].copy()
        if use_stage_duration:
            for stage in stage_order:
                nightly[stage] = nightly[stage] / 60
        else:
            for stage in stage_order:
                nightly[stage] = nightly[stage] / nightly["stage_total"] * 100
        # Every aid with nights in range is shown, with how many of its nights
        # have stage data; aids with none (e.g. only Sleepmeter-era nights) get
        # an empty, labelled slot instead of silently disappearing.
        all_nights = nights.groupby("aid_name")["session_id"].nunique()
        composition = (
            nightly.groupby("aid_name")[stage_order].mean().reindex(all_nights.index)
        )
        counts = (
            nightly.groupby("aid_name")["session_id"]
            .nunique()
            .reindex(all_nights.index, fill_value=0)
        )
        aid_order = sorted(
            all_nights.index.tolist(),
            key=lambda aid: (
                str(aid).casefold() != "none",
                -int(counts.loc[aid]),
                -int(all_nights.loc[aid]),
                str(aid).casefold(),
            ),
        )
        labels = {
            aid: f"{aid}<br>{int(counts.loc[aid])} of {int(all_nights.loc[aid])} nights"
            for aid in aid_order
        }
        colors = {
            "Deep": "#331B8C",
            "REM": "#EE55C0",
            "Light": "#5B32D6",
            "Awake": "#9CA3AF",
            "Unavailable": "#6345CC",
        }
        for stage in stage_order:
            figure.add_trace(
                go.Bar(
                    name=stage,
                    x=[labels[aid] for aid in aid_order],
                    y=[composition.loc[aid, stage] for aid in aid_order],
                    marker=dict(color=colors[stage], line_width=0),
                    customdata=[[str(aid)] for aid in aid_order],
                    hovertemplate=(
                        f"%{{customdata[0]}}<br>{stage}: %{{y:.2f}}"
                        f"{' h' if use_stage_duration else '%'}<extra></extra>"
                    ),
                )
            )
        for aid in aid_order:
            if counts.loc[aid] == 0:
                figure.add_annotation(
                    x=labels[aid],
                    y=0,
                    yanchor="bottom",
                    text="No stage data",
                    showarrow=False,
                    font=dict(color="#9CA3AF"),
                )
    figure.update_layout(
        title=(
            "Average sleep-stage duration by recorded aid"
            if use_stage_duration
            else "Average sleep-stage composition by recorded aid"
        ),
        barmode="stack",
        bargap=0.25,
        height=500,
        margin=dict(l=72, r=28, t=82, b=82),
        xaxis=dict(title="Recorded aid", showgrid=False),
        yaxis=(
            dict(title="Average stage duration per night (hours)", rangemode="tozero")
            if use_stage_duration
            else dict(
                title="Average share of recorded sleep window",
                range=[0, 100],
                ticksuffix="%",
            )
        ),
        legend=dict(orientation="h", y=1.02),
    )
    return figure


def aid_duration_timeline_figure(
    daily: pd.DataFrame,
    sessions: pd.DataFrame,
    view_start: date | None = None,
    view_end: date | None = None,
) -> go.Figure:
    """The daily duration chart's bars, coloured by the night's sleep aid.

    Uses the same daily totals (including Garmin naps), awake time, date axis,
    zero-sleep markers, and pannable view as ``daily_duration_figure``.
    """
    frame = daily.copy()
    frame["date_plot"] = pd.to_datetime(frame["date"]) + pd.Timedelta(hours=12)
    frame["night_hours"] = frame["night_minutes"] / 60
    frame["nap_hours"] = frame["nap_minutes"] / 60
    nights = sessions[sessions["session_type"] == "NIGHT_SLEEP"]
    # A day's aid is the one recorded for its longest night session.
    longest_nights = nights.sort_values("sleep_minutes").groupby("sleep_date").last()
    frame["aid_name"] = frame["date"].map(longest_nights["aid_name"]).fillna("None")
    frame["awake_hours"] = (
        frame["date"].map(nights.groupby("sleep_date")["interruption_minutes"].sum())
        .fillna(0)
        / 60
    )

    figure = go.Figure()
    order = _aid_order_by_sample_count(frame, "date")
    for aid in order:
        subset = frame[frame["aid_name"] == aid]
        figure.add_trace(
            go.Bar(
                name=f"{aid} · night",
                legendgroup=f"{aid} · night",
                x=subset["date_plot"],
                y=subset["night_hours"],
                marker=dict(color=_aid_color(aid), line=dict(width=0)),
                hovertemplate=f"%{{x|%d/%m/%Y}}<br>{aid} · night: %{{y:.2f}} h<extra></extra>",
            )
        )
    figure.add_trace(
        go.Bar(
            name="Awake / interruptions",
            x=frame["date_plot"],
            y=frame["awake_hours"],
            marker=dict(color="#9CA3AF", line=dict(width=0)),
            hovertemplate="%{x|%d/%m/%Y}<br>Awake / interruptions: %{y:.2f} h<extra></extra>",
        )
    )
    for aid in order:
        subset = frame[(frame["aid_name"] == aid) & (frame["nap_hours"] > 0)]
        if subset.empty:
            continue
        figure.add_trace(
            go.Bar(
                name=f"{aid} · nap",
                legendgroup=f"{aid} · nap",
                x=subset["date_plot"],
                y=subset["nap_hours"],
                marker=dict(color=_aid_nap_color(aid), line=dict(width=0)),
                hovertemplate=f"%{{x|%d/%m/%Y}}<br>{aid} · nap: %{{y:.2f}} h<extra></extra>",
            )
        )
    zero = frame[frame["zero_sleep"]]
    if not zero.empty:
        figure.add_trace(
            go.Bar(
                name="Zero sleep",
                x=zero["date_plot"],
                y=[0.08] * len(zero),
                base=[0] * len(zero),
                marker_color=ZERO_COLOR,
                hovertemplate="<b>Zero sleep</b><br>%{x|%d/%m/%Y}<extra></extra>",
            )
        )
    _add_no_data_markers(figure, frame)

    scored_nights = nights[nights["legacy_garmin_score_10"].notna()]
    daily_scores = scored_nights.groupby("sleep_date")["legacy_garmin_score_10"].mean() * 10
    frame["sleep_score_100"] = frame["date"].map(daily_scores)
    figure.add_trace(
        go.Scatter(
            name="Garmin Sleep score (0–100)",
            x=frame["date_plot"],
            y=frame["sleep_score_100"],
            yaxis="y2",
            mode="lines",
            visible="legendonly",
            connectgaps=False,
            line=dict(color=SLEEP_SCORE_PINK, width=1),
            hovertemplate="%{x|%d/%m/%Y}<br>Garmin Sleep score: %{y:.0f}/100<extra></extra>",
        )
    )
    figure.update_layout(
        title="Daily sleep duration by recorded sleep aid",
        barmode="stack",
        height=520,
        margin=dict(**DATE_CHART_SIDE_MARGINS, t=64, b=72),
        xaxis=dict(
            title="Sleep date (wake date)",
            type="date",
            tickformat="%d/%m",
            tickangle=0,
        ),
        yaxis=dict(
            title="Hours (night sleep, awake, naps)",
            range=[0, max(_stacked_bar_top(figure) * 1.05, 1.0)],
        ),
        yaxis2=dict(
            title="Garmin Sleep score (0–100)",
            overlaying="y",
            side="right",
            range=[0, 100],
            dtick=20,
            showgrid=False,
            zeroline=False,
        ),
        hovermode="x unified",
    )
    if not frame.empty:
        _pannable_date_view(
            figure, frame["date"].min(), frame["date"].max(), view_start, view_end
        )
    return figure


def aid_score_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    frame["garmin_score_100"] = frame["legacy_garmin_score_10"] * 10
    order = _aid_order_by_sample_count(frame, "session_id")
    figure = go.Figure()
    for aid in order:
        subset = frame[frame["aid_name"] == aid]
        color = _aid_color(aid)
        fill_color = _aid_fill_color(aid)
        figure.add_trace(
            go.Box(
                name=str(aid),
                y=subset["garmin_score_100"],
                boxpoints="all",
                jitter=0.28,
                pointpos=0,
                marker_color=color,
                line_color=color,
                fillcolor=fill_color,
                customdata=subset[["sleep_date", "sleep_minutes"]],
                hovertemplate=(
                    "%{customdata[0]}<br>Garmin Sleep score: %{y}/100<br>"
                    "Sleep: %{customdata[1]:.0f} min"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        title="Garmin Sleep score by recorded sleep aid",
        showlegend=False,
        height=500,
        margin=dict(l=60, r=24, t=64, b=90),
        xaxis_title="Recorded aid",
        yaxis=dict(
            title="Garmin Sleep score (0–100)",
            range=[-2.5, 102.5],
            dtick=10,
        ),
    )
    return figure


def _duration_histogram(
    values: pd.Series, *, title: str, color: str
) -> go.Figure:
    hours = pd.Series(values, dtype="float64").dropna()
    upper_bound = max(1, math.ceil(float(hours.max()))) if not hours.empty else 1
    figure = go.Figure(
        go.Histogram(
            x=hours,
            xbins=dict(start=0, end=upper_bound + 1, size=1),
            marker=dict(
                color=color,
                line=dict(color="rgba(14, 17, 23, 0.9)", width=0.7),
            ),
            hovertemplate=(
                "Duration bin: %{x:.0f} h<br>Observations: %{y}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        height=460,
        margin=dict(l=60, r=24, t=64, b=64),
        bargap=0.08,
        xaxis=dict(
            title="Sleep duration (hours)",
            range=[0, upper_bound + 1],
            dtick=2,
            zeroline=False,
        ),
        yaxis=dict(title="Number of observations", rangemode="tozero"),
    )
    return figure


def nightly_duration_histogram(sessions: pd.DataFrame) -> go.Figure:
    night_sleep = sessions[sessions["session_type"] == "NIGHT_SLEEP"]
    return _duration_histogram(
        night_sleep["sleep_minutes"] / 60,
        title="Nightly sleep duration distribution (naps excluded)",
        color=NIGHT_BAR_COLOR,
    )


def daily_duration_histogram(daily: pd.DataFrame) -> go.Figure:
    return _duration_histogram(
        daily["total_sleep_minutes"] / 60,
        title="Daily sleep duration distribution (naps included)",
        color=NAP_BAR_COLOR,
    )


def sleep_state_pie_figure(
    sessions: pd.DataFrame, daily: pd.DataFrame
) -> go.Figure:
    total_minutes = float(len(daily) * 24 * 60)
    night_minutes = float(
        sessions.loc[
            sessions["session_type"] == "NIGHT_SLEEP", "sleep_minutes"
        ].sum()
    )
    nap_minutes = float(
        sessions.loc[sessions["session_type"] == "NAP", "sleep_minutes"].sum()
    )
    hole_minutes = float(sessions["interruption_minutes"].sum())
    awake_minutes = max(
        0.0,
        total_minutes - night_minutes - nap_minutes - hole_minutes,
    )

    figure = go.Figure()
    if total_minutes == 0:
        figure.add_annotation(
            text="No calendar days in the selected date range",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    else:
        labels = ["Sleep", "Nap", "Awake", "Awake in holes"]
        values = [night_minutes, nap_minutes, awake_minutes, hole_minutes]
        figure.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[NIGHT_BAR_COLOR, NAP_BAR_COLOR, "#64748b", AVERAGE_3_COLOR],
                    line=dict(color="rgba(14, 17, 23, 0.9)", width=1),
                ),
                textinfo="label+percent",
                customdata=[[value / 60] for value in values],
                hovertemplate=(
                    "%{label}<br>%{customdata[0]:.2f} hours<br>"
                    "%{percent}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Time breakdown across selected days",
        height=520,
        margin=dict(l=40, r=40, t=64, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, x=0),
    )
    return figure


def sleep_nap_pie_figure(sessions: pd.DataFrame) -> go.Figure:
    night_minutes = float(
        sessions.loc[
            sessions["session_type"] == "NIGHT_SLEEP", "sleep_minutes"
        ].sum()
    )
    nap_minutes = float(
        sessions.loc[sessions["session_type"] == "NAP", "sleep_minutes"].sum()
    )
    total_sleep_minutes = night_minutes + nap_minutes

    figure = go.Figure()
    if total_sleep_minutes == 0:
        figure.add_annotation(
            text="No sleep in the selected date range",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    else:
        values = [night_minutes, nap_minutes]
        figure.add_trace(
            go.Pie(
                labels=["Sleep", "Nap"],
                values=values,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[NIGHT_BAR_COLOR, NAP_BAR_COLOR],
                    line=dict(color="rgba(14, 17, 23, 0.9)", width=1),
                ),
                textinfo="label+percent",
                customdata=[[value / 60] for value in values],
                hovertemplate=(
                    "%{label}<br>%{customdata[0]:.2f} hours<br>"
                    "%{percent}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Sleep vs naps",
        height=520,
        margin=dict(l=40, r=40, t=64, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, x=0),
    )
    return figure


WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _duration_by_weekday_figure(
    dates: pd.Series,
    hours: pd.Series,
    *,
    title: str,
    color: str,
) -> go.Figure:
    frame = pd.DataFrame(
        {
            "weekday": pd.to_datetime(dates).dt.day_name(),
            "hours": pd.Series(hours, dtype="float64").to_numpy(),
        }
    )
    summary = (
        frame.groupby("weekday")["hours"]
        .agg(["min", "mean", "max", "count"])
        .reindex(WEEKDAY_ORDER)
        .dropna(subset=["max"])
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Duration range",
            x=summary.index,
            y=summary["max"],
            marker=dict(
                color=color,
                opacity=0.55,
                line=dict(color=color, width=1),
            ),
            customdata=summary[["count"]],
            hovertemplate=(
                "%{x}<br>Maximum: %{y:.2f} h<br>"
                "Observations: %{customdata[0]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    for statistic, label, marker_color, symbol in [
        ("min", "Minimum", ZERO_COLOR, "circle"),
        ("mean", "Mean", AVERAGE_7_COLOR, "diamond"),
        ("max", "Maximum", MAXIMUM_MARKER_COLOR, "circle"),
    ]:
        figure.add_trace(
            go.Scatter(
                name=label,
                x=summary.index,
                y=summary[statistic],
                mode="markers",
                marker=dict(
                    color=marker_color,
                    symbol=symbol,
                    size=10,
                    line=dict(color="rgba(14, 17, 23, 0.9)", width=1),
                ),
                hovertemplate=(
                    f"%{{x}}<br>{label}: %{{y:.2f}} h<extra></extra>"
                ),
            )
        )
    y_max = max(1.0, math.ceil(float(summary["max"].max()))) if not summary.empty else 1.0
    figure.update_layout(
        title=title,
        height=520,
        margin=dict(l=60, r=24, t=64, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Day of week", categoryorder="array", categoryarray=WEEKDAY_ORDER),
        yaxis=dict(title="Sleep duration (hours)", range=[0, y_max]),
        hovermode="closest",
    )
    return figure


def daily_duration_by_weekday_figure(daily: pd.DataFrame) -> go.Figure:
    return _duration_by_weekday_figure(
        daily["date"],
        daily["total_sleep_minutes"] / 60,
        title="Duration vs Day of Week (naps included)",
        color=NAP_BAR_COLOR,
    )


def nightly_duration_by_weekday_figure(sessions: pd.DataFrame) -> go.Figure:
    night_sleep = sessions[sessions["session_type"] == "NIGHT_SLEEP"]
    return _duration_by_weekday_figure(
        night_sleep["sleep_date"],
        night_sleep["sleep_minutes"] / 60,
        title="Nightly Duration vs Day of Week (naps excluded)",
        color=NIGHT_BAR_COLOR,
    )


def _duration_by_weekday_boxplot(
    dates: pd.Series,
    hours: pd.Series,
    *,
    title: str,
    color: str,
    fill_color: str,
    weekend_color: str,
    weekend_fill_color: str,
) -> go.Figure:
    frame = pd.DataFrame(
        {
            "weekday": pd.to_datetime(dates).dt.day_name(),
            "hours": pd.Series(hours, dtype="float64").to_numpy(),
        }
    )
    figure = go.Figure()
    for weekday in WEEKDAY_ORDER:
        subset = frame[frame["weekday"] == weekday]
        if subset.empty:
            continue
        is_weekend = weekday in {"Saturday", "Sunday"}
        box_color = weekend_color if is_weekend else color
        box_fill_color = weekend_fill_color if is_weekend else fill_color
        figure.add_trace(
            go.Box(
                name=weekday,
                x=[weekday] * len(subset),
                y=subset["hours"],
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
                marker=dict(color=box_color, size=5),
                line_color=box_color,
                fillcolor=box_fill_color,
                hovertemplate=(
                    "%{x}<br>Duration: %{y:.2f} h<extra></extra>"
                ),
                showlegend=False,
            )
        )
    weekday_means = frame.groupby("weekday")["hours"].mean().reindex(WEEKDAY_ORDER)
    weekday_means = weekday_means.dropna()
    figure.add_trace(
        go.Scatter(
            name="Mean",
            x=weekday_means.index,
            y=weekday_means.values,
            mode="markers",
            marker=dict(
                color=AVERAGE_7_COLOR,
                symbol="diamond",
                size=5,
                line=dict(color="#F8DC7A", width=1),
            ),
            hovertemplate="%{x}<br>Mean: %{y:.2f} h<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=520,
        margin=dict(l=60, r=24, t=64, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
        ),
        yaxis=dict(title="Sleep duration (hours)", rangemode="tozero"),
    )
    return figure


def nightly_duration_by_weekday_boxplot(sessions: pd.DataFrame) -> go.Figure:
    night_sleep = sessions[sessions["session_type"] == "NIGHT_SLEEP"]
    return _duration_by_weekday_boxplot(
        night_sleep["sleep_date"],
        night_sleep["sleep_minutes"] / 60,
        title="Nightly Duration by Day of Week — boxplots",
        color=NIGHT_BAR_COLOR,
        fill_color="rgba(91, 50, 214, 0.35)",
        weekend_color="#A78BFA",
        weekend_fill_color="rgba(167, 139, 250, 0.35)",
    )


def daily_duration_by_weekday_boxplot(daily: pd.DataFrame) -> go.Figure:
    return _duration_by_weekday_boxplot(
        daily["date"],
        daily["total_sleep_minutes"] / 60,
        title="Daily Duration by Day of Week — boxplots",
        color=NAP_BAR_COLOR,
        fill_color="rgba(8, 124, 255, 0.35)",
        weekend_color="#8FC5FF",
        weekend_fill_color="rgba(143, 197, 255, 0.35)",
    )


def hourly_sleep_punch_card(
    intervals: pd.DataFrame,
    *,
    title: str = "Hourly sleep punch card (night sleep + naps)",
    color: str = NAP_BAR_COLOR,
    light_color: str = "#8FC5FF",
) -> go.Figure:
    asleep_hours: set[tuple[date, int]] = set()
    for row in intervals[["interval_start", "interval_end"]].itertuples(index=False):
        cursor = pd.Timestamp(row.interval_start).floor("h")
        interval_end = pd.Timestamp(row.interval_end)
        while cursor < interval_end:
            hour_end = cursor + pd.Timedelta(hours=1)
            if max(cursor, pd.Timestamp(row.interval_start)) < min(hour_end, interval_end):
                asleep_hours.add((cursor.date(), cursor.hour))
            cursor = hour_end

    counts = pd.DataFrame(asleep_hours, columns=["calendar_date", "hour"])
    if counts.empty:
        summary = pd.DataFrame(columns=["weekday", "hour", "count"])
    else:
        counts["weekday"] = pd.to_datetime(counts["calendar_date"]).dt.day_name()
        summary = (
            counts.groupby(["weekday", "hour"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

    max_count = int(summary["count"].max()) if not summary.empty else 1
    marker_sizes = 5 + (summary["count"] / max_count * 25) if not summary.empty else []
    figure = go.Figure(
        go.Scatter(
            x=summary["hour"],
            y=summary["weekday"],
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=summary["count"],
                colorscale=[[0, light_color], [1, color]],
                cmin=0,
                cmax=max_count,
                line=dict(color="rgba(255, 255, 255, 0.35)", width=0.6),
                colorbar=dict(title="Days asleep"),
            ),
            customdata=summary[["count"]],
            hovertemplate=(
                "%{y}<br>%{x}:00–%{x}:59<br>"
                "Days asleep: %{customdata[0]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        title=title,
        height=500,
        margin=dict(l=90, r=80, t=64, b=64),
        xaxis=dict(
            title="Hour of day",
            range=[-0.5, 23.5],
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{hour:02d}:00" for hour in range(24)],
            tickangle=-45,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        ),
    )
    return figure


def asleep_time_vs_duration_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (sessions["sleep_minutes"] > 0)
    ].copy()
    frame["asleep_hour"] = frame["sleep_start"].map(
        lambda value: _clock_hour(value, 12)
    )
    frame["sleep_hours"] = frame["sleep_minutes"] / 60
    frame = frame.sort_values("asleep_hour")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            name="Night sleep",
            x=frame["asleep_hour"],
            y=frame["sleep_hours"],
            mode="markers",
            marker=dict(color=NIGHT_BAR_COLOR, size=7, opacity=0.75),
            customdata=frame[["sleep_date", "aid_name", "sleep_start"]],
            hovertemplate=(
                "%{customdata[0]}<br>Asleep: %{customdata[2]|%H:%M}<br>"
                "Duration: %{y:.2f} h<br>Aid: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    if len(frame) >= 2:
        centered = frame["asleep_hour"] - frame["asleep_hour"].mean()
        denominator = float((centered**2).sum())
        slope = (
            float(
                (
                    centered
                    * (frame["sleep_hours"] - frame["sleep_hours"].mean())
                ).sum()
                / denominator
            )
            if denominator
            else 0.0
        )
        intercept = float(frame["sleep_hours"].mean()) - slope * float(
            frame["asleep_hour"].mean()
        )
        x_start = float(frame["asleep_hour"].min())
        x_end = float(frame["asleep_hour"].max())
        figure.add_trace(
            go.Scatter(
                name="Trend",
                x=[x_start, x_end],
                y=[intercept + slope * x_start, intercept + slope * x_end],
                mode="lines",
                line=dict(color=NIGHT_BAR_COLOR, width=2.2),
                hoverinfo="skip",
            )
        )
    y_max = max(12.0, math.ceil(float(frame["sleep_hours"].max()))) if not frame.empty else 12.0
    figure.update_layout(
        title="Asleep time vs nightly sleep duration",
        height=540,
        margin=dict(l=70, r=24, t=64, b=64),
        showlegend=False,
        xaxis=dict(
            title="Asleep time",
            range=[12, 36],
            tickmode="array",
            tickvals=[13, 17, 21, 25, 29, 33],
            ticktext=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
            zeroline=False,
        ),
        yaxis=dict(
            title="Nightly sleep duration (hours)",
            range=[0, y_max],
            zeroline=False,
        ),
    )
    _add_spearman_annotation(figure, frame["asleep_hour"], frame["sleep_hours"])
    return figure


def calendar_activity_heatmap(
    daily: pd.DataFrame, activities: pd.DataFrame
) -> go.Figure:
    calendar = daily[["date"]].copy()
    calendar["timestamp"] = pd.to_datetime(calendar["date"])
    calendar["week_start"] = calendar["timestamp"] - pd.to_timedelta(
        calendar["timestamp"].dt.dayofweek, unit="D"
    )
    calendar["weekday_number"] = calendar["timestamp"].dt.dayofweek
    weeks = sorted(calendar["week_start"].drop_duplicates())
    week_lookup = {week: index for index, week in enumerate(weeks)}

    if activities.empty:
        activity_days = pd.DataFrame()
    else:
        activity_days = (
            activities.groupby("activity_date", as_index=False)
            .agg(
                activity_type=(
                    "activity_type",
                    lambda values: (
                        str(values.iloc[0])
                        if values.nunique() == 1
                        else "multiple_activities"
                    ),
                ),
                activity_names=(
                    "activity_name",
                    lambda values: ", ".join(dict.fromkeys(map(str, values))),
                ),
                activity_minutes=(
                    "duration_seconds", lambda values: values.sum() / 60
                ),
            )
            .rename(columns={"activity_date": "date"})
        )
        activity_days = activity_days.merge(
            calendar[["date", "timestamp", "week_start", "weekday_number"]],
            on="date",
            how="inner",
        )
        activity_days["week_number"] = activity_days["week_start"].map(week_lookup)
        activity_days["weekday"] = activity_days["weekday_number"].map(
            dict(enumerate(WEEKDAY_ORDER))
        )
        activity_days["date_text"] = activity_days["timestamp"].dt.strftime("%Y-%m-%d")

    figure = go.Figure()
    if not activity_days.empty:
        figure.add_trace(
            go.Scatter(
                x=activity_days["week_number"],
                y=activity_days["weekday"],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=16,
                    color=activity_days["activity_minutes"],
                    cmin=0,
                    cmax=float(activity_days["activity_minutes"].max()),
                    colorscale=[
                        [0.0, "#17251F"],
                        [0.5, "#257A59"],
                        [1.0, "#2FD6A3"],
                    ],
                    line=dict(color="rgba(255,255,255,0.25)", width=1.5),
                    colorbar=dict(title="Minutes", thickness=12),
                ),
                customdata=activity_days[
                    ["date_text", "activity_names", "activity_minutes"]
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]}<br>"
                    "%{customdata[2]:.0f} minutes<extra></extra>"
                ),
                showlegend=False,
            )
        )
    tick_positions = list(range(0, len(weeks), max(1, len(weeks) // 10 or 1)))
    figure.update_layout(
        title="Recorded activity calendar",
        height=330,
        margin=dict(l=80, r=65, t=64, b=56),
        xaxis=dict(
            title="Week starting",
            tickmode="array",
            tickvals=tick_positions,
            ticktext=[pd.Timestamp(weeks[index]).strftime("%d/%m") for index in tick_positions],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
            autorange="reversed",
            showgrid=False,
        ),
    )
    return figure


def calendar_intensity_minutes_heatmap(
    daily: pd.DataFrame, metrics: pd.DataFrame
) -> go.Figure:
    frame = daily[["date"]].copy()
    intensity = (
        metrics.set_index("date")["daily_intensity_minutes"]
        if not metrics.empty and "daily_intensity_minutes" in metrics
        else pd.Series(dtype=float)
    )
    frame["intensity_minutes"] = frame["date"].map(intensity)
    frame = frame[frame["intensity_minutes"].notna()].copy()
    frame["timestamp"] = pd.to_datetime(frame["date"])
    frame["week_start"] = frame["timestamp"] - pd.to_timedelta(
        frame["timestamp"].dt.dayofweek, unit="D"
    )
    frame["weekday_number"] = frame["timestamp"].dt.dayofweek

    calendar_dates = daily[["date"]].copy()
    calendar_dates["timestamp"] = pd.to_datetime(calendar_dates["date"])
    calendar_dates["week_start"] = calendar_dates["timestamp"] - pd.to_timedelta(
        calendar_dates["timestamp"].dt.dayofweek, unit="D"
    )
    weeks = sorted(calendar_dates["week_start"].drop_duplicates())
    week_lookup = {week: index for index, week in enumerate(weeks)}
    frame["week_number"] = frame["week_start"].map(week_lookup)
    frame["weekday"] = frame["weekday_number"].map(dict(enumerate(WEEKDAY_ORDER)))
    frame["date_text"] = frame["timestamp"].dt.strftime("%Y-%m-%d")

    figure = go.Figure()
    if not frame.empty:
        figure.add_trace(
            go.Scatter(
                x=frame["week_number"],
                y=frame["weekday"],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=16,
                    color=frame["intensity_minutes"],
                    cmin=0,
                    cmax=max(1.0, float(frame["intensity_minutes"].max())),
                    colorscale=[
                        [0.0, "#17251F"],
                        [0.5, "#257A59"],
                        [1.0, "#2FD6A3"],
                    ],
                    line=dict(color="rgba(255,255,255,0.25)", width=1.5),
                    colorbar=dict(title="Minutes", thickness=12),
                ),
                customdata=frame[["date_text", "intensity_minutes"]],
                hovertemplate=(
                    "%{customdata[0]}<br>Intensity minutes: "
                    "%{customdata[1]:.0f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    tick_positions = list(range(0, len(weeks), max(1, len(weeks) // 10 or 1)))
    figure.update_layout(
        title="Intensity minutes calendar",
        height=330,
        margin=dict(l=80, r=65, t=64, b=56),
        xaxis=dict(
            title="Week starting",
            tickmode="array",
            tickvals=tick_positions,
            ticktext=[
                pd.Timestamp(weeks[index]).strftime("%d/%m")
                for index in tick_positions
            ],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
            autorange="reversed",
            showgrid=False,
        ),
    )
    return figure


def calendar_sleep_heatmap(daily: pd.DataFrame) -> go.Figure:
    frame = daily.copy()
    frame["timestamp"] = pd.to_datetime(frame["date"])
    frame["week_start"] = frame["timestamp"] - pd.to_timedelta(
        frame["timestamp"].dt.dayofweek, unit="D"
    )
    frame["weekday_number"] = frame["timestamp"].dt.dayofweek
    frame["sleep_hours"] = frame["total_sleep_minutes"] / 60
    weeks = sorted(frame["week_start"].drop_duplicates())
    week_lookup = {week: index for index, week in enumerate(weeks)}
    frame["week_number"] = frame["week_start"].map(week_lookup)
    frame["weekday"] = frame["weekday_number"].map(dict(enumerate(WEEKDAY_ORDER)))
    frame["date_text"] = frame["timestamp"].dt.strftime("%Y-%m-%d")

    figure = go.Figure(
        go.Scatter(
            x=frame["week_number"],
            y=frame["weekday"],
            mode="markers",
            marker=dict(
                symbol="square",
                size=16,
                color=frame["sleep_hours"],
                cmin=0,
                cmax=max(12.0, float(frame["sleep_hours"].max())),
                colorscale=[
                    [0.0, "#111b2e"],
                    [0.5, "#145DA0"],
                    [1.0, NAP_BAR_COLOR],
                ],
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1.5),
                colorbar=dict(title="Hours", thickness=12),
            ),
            customdata=frame[["date_text", "sleep_hours"]],
            hovertemplate=(
                "%{customdata[0]}<br>Sleep: %{customdata[1]:.2f} h<extra></extra>"
            ),
            showlegend=False,
        )
    )
    tick_positions = list(range(0, len(weeks), max(1, len(weeks) // 10 or 1)))
    figure.update_layout(
        title="Daily sleep duration calendar",
        height=330,
        margin=dict(l=80, r=65, t=64, b=56),
        xaxis=dict(
            title="Week starting",
            tickmode="array",
            tickvals=tick_positions,
            ticktext=[pd.Timestamp(weeks[index]).strftime("%d/%m") for index in tick_positions],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
            autorange="reversed",
            showgrid=False,
        ),
    )
    return figure


def calendar_sleep_score_heatmap(
    daily: pd.DataFrame, sessions: pd.DataFrame
) -> go.Figure:
    frame = daily[["date"]].copy()
    scored_nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & sessions["legacy_garmin_score_10"].notna()
    ]
    scores = (
        scored_nights.groupby("sleep_date")["legacy_garmin_score_10"].mean() * 10
    )
    frame["score"] = frame["date"].map(scores)
    frame["timestamp"] = pd.to_datetime(frame["date"])
    frame["week_start"] = frame["timestamp"] - pd.to_timedelta(
        frame["timestamp"].dt.dayofweek, unit="D"
    )
    frame["weekday_number"] = frame["timestamp"].dt.dayofweek
    weeks = sorted(frame["week_start"].drop_duplicates())
    week_lookup = {week: index for index, week in enumerate(weeks)}
    frame["week_number"] = frame["week_start"].map(week_lookup)
    frame["weekday"] = frame["weekday_number"].map(dict(enumerate(WEEKDAY_ORDER)))
    frame["date_text"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    scored_frame = frame[frame["score"].notna()]

    figure = go.Figure(
        go.Scatter(
            x=scored_frame["week_number"],
            y=scored_frame["weekday"],
            mode="markers",
            marker=dict(
                symbol="square",
                size=16,
                color=scored_frame["score"],
                cmin=0,
                cmax=100,
                colorscale=[
                    [0.0, "#2A1721"],
                    [0.5, "#9C496F"],
                    [1.0, SLEEP_SCORE_PINK],
                ],
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1.5),
                colorbar=dict(title="Score", thickness=12),
            ),
            customdata=scored_frame[["date_text", "score"]],
            hovertemplate=(
                "%{customdata[0]}<br>Garmin Sleep score: "
                "%{customdata[1]:.0f}/100<extra></extra>"
            ),
            showlegend=False,
        )
    )
    tick_positions = list(range(0, len(weeks), max(1, len(weeks) // 10 or 1)))
    figure.update_layout(
        title="Garmin Sleep score calendar",
        height=330,
        margin=dict(l=80, r=65, t=64, b=56),
        xaxis=dict(
            title="Week starting",
            tickmode="array",
            tickvals=tick_positions,
            ticktext=[pd.Timestamp(weeks[index]).strftime("%d/%m") for index in tick_positions],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
            autorange="reversed",
            showgrid=False,
        ),
    )
    return figure


def sleep_efficiency_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (sessions["gross_minutes"] > 0)
    ].copy()
    frame = frame.sort_values("sleep_date")
    frame["efficiency"] = frame["sleep_minutes"] / frame["gross_minutes"] * 100
    figure = go.Figure()
    if not frame.empty:
        mean = float(frame["efficiency"].mean())
        std = float(frame["efficiency"].std(ddof=1)) if len(frame) > 1 else 0.0
        figure.add_hrect(
            y0=max(0, mean - std),
            y1=min(100, mean + std),
            fillcolor=NIGHT_BAR_COLOR,
            opacity=0.14,
            line_width=0,
        )
        figure.add_hline(
            y=mean,
            line=dict(color=NIGHT_BAR_COLOR, width=1.5, dash="dash"),
        )
        figure.add_trace(
            go.Scatter(
                x=pd.to_datetime(frame["sleep_date"]),
                y=frame["efficiency"],
                mode="lines",
                line=dict(color=NIGHT_BAR_COLOR, width=1.8),
                customdata=frame[["interruption_minutes"]],
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Efficiency: %{y:.1f}%<br>"
                    "Awake in holes: %{customdata[0]:.0f} min<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.update_layout(
        title="Nightly sleep efficiency",
        height=500,
        margin=dict(l=70, r=24, t=64, b=64),
        xaxis=dict(title="Sleep date", tickformat="%d/%m"),
        yaxis=dict(title="Sleep efficiency", range=[0, 100], ticksuffix="%"),
    )
    return figure


def _night_sleep_midpoints(sessions: pd.DataFrame) -> pd.DataFrame:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (sessions["gross_minutes"] > 0)
    ].copy()
    frame["midpoint"] = frame["sleep_start"] + (
        frame["wake_time"] - frame["sleep_start"]
    ) / 2
    frame["midpoint_hour"] = frame["midpoint"].map(
        lambda value: _clock_hour(value, 12)
    )
    frame["timeline"] = pd.to_datetime(frame["sleep_date"])
    frame["is_weekend"] = frame["timeline"].dt.dayofweek >= 5
    return frame.sort_values("timeline")


def sleep_midpoint_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = _night_sleep_midpoints(sessions)
    figure = go.Figure()
    if not frame.empty:
        mean = float(frame["midpoint_hour"].mean())
        std = float(frame["midpoint_hour"].std(ddof=1)) if len(frame) > 1 else 0.0
        figure.add_hrect(
            y0=max(12, mean - std),
            y1=min(36, mean + std),
            fillcolor=WAKE_TIME_COLOR,
            opacity=0.14,
            line_width=0,
        )
        figure.add_hline(
            y=mean,
            line=dict(color=WAKE_TIME_COLOR, width=1.5, dash="dash"),
        )
        figure.add_trace(
            go.Scatter(
                x=frame["timeline"],
                y=frame["midpoint_hour"],
                mode="lines",
                line=dict(color=WAKE_TIME_COLOR, width=1.8),
                customdata=[[_format_clock_hour(value)] for value in frame["midpoint_hour"]],
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Sleep midpoint: "
                    "%{customdata[0]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.update_layout(
        title="Sleep midpoint regularity",
        height=500,
        margin=dict(l=80, r=24, t=64, b=64),
        xaxis=dict(title="Sleep date", tickformat="%d/%m"),
        yaxis=dict(
            title="Sleep midpoint",
            range=[12, 36],
            tickmode="array",
            tickvals=[13, 17, 21, 25, 29, 33],
            ticktext=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
        ),
    )
    return figure


def social_jetlag_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = _night_sleep_midpoints(sessions)
    groups = ["Weekday", "Weekend"]
    means = [
        float(frame.loc[~frame["is_weekend"], "midpoint_hour"].mean()),
        float(frame.loc[frame["is_weekend"], "midpoint_hour"].mean()),
    ]
    figure = go.Figure(
        go.Bar(
            x=groups,
            y=[value - 12 for value in means],
            base=12,
            marker=dict(color=[WAKE_TIME_COLOR, "#FDBA74"]),
            customdata=[[_format_clock_hour(value)] for value in means],
            hovertemplate=(
                "%{x}<br>Mean sleep midpoint: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    if all(pd.notna(value) for value in means):
        difference = abs(means[1] - means[0])
        figure.add_annotation(
            text=f"Difference: {_format_duration_hour(difference)}",
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    figure.update_layout(
        title="Social jetlag — mean sleep midpoint",
        height=460,
        margin=dict(l=80, r=24, t=88, b=64),
        showlegend=False,
        xaxis_title="Day group",
        yaxis=dict(
            title="Mean sleep midpoint",
            range=[12, 36],
            tickmode="array",
            tickvals=[13, 17, 21, 25, 29, 33],
            ticktext=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
        ),
    )
    return figure


def score_by_weekday_boxplot(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    frame["weekday"] = pd.to_datetime(frame["sleep_date"]).dt.day_name()
    frame["score_100"] = frame["legacy_garmin_score_10"] * 10

    figure = go.Figure()
    for weekday in WEEKDAY_ORDER:
        subset = frame[frame["weekday"] == weekday]
        if subset.empty:
            continue
        is_weekend = weekday in {"Saturday", "Sunday"}
        color = "#FFB3D1" if is_weekend else SLEEP_SCORE_PINK
        fill_color = (
            "rgba(255, 179, 209, 0.35)"
            if is_weekend
            else "rgba(255, 122, 182, 0.35)"
        )
        figure.add_trace(
            go.Box(
                name=weekday,
                x=[weekday] * len(subset),
                y=subset["score_100"],
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
                marker=dict(color=color, size=5),
                line_color=color,
                fillcolor=fill_color,
                hovertemplate=(
                    "%{x}<br>Garmin Sleep score: %{y:.0f}/100<extra></extra>"
                ),
                showlegend=False,
            )
        )
    means = frame.groupby("weekday")["score_100"].mean().reindex(WEEKDAY_ORDER)
    means = means.dropna()
    figure.add_trace(
        go.Scatter(
            name="Mean",
            x=means.index,
            y=means.values,
            mode="markers",
            marker=dict(
                color=AVERAGE_7_COLOR,
                symbol="diamond",
                size=7,
                line=dict(color="#F8DC7A", width=1),
            ),
            hovertemplate=(
                "%{x}<br>Mean Garmin Sleep score: %{y:.1f}/100<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Garmin Sleep score by day of week",
        height=520,
        margin=dict(l=70, r=24, t=64, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
        ),
        yaxis=dict(title="Garmin Sleep score (0–100)", range=[0, 100], dtick=10),
    )
    return figure


def _add_scatter_trend(
    figure: go.Figure,
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    color: str,
) -> None:
    if len(x_values) < 2:
        return
    x_numeric = pd.Series(x_values, dtype="float64").reset_index(drop=True)
    y_numeric = pd.Series(y_values, dtype="float64").reset_index(drop=True)
    centered = x_numeric - x_numeric.mean()
    denominator = float((centered**2).sum())
    slope = (
        float((centered * (y_numeric - y_numeric.mean())).sum() / denominator)
        if denominator
        else 0.0
    )
    intercept = float(y_numeric.mean()) - slope * float(x_numeric.mean())
    x_start = float(x_numeric.min())
    x_end = float(x_numeric.max())
    figure.add_trace(
        go.Scatter(
            name="Trend",
            x=[x_start, x_end],
            y=[intercept + slope * x_start, intercept + slope * x_end],
            mode="lines",
            line=dict(color=color, width=2.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def asleep_time_vs_score_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (sessions["sleep_minutes"] > 0)
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    frame["asleep_hour"] = frame["sleep_start"].map(
        lambda value: _clock_hour(value, 12)
    )
    frame["asleep_text"] = frame["asleep_hour"].map(_format_clock_hour)
    frame["score_100"] = frame["legacy_garmin_score_10"] * 10
    figure = go.Figure(
        go.Scatter(
            x=frame["asleep_hour"],
            y=frame["score_100"],
            mode="markers",
            marker=dict(color=SLEEP_SCORE_PINK, size=7, opacity=0.75),
            customdata=frame[["sleep_date", "aid_name", "asleep_text"]],
            hovertemplate=(
                "%{customdata[0]}<br>Asleep: %{customdata[2]}<br>"
                "Garmin Sleep score: %{y:.0f}/100<br>"
                "Aid: %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    _add_scatter_trend(
        figure,
        frame["asleep_hour"],
        frame["score_100"],
        color=SLEEP_SCORE_PINK,
    )
    figure.update_layout(
        title="Asleep time vs Garmin Sleep score",
        height=520,
        margin=dict(l=70, r=24, t=64, b=64),
        xaxis=dict(
            title="Asleep time",
            range=[12, 36],
            tickmode="array",
            tickvals=[13, 17, 21, 25, 29, 33],
            ticktext=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
        ),
        yaxis=dict(title="Garmin Sleep score (0–100)", range=[0, 100], dtick=10),
    )
    _add_spearman_annotation(figure, frame["asleep_hour"], frame["score_100"])
    return figure


def sleep_efficiency_vs_score_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (sessions["gross_minutes"] > 0)
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    frame["efficiency"] = frame["sleep_minutes"] / frame["gross_minutes"] * 100
    frame["score_100"] = frame["legacy_garmin_score_10"] * 10
    figure = go.Figure(
        go.Scatter(
            x=frame["efficiency"],
            y=frame["score_100"],
            mode="markers",
            marker=dict(color=SLEEP_SCORE_PINK, size=7, opacity=0.75),
            customdata=frame[["sleep_date", "interruption_minutes", "aid_name"]],
            hovertemplate=(
                "%{customdata[0]}<br>Sleep efficiency: %{x:.1f}%<br>"
                "Garmin Sleep score: %{y:.0f}/100<br>"
                "Awake in holes: %{customdata[1]:.0f} min<br>"
                "Aid: %{customdata[2]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    _add_scatter_trend(
        figure,
        frame["efficiency"],
        frame["score_100"],
        color=SLEEP_SCORE_PINK,
    )
    figure.update_layout(
        title="Sleep efficiency vs Garmin Sleep score",
        height=520,
        margin=dict(l=70, r=24, t=64, b=64),
        xaxis=dict(title="Sleep efficiency", range=[0, 100], ticksuffix="%"),
        yaxis=dict(title="Garmin Sleep score (0–100)", range=[0, 100], dtick=10),
    )
    _add_spearman_annotation(figure, frame["efficiency"], frame["score_100"])
    return figure
