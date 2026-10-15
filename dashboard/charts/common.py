"""Constants and helpers shared by charts on more than one tab."""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go

from dashboard.config import SLEEP_TARGET_HOURS


NIGHT_BAR_COLOR = "#5B32D6"


NAP_BAR_COLOR = "#087CFF"


ZERO_COLOR = "#ff3030"


NO_DATA_COLOR = "#6B7280"


AVERAGE_7_COLOR = "#E6B84F"


SLEEP_SCORE_PINK = "#FF7AB6"
# One colour per sleep outcome, shared by the "vs the following night" charts.
DURATION_COLOR = NIGHT_BAR_COLOR
SCORE_COLOR = SLEEP_SCORE_PINK
HRV_COLOR = "#4DB88A"
STRESS_COLOR = "#E6B84F"


STATS_GREEN = "#00ee28"


STATS_RED = "#ff2020"


TARGET_SLEEP_HOURS = SLEEP_TARGET_HOURS


MAXIMUM_MARKER_COLOR = "#22c55e"


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


def _clock_hour(value: datetime, boundary_hour: int = 0) -> float:
    hour = value.hour + value.minute / 60 + value.second / 3600
    return hour + 24 if hour < boundary_hour else hour


def _date_axis_limits(values: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return day-aligned bounds that contain every timestamp without empty tails."""
    start = pd.Timestamp(values.min()).normalize()
    end = pd.Timestamp(values.max()).normalize() + pd.Timedelta(days=1)
    return start, end


WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


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
