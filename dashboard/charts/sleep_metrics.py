"""Shared layout for charts that set one predictor against four sleep metrics.

Used by the Activity & sleep and Timing tabs: one row of four panels
(duration, sleep score, HRV, stress), each with a coloured header, its
Spearman correlation and a dotted trend line.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.charts.common import (
    DURATION_COLOR,
    HRV_COLOR,
    SCORE_COLOR,
    STRESS_COLOR,
    _spearman_summary,
)


# The four sleep metrics every "vs sleep" chart shows, in this
# column order and colour: (column, name, unit shown in the panel header,
# hover unit, colour, fixed y range).
SLEEP_METRICS = (
    ("sleep_hours", "Duration", "hours", "h", DURATION_COLOR, None),
    ("sleep_score", "Sleep score", "0–100", "/100", SCORE_COLOR, [0, 100]),
    ("overnight_hrv", "HRV", "ms", "ms", HRV_COLOR, None),
    ("sleep_stress", "Stress", "0–100", "", STRESS_COLOR, None),
)
HEADER_UNIT_COLOR = "#94a3b8"


def _sleep_metric_grid(rows: int = 1, height: int = 430) -> go.Figure:
    """One row of four sleep-metric panels per predictor, without panel titles."""
    figure = make_subplots(
        rows=rows,
        cols=len(SLEEP_METRICS),
        horizontal_spacing=0.055,
        vertical_spacing=0.2 if rows > 1 else 0.1,
    )
    for column, (_, name, unit, _, color, y_range) in enumerate(SLEEP_METRICS, 1):
        for row in range(1, rows + 1):
            figure.update_yaxes(
                range=y_range,
                # HRV and stress start at zero; duration zooms to its spread.
                rangemode="tozero" if column >= 3 else None,
                row=row,
                col=column,
            )
            _add_panel_header(figure, row, column, name, unit, color)
    figure.update_layout(height=height, margin=dict(l=50, r=30, t=80, b=80))
    return figure


def _add_panel_header(
    figure: go.Figure, row: int, col: int, name: str, unit: str, color: str
) -> None:
    """Horizontal label above a panel: the metric in its colour, then its unit.

    Replaces rotated y-axis titles, which crowd the tick labels in narrow panels.
    """
    suffix = _axis_suffix(figure, row, col)
    figure.add_annotation(
        text=f'<b>{name}</b> <span style="color:{HEADER_UNIT_COLOR}">{unit}</span>',
        xref=f"x{suffix} domain",
        yref=f"y{suffix} domain",
        x=0,
        y=1,
        xanchor="left",
        yanchor="bottom",
        yshift=8,
        showarrow=False,
        font=dict(size=13, color=color),
    )


def _axis_suffix(figure: go.Figure, row: int, col: int) -> str:
    index = (row - 1) * len(SLEEP_METRICS) + col
    return "" if index == 1 else str(index)


def _add_correlation(
    figure: go.Figure, row: int, col: int, x_values: pd.Series, y_values: pd.Series
) -> None:
    """Spearman ρ and pair count on the panel's header line, right-aligned."""
    summary = _spearman_summary(x_values, y_values)
    if summary is None:
        return
    rho, count = summary
    suffix = _axis_suffix(figure, row, col)
    figure.add_annotation(
        text=f"ρ {rho:+.2f} · n={count}",
        xref=f"x{suffix} domain",
        yref=f"y{suffix} domain",
        x=1,
        y=1,
        xanchor="right",
        yanchor="bottom",
        yshift=8,
        showarrow=False,
        font=dict(size=12, color=HEADER_UNIT_COLOR),
        hovertext=(
            f"Spearman correlation ρ = {rho:+.2f}<br>"
            f"n = {count} nights"
        ),
    )


def _add_row_x_title(figure: go.Figure, row: int, text: str) -> None:
    """One x-axis title centred under a row instead of one per panel."""
    suffix = _axis_suffix(figure, row, 1)
    bottom = figure.layout[f"yaxis{suffix}"].domain[0]
    figure.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=bottom,
        yshift=-34,
        yanchor="top",
        showarrow=False,
        font=dict(size=13),
    )


def _add_trend(figure: go.Figure, row: int, col: int, x: pd.Series, y: pd.Series, color: str) -> None:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2:
        return
    centered = valid["x"] - valid["x"].mean()
    slope = float((centered * (valid["y"] - valid["y"].mean())).sum() / (centered**2).sum())
    intercept = float(valid["y"].mean() - slope * valid["x"].mean())
    trend_x = [float(valid["x"].min()), float(valid["x"].max())]
    figure.add_trace(
        go.Scatter(
            x=trend_x,
            y=[intercept + slope * value for value in trend_x],
            mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _plot_sleep_metric_row(
    figure: go.Figure,
    frame: pd.DataFrame,
    x_column: str,
    x_label: str,
    row: int,
    hover_columns: list[str],
    hover_prefix: str,
    trend: bool = True,
    x_hover: str | None = None,
) -> None:
    """Scatter one predictor against each sleep metric along one row."""
    for col, (column, name, _, unit, color, _) in enumerate(SLEEP_METRICS, 1):
        valid = frame[[x_column, column, *hover_columns]].dropna(subset=[x_column, column])
        figure.add_trace(
            go.Scatter(
                x=valid[x_column],
                y=valid[column],
                mode="markers",
                marker=dict(color=color, size=7, opacity=0.8),
                customdata=valid[hover_columns],
                hovertemplate=(
                    f"{hover_prefix}{x_hover or x_label + ': %{x:.1f}'}<br>"
                    f"Following {name.lower()}: %{{y:.1f}} {unit}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        if trend:
            _add_trend(figure, row, col, valid[x_column], valid[column], color)
        _add_correlation(figure, row, col, valid[x_column], valid[column])
    _add_row_x_title(figure, row, x_label)


def _no_pairs(figure: go.Figure, text: str) -> go.Figure:
    figure.add_annotation(
        text=text, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
    )
    return figure
