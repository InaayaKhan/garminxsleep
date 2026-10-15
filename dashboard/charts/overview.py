"""Charts on the Overview tab: daily duration, 24-hour stages, composition, credit/debit."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.charts.common import (
    AVERAGE_7_COLOR,
    DATE_CHART_RIGHT_LEGEND,
    DATE_CHART_SIDE_MARGINS,
    NAP_BAR_COLOR,
    NIGHT_BAR_COLOR,
    SLEEP_SCORE_PINK,
    TARGET_SLEEP_HOURS,
    ZERO_COLOR,
    _add_no_data_markers,
    _date_axis_limits,
    _linear_trend,
    _pannable_date_view,
    _stacked_bar_top,
)


AVERAGE_28_COLOR = "#4DB88A"


HRV_GREEN = "#4DB88A"


HRV_WEEKLY_GREEN = "#8ED8B8"


CLOCK_BOUNDARY_MINUTE = 0


MENSTRUAL_PHASE_COLORS = {
    1: "#E14E5C",
    2: "#2FD6A3",
    3: "#FFB547",
    4: "#F29BDA",
}


MENSTRUAL_STRIP_PX = 7


def hour_px(figure: go.Figure) -> float:
    """Pixels per hour on the primary y-axis of a figure with fixed geometry."""
    layout = figure.layout
    plot_height = layout.height - layout.margin.t - layout.margin.b
    domain_start, domain_end = layout.yaxis.domain or (0.0, 1.0)
    range_start, range_end = layout.yaxis.range
    return plot_height * (domain_end - domain_start) / (range_end - range_start)


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


def _nice_tick_step(span: float, target_ticks: int = 5) -> float:
    rough_step = max(span / target_ticks, 1.0)
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    for candidate in (1, 2, 5, 10):
        if normalized <= candidate:
            return candidate * magnitude
    return 10 * magnitude


# Cycle-phase colours: ovulation yellow for the balance line, follicular green
# when ahead of the target, menstrual red when behind.
BALANCE_LINE_COLOR = MENSTRUAL_PHASE_COLORS[3]
CREDIT_COLOR = MENSTRUAL_PHASE_COLORS[2]
DEBIT_COLOR = MENSTRUAL_PHASE_COLORS[1]


def _format_signed_duration(hours: float) -> str:
    total_minutes = int(round(abs(float(hours)) * 60))
    sign = "-" if hours < 0 else "+" if hours > 0 else ""
    return f"{sign}{total_minutes // 60}:{total_minutes % 60:02d}"


CREDIT_WINDOW_DAYS = 14


def _signed_duration_ticks(low: float, high: float) -> tuple[list[float], list[str]]:
    """Round tick positions across low..high, labelled as signed HH:MM."""
    step = _nice_tick_step(high - low)
    values: list[float] = []
    value = math.ceil(low / step) * step
    while value <= high + step * 0.01:
        values.append(value)
        value += step
    return values, [_format_signed_duration(item) for item in values]


def credit_debit_figure(
    daily: pd.DataFrame, target_sleep_hours: float = TARGET_SLEEP_HOURS
) -> go.Figure:
    """Sleep balance against a nightly target, in hours of sleep.

    Each day adds ``slept - target`` hours (naps included). The top panel is the
    running total over the selected range; the bottom panel is the balance over
    the last ``CREDIT_WINDOW_DAYS`` days, i.e. how far ahead or behind you are
    right now. Days without any sleep record are left out rather than counted
    as zero.
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
        figure.update_layout(title="Sleep balance", height=600)
        return figure
    frame["daily_balance"] = frame["sleep_hours"] - target_sleep_hours
    frame["cumulative_balance"] = frame["daily_balance"].cumsum()
    frame["recent_balance"] = (
        frame.set_index(pd.to_datetime(frame["date"]))["daily_balance"]
        .rolling(f"{CREDIT_WINDOW_DAYS}D")
        .sum()
        .to_numpy()
    )
    frame["trend"] = _linear_trend(frame["cumulative_balance"])
    x_start, x_end = _date_axis_limits(frame["date"])

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.1,
        subplot_titles=[
            "Running balance over the selected range",
            f"Balance over the last {CREDIT_WINDOW_DAYS} days",
        ],
    )
    customdata = frame[
        ["sleep_hours", "daily_balance", "cumulative_balance", "recent_balance"]
    ]
    hovertemplate = (
        "%{x|%d/%m/%Y}<br>Slept: %{customdata[0]:.2f} h "
        "(%{customdata[1]:+.2f} h vs target)<br>"
        "Running balance: %{customdata[2]:+.1f} h<br>"
        f"Last {CREDIT_WINDOW_DAYS} days: %{{customdata[3]:+.1f}} h<extra></extra>"
    )
    figure.add_trace(
        go.Scatter(
            name="Running balance",
            x=frame["date"],
            y=frame["cumulative_balance"],
            mode="lines",
            line=dict(color=BALANCE_LINE_COLOR, width=1.8),
            customdata=customdata,
            hovertemplate=hovertemplate,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            name="Trend",
            x=frame["date"],
            y=frame["trend"],
            mode="lines",
            line=dict(color=BALANCE_LINE_COLOR, width=2.2, dash="dash"),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            name=f"Last {CREDIT_WINDOW_DAYS} days",
            x=frame["date"],
            y=frame["recent_balance"],
            marker=dict(
                color=[
                    DEBIT_COLOR if value < 0 else CREDIT_COLOR
                    for value in frame["recent_balance"]
                ],
                line=dict(width=0),
            ),
            customdata=customdata,
            hovertemplate=hovertemplate,
        ),
        row=2,
        col=1,
    )
    for row in (1, 2):
        figure.add_hline(y=0, line=dict(color=DEBIT_COLOR, width=1.5), row=row, col=1)

    for row, column in ((1, "cumulative_balance"), (2, "recent_balance")):
        low = min(float(frame[column].min()), 0.0)
        high = max(float(frame[column].max()), 0.0)
        padding = max((high - low) * 0.06, 1.0)
        tick_values, tick_text = _signed_duration_ticks(low - padding, high + padding)
        figure.update_yaxes(
            range=[low - padding, high + padding],
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            gridcolor="#303030",
            zeroline=False,
            row=row,
            col=1,
        )
    figure.update_yaxes(title_text="Hours (+ ahead / − behind)", row=1, col=1)
    figure.update_yaxes(title_text="Hours", row=2, col=1)
    figure.update_xaxes(
        type="date",
        range=[x_start, x_end],
        minallowed=x_start,
        maxallowed=x_end,
        tickformat="%d/%m",
        gridcolor="#303030",
    )
    figure.update_xaxes(title_text="Date", row=2, col=1)
    figure.update_layout(
        title=(
            f"Sleep balance (target {int(target_sleep_hours)}h "
            f"{round(target_sleep_hours % 1 * 60):02d}m)"
        ),
        showlegend=False,
        height=600,
        margin=dict(l=90, r=24, t=90, b=64),
        hovermode="x",
        bargap=0.1,
    )
    return figure
