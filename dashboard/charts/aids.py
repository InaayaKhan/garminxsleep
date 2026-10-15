"""Charts on the Aids tab, and per-aid colours."""

from __future__ import annotations

import json
import zlib
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from dashboard.config import DATA_ROOT

from dashboard.charts.common import (
    DATE_CHART_SIDE_MARGINS,
    NAP_BAR_COLOR,
    NIGHT_BAR_COLOR,
    SLEEP_SCORE_PINK,
    ZERO_COLOR,
    _add_no_data_markers,
    _pannable_date_view,
    _stacked_bar_top,
)


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
