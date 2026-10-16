"""Shared Streamlit presentation: chart defaults, linked panning, formatting."""

from __future__ import annotations

import math
import re

import streamlit as st


def format_duration(minutes: float) -> str:
    if not math.isfinite(minutes):
        return "No data"
    hours = int(minutes // 60)
    remainder = int(round(minutes % 60))
    return f"{hours}h {remainder:02d}m"


# Streamlit's own function, even if st.plotly_chart was already replaced by an
# earlier run, so the defaults below are applied exactly once per chart.
_streamlit_plotly_chart = getattr(st.plotly_chart, "_original", st.plotly_chart)
LEGEND_ENTRIES_PER_ROW = 6
LEGEND_ROW_HEIGHT = 24


# A chart title drawn by Streamlit (so it can carry an info icon) is pulled
# down into the chart's top margin, where Plotly's own title would have been.
_INFO_TITLE_CSS = """
<style>
[class*="st-key-chart_info_title_"] {
    margin-bottom: -2.9rem; position: relative; z-index: 5;
    width: fit-content; padding-left: 0.35rem;
}
[class*="st-key-chart_info_title_"] p { font-size: 1.0rem; }
</style>
"""


def _chart_info_title(figure, info: str) -> None:
    """Replace the figure's title with the same title plus a hover info icon."""
    title = re.sub(r"<[^>]+>", "", figure.layout.title.text or "")
    figure.update_layout(title_text="")
    st.markdown(_INFO_TITLE_CSS, unsafe_allow_html=True)
    key = "chart_info_title_" + re.sub(r"\W+", "_", title.lower()).strip("_")
    with st.container(key=key):
        st.markdown(f"**{title}**", help=info)


def _toolbar_only_plotly_chart(figure, *args, config=None, info=None, **kwargs):
    """Disable accidental chart gestures while retaining Plotly's toolbar tools.

    Legends sit along the bottom edge of every chart so wrapped rows grow into
    the bottom margin instead of colliding with the title. The chart grows by
    roughly the legend's height so the plot itself is not squeezed, unless the
    figure opts out with ``meta=dict(fixed_height=True)``.

    ``info`` adds a grey info icon after the chart title that shows the text on
    hover, instead of a caption under the chart.
    """
    if info:
        _chart_info_title(figure, info)
    fixed_height = (figure.layout.meta or {}).get("fixed_height", False)
    entries = {
        trace.legendgroup or trace.name
        for trace in figure.data
        if trace.showlegend is not False and trace.name
    }
    # A one-item legend says nothing the title doesn't (pies label their slices).
    if len(entries) <= 1 and not any(trace.type == "pie" for trace in figure.data):
        figure.update_layout(showlegend=False)
    if (
        figure.layout.height
        and figure.layout.showlegend is not False
        and not fixed_height
    ):
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
            # Compact row: each entry only as wide as its text.
            entrywidth=0,
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



def install_chart_defaults() -> None:
    """Route every ``st.plotly_chart`` call through the shared chart defaults."""
    _toolbar_only_plotly_chart._original = _streamlit_plotly_chart
    st.plotly_chart = _toolbar_only_plotly_chart

# Plotly charts render in the main page, and st.iframe embeds HTML strings with
# same-origin access, so this script can reach them: panning one pannable chart
# pans the others too.
_LINK_PANNING_SCRIPT = """
<script>
const page = window.parent;
// This frame only hosts the script: hide it so it leaves no gap in the page.
const host = window.frameElement && window.frameElement.closest('[data-testid="stElementContainer"]');
if (host) host.style.display = "none";
// Charts pan together only within a group (meta.pan_group): the date charts
// by default, the week-numbered activity calendars as their own group.
const panGroup = (chart) => chart.layout.meta.pan_group || "dates";
function pannableCharts() {
  return [...page.document.querySelectorAll(".js-plotly-plot")]
    .filter((chart) => chart.layout && chart.layout.meta && chart.layout.meta.pan);
}
function linkPannableCharts() {
  for (const chart of pannableCharts()) {
    if (chart._panLinked) continue;
    chart._panLinked = true;
    chart.on("plotly_relayout", (change) => {
      if (page._panSyncing) return;
      const start = change["xaxis.range[0]"] ?? (change["xaxis.range"] || [])[0];
      const end = change["xaxis.range[1]"] ?? (change["xaxis.range"] || [])[1];
      if (start === undefined || end === undefined) return;
      page._panSyncing = true;
      Promise.all(
        pannableCharts()
          .filter((other) => other !== chart && panGroup(other) === panGroup(chart))
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
    # st.iframe needs a positive height; the script then hides its own frame.
    st.iframe(_LINK_PANNING_SCRIPT, height=1, alt="Links panning across date charts")
