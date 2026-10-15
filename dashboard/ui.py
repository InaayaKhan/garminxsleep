"""Shared Streamlit presentation: chart defaults, linked panning, formatting."""

from __future__ import annotations

import math

import streamlit as st
import streamlit.components.v1 as components


def format_duration(minutes: float) -> str:
    if not math.isfinite(minutes):
        return "No data"
    hours = int(minutes // 60)
    remainder = int(round(minutes % 60))
    return f"{hours}h {remainder:02d}m"


# Streamlit's own function, even if st.plotly_chart was already replaced by an
# earlier run, so the defaults below are applied exactly once per chart.
_streamlit_plotly_chart = getattr(st.plotly_chart, "_original", st.plotly_chart)
LEGEND_ENTRIES_PER_ROW = 3
LEGEND_ROW_HEIGHT = 24


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



def install_chart_defaults() -> None:
    """Route every ``st.plotly_chart`` call through the shared chart defaults."""
    _toolbar_only_plotly_chart._original = _streamlit_plotly_chart
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
