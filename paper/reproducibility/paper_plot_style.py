"""Canonical visual style and stable TSFM identities for paper figures.

The palette and stroke vocabulary are anchored to
``fig7_multimodel_pareto_overlay``.  Plot generators should import model
appearance from here instead of defining local colors, markers, or dashes.
"""

from __future__ import annotations

from matplotlib.lines import Line2D


FIGURE_WIDTH = 7.8
INK = "#263746"
MUTED = "#667085"
AXIS = "#8B99A6"
GRID = "#DDE3E8"

PAPER_MPL_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": "#52616B",
    "ytick.color": "#52616B",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "grid.color": GRID,
    "grid.linewidth": 0.65,
    "grid.alpha": 0.9,
    "legend.fontsize": 7.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

MODEL_ORDER = [
    "Chronos2-Base",
    "Chronos2-Small",
    "Chronos2-Synth",
    "ChronosBolt-Base",
    "FlowState-R1",
    "Moirai2-Small",
    "PatchTST-FM-R1",
    "Sundial-Base-128M",
    "TiRex2",
    "TimesFM2.5-200M",
    "Toto-2.0-313m",
]

MODEL_DISPLAY = {
    "Chronos2-Base": "Chronos2 Base",
    "Chronos2-Small": "Chronos2 Small",
    "Chronos2-Synth": "Chronos2 Synth",
    "ChronosBolt-Base": "Chronos-Bolt Base",
    "FlowState-R1": "FlowState R1",
    "Moirai2-Small": "Moirai 2 Small",
    "PatchTST-FM-R1": "PatchTST-FM",
    "Sundial-Base-128M": "Sundial 128M",
    "TiRex2": "TiRex 2",
    "TimesFM2.5-200M": "TimesFM 2.5",
    "Toto-2.0-313m": "Toto 2.0",
}

# These colors and dashes are the Figure 7 identities.
MODEL_COLORS = {
    "Chronos2-Base": "#0072B2",
    "Chronos2-Small": "#D55E00",
    "Chronos2-Synth": "#56B4E9",
    "ChronosBolt-Base": "#E69F00",
    "FlowState-R1": "#009E73",
    "Moirai2-Small": "#CC79A7",
    "PatchTST-FM-R1": "#6F4E9C",
    "Sundial-Base-128M": "#A97500",
    "TiRex2": "#4C566A",
    "TimesFM2.5-200M": "#008E9B",
    "Toto-2.0-313m": "#C83E73",
}

MODEL_LINE_STYLES = {
    "Chronos2-Base": "-",
    "Chronos2-Small": "-",
    "Chronos2-Synth": "--",
    "ChronosBolt-Base": "-.",
    "FlowState-R1": "-",
    "Moirai2-Small": "--",
    "PatchTST-FM-R1": "-.",
    "Sundial-Base-128M": ":",
    "TiRex2": (0, (5, 2)),
    "TimesFM2.5-200M": (0, (3, 1, 1, 1)),
    "Toto-2.0-313m": (0, (1, 1)),
}

# Markers are model identities whenever a plot uses model-specific points.
MODEL_MARKERS = {
    "Chronos2-Base": "o",
    "Chronos2-Small": "s",
    "Chronos2-Synth": "^",
    "ChronosBolt-Base": "D",
    "FlowState-R1": "v",
    "Moirai2-Small": "P",
    "PatchTST-FM-R1": "X",
    "Sundial-Base-128M": "<",
    "TiRex2": ">",
    "TimesFM2.5-200M": "h",
    "Toto-2.0-313m": "d",
}


def model_plot_kwargs(model: str, *, marker: bool = True) -> dict:
    """Return the canonical Matplotlib kwargs for one model."""
    kwargs = {
        "color": MODEL_COLORS[model],
        "ls": MODEL_LINE_STYLES[model],
        "solid_capstyle": "round",
        "dash_capstyle": "round",
    }
    if marker:
        kwargs["marker"] = MODEL_MARKERS[model]
    return kwargs


def model_legend_handles(models=MODEL_ORDER, *, markers: bool = True) -> list[Line2D]:
    return [
        Line2D(
            [0], [0],
            lw=1.5,
            ms=4.5,
            marker=MODEL_MARKERS[model] if markers else None,
            label=MODEL_DISPLAY[model],
            **model_plot_kwargs(model, marker=False),
        )
        for model in models
    ]


def style_axes(ax, *, grid_axis: str = "y") -> None:
    """Apply the Figure 7 axis treatment."""
    ax.grid(True, axis=grid_axis)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(width=0.8)


def style_legend(legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(GRID)
    frame.set_linewidth(0.7)


def add_figure_title(fig, title: str, *, y: float = 0.985) -> None:
    fig.suptitle(
        title,
        x=0.5,
        y=y,
        ha="center",
        color=INK,
        fontsize=11,
        fontweight="semibold",
    )
