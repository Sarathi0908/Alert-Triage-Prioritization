"""
plots.py  —  shared chart styling for the Alert Triage (#01) charts.

The model run and the EDA run previously each carried their own copy of the same
palette + rcParams + `style()` helper; both now import from here so every PNG in
outputs/ comes out of one visual system.

Headless by construction: the Agg backend is selected on import, before pyplot,
so these modules run under CI and over SSH with no display.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- palette -----------------------------------------------------------------
BLUE = "#2a78d6"      # the model / primary series
RED = "#d03b3b"       # attacks, priority
AQUA = "#1baf7a"      # secondary series
INK = "#0b0b0b"       # titles
INK2 = "#52514e"      # axis labels, annotations
MUTED = "#898781"     # ticks, reference lines
SURF = "#fcfcfb"      # figure/axes background
EDGE = "#c3c2b7"
GRID = "#e1e0d9"

# Sequential ramp (confusion matrix, count heatmaps).
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging ramp (correlation, z-scored fingerprints) — blue low, red high.
DIV = ["#2a78d6", "#f0efec", "#e34948"]

SEQ_CMAP = LinearSegmentedColormap.from_list("bhairava_seq", SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list("bhairava_div", DIV)

RC = {
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "font.size": 9,
    "axes.edgecolor": EDGE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_style() -> None:
    """Install the house rcParams. Call once at module import in a chart script."""
    plt.rcParams.update(RC)


def style(ax, t: str | None = None, xl: str | None = None, yl: str | None = None) -> None:
    """Title/label/grid an axes consistently."""
    if t:
        ax.set_title(t, fontsize=12, fontweight="bold", color=INK, pad=12)
    if xl:
        ax.set_xlabel(xl, color=INK2)
    if yl:
        ax.set_ylabel(yl, color=INK2)
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def save(fig, path, dpi: int = 150) -> None:
    """tight_layout + write + close, so no figure is left open across a long run."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def signed_colors(values) -> list[str]:
    """Red for positive contributions, blue for negative (SHAP-local bars)."""
    return ["#e34948" if v > 0 else BLUE for v in values]


apply_style()
