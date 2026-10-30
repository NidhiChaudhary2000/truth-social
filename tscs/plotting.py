"""Consistent, colour-blind-validated matplotlib style for all figures."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# categorical slots in fixed order (validated palette); never cycled past 8
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
DIVERGING = ("#2a78d6", "#e9e8e4", "#e34948")  # cool pole, neutral mid, warm pole


def setup():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9.5, "axes.grid": True, "grid.color": "#e9e8e4",
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.prop_cycle": mpl.cycler(color=C), "font.family": "DejaVu Sans",
    })


def save(fig, name, fig_dir="figs"):
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), bbox_inches="tight")
    return fig
