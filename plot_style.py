"""Shared figure style for the paper: bold black axes, legend top left, no decoration."""
import matplotlib as mpl

BLUE, ORANGE = "tab:blue", "tab:orange"


def apply():
    mpl.rcParams.update({
        "font.size": 14,
        "font.weight": "bold",
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "axes.labelcolor": "black",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "legend.loc": "upper left",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "legend.fontsize": 13,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def bold_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
