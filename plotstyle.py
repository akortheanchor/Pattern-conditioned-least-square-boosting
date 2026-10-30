"""Shared figure style (validated reference palette; recessive axes; thin marks)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = {
    "ours": "#2a78d6",   # slot 1 blue   -> proposed PC-LSB
    "mac": "#eb6834",    # slot 2 orange -> MAC (the PDF's method)
    "deep": "#1baf7a",   # slot 3 aqua   -> deep-learning baselines
    "cs": "#4a3aa7",     # slot 7 violet -> compressed-sensing baselines
    "base": "#8a8985",   # neutral gray  -> classical baselines
    "gt": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e4e3df",
    "surface": "#fcfcfb",
}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup():
    plt.rcParams.update({
        "figure.facecolor": PAL["surface"], "axes.facecolor": PAL["surface"],
        "savefig.facecolor": PAL["surface"], "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.edgecolor": "#b9b8b3", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": PAL["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "xtick.color": PAL["ink2"], "ytick.color": PAL["ink2"], "axes.labelcolor": PAL["ink2"],
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })


def method_color(name):
    n = name.lower()
    if n.startswith("pc-lsb+bs"):
        return "#0d366b"
    if n.startswith("pc-lsb") or n.startswith("vdsr + pc-lsb"):
        return PAL["ours"]
    if n.startswith("mac"):
        return PAL["mac"]
    if n in ("srcnn", "vdsr", "srcnn+dc", "vdsr+dc", "cnn"):
        return PAL["deep"]
    if n.startswith("cs-") or n.startswith("omp"):
        return PAL["cs"]
    return PAL["base"]


def save(fig, path, dpi=160):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
