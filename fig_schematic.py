"""F2 -- PC-LSB pipeline schematic."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import setup, PAL, save
from common import ROOT

setup()
fig, ax = plt.subplots(figsize=(15, 5.2))
ax.set_xlim(0, 15); ax.set_ylim(-0.45, 5.2); ax.axis("off")


def box(x, y, w, h, title, body, col, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=col, ec="none"))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", color=tc, fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.68, body, ha="center", va="top", color=tc, fontsize=8.6, linespacing=1.35)


def arrow(x0, y0, x1, y1, txt=None, col="#52514e"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.5, color=col))
    if txt:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, txt, ha="center", fontsize=8, color=col)


g = "#e9e8e4"
box(0.1, 1.6, 2.0, 2.2, "Acquisition", "deterministic samples\ny = A x\n(time/space decimation\nor k-space lines)", g, "#0b0b0b")
box(2.6, 1.6, 2.3, 2.2, "Anchor (MAC)", "learn pattern d = P y\n(real Fourier transform)\nzero-pad + invert\nx0 = boundary-aware RFT", PAL["mac"])
box(5.4, 2.85, 3.1, 1.95, "Pattern code", "patch f around each output\nsample, contrast-normalised\nf~ = s(f - mean)/||f - mean||\nnearest of K sign-invariant\nanchors  ->  k", PAL["ours"])
box(5.4, 0.35, 3.1, 2.2, "Per-(anchor, phase) ridge", "residual = ||f|| (W_k,phi f~ + b_k s)\nhierarchical shrinkage to\nphase-global model\n+ variance  sigma^2_k,phi\n(closed form, no SGD)", PAL["ours"])
box(9.0, 1.6, 2.4, 2.2, "Frequency gate", "split correction into\nB radial bands,\ngamma_b fit by LS\non validation\n=> val. error <= MAC", "#256abf")
box(11.9, 1.6, 2.9, 2.2, "G-LSB consistency", "x = p + A+(y - A p)\nunique min ||x - p||\ns.t. A x = y\n(samples kept intact,\nnon-expansive to truth)", "#184f95")
arrow(2.1, 2.7, 2.6, 2.7)
arrow(4.9, 3.2, 5.4, 3.6)
arrow(6.95, 2.85, 6.95, 2.55)
arrow(8.5, 1.6, 9.0, 2.3, "Δ (correction)")
arrow(11.4, 2.7, 11.9, 2.7)
ax.text(11.65, 3.95, "p = x0 + Σ γ_b Δ_b", ha="center", fontsize=8.5, color="#52514e")
ax.plot([13.35, 13.35, 3.75], [1.6, 0.05, 0.05], color="#0b0b0b", lw=1.3)
ax.add_patch(FancyArrowPatch((3.75, 0.05), (3.75, 1.6), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#0b0b0b"))
ax.text(10.9, -0.25, "boosting: repeat T stages on the new estimate (x0 <- x)", ha="center", fontsize=8.5)
ax.text(13.35, 4.15, "+ ECG: beat-synchronous pooling\n(BS-LSB) of sub-sample-shifted\nbeats, LOBO self-validated", ha="center",
        fontsize=8.5, color=PAL["ours"], weight="bold")
ax.text(7.5, 5.05, "PC-LSB: Pattern-Conditioned Least-Square Boost", ha="center", fontsize=13, weight="bold")
save(fig, f"{ROOT}/figs/F2_method_schematic.png")
