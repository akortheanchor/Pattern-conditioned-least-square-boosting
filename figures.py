"""All result figures (PNG). Each function is skipped if its inputs are missing."""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import setup, PAL, save, method_color, SEQ
from common import ROOT, NAMES, ND

FIG = f"{ROOT}/figs"
setup()
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f3b3a6", "#f0efec", "#9ec5f4", "#184f95"])
DSORDER = ["ecg", "eeg", "mri", "cxr", "us"]
KEY = ["MAC (RFT)", "Cubic spline", "CS-TV", "OMP-DCT", "VDSR", "PC-LSB"]


def cfg_label(c):
    ds, r = c.rsplit("_r", 1)
    nd = ND[ds]
    return f"{NAMES[ds]}\n{100 / int(r) ** nd:.4g}% samples"


def bench():
    fs = sorted(glob.glob(f"{ROOT}/results/bench_*_r*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None


# ------------------------------------------------------------------ F3 heatmap
def fig_heatmap():
    p = f"{ROOT}/results/summary_means.csv"
    if not os.path.exists(p):
        return
    M = pd.read_csv(p, index_col=0)
    M = M.loc[sorted(M.index, key=lambda c: (DSORDER.index(c.rsplit("_r", 1)[0]), int(c.rsplit("_r", 1)[1])))]
    D = M.sub(M["MAC (RFT)"], axis=0)
    order = D.mean().sort_values(ascending=False).index
    D = D[order]; M = M[order]
    fig, ax = plt.subplots(figsize=(1.0 * len(order) + 2.5, 0.62 * len(D) + 2))
    lim = np.nanpercentile(np.abs(D.values), 97)
    im = ax.imshow(D.values, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            v = D.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{M.values[i, j]:.2f}\n({v:+.2f})", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * lim else "#0b0b0b")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_yticks(range(len(D))); ax.set_yticklabels([cfg_label(c) for c in D.index], fontsize=8)
    ax.grid(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.025); cb.set_label("Δ dB vs MAC (RFT)")
    ax.set_title("Mean reconstruction quality (PSNR for images, SNR for biosignals) and gain over MAC")
    save(fig, f"{FIG}/F3_main_results_heatmap.png")


# ------------------------------------------------------------------ F4 CD diagram
def fig_cd():
    p = f"{ROOT}/results/stats_summary.json"
    if not os.path.exists(p):
        return
    S = json.load(open(p))["global"]
    ar = pd.Series(S["avg_rank"]).sort_values()
    k, cd = S["k"], S["CD"]
    half_ = (k + 1) // 2
    ytop = 0.6 + half_ * 0.42 + 0.9
    fig, ax = plt.subplots(figsize=(12, 0.27 * k + 1.6))
    ax.set_xlim(0.5, k + 0.5); ax.set_ylim(-0.8, ytop); ax.axis("off")
    ax.hlines(0, 1, k, color="#52514e", lw=1)
    for t in range(1, k + 1):
        ax.vlines(t, -0.1, 0.1, color="#52514e", lw=1); ax.text(t, -0.45, str(t), ha="center", fontsize=8)
    half = (k + 1) // 2
    for i, (name, r) in enumerate(ar.items()):
        left = i < half
        y = 0.6 + (i if left else k - 1 - i) * 0.42
        xe = 0.6 if left else k + 0.4
        c = method_color(name)
        ax.plot([r, r, xe], [0, y, y], color=c, lw=1.6 if name in ("PC-LSB", "MAC (RFT)") else 1.0)
        ax.plot(r, 0, "o", color=c, ms=6)
        ax.text(xe + (-0.05 if left else 0.05), y, f"{name} ({r:.2f})", ha="right" if left else "left", va="center",
                fontsize=8.5, color="#0b0b0b", weight="bold" if name == "PC-LSB" else "normal")
    ax.plot([1, 1 + cd], [ytop - 0.45] * 2, color="#0b0b0b", lw=2)
    ax.text(1 + cd / 2, ytop - 0.32, f"CD = {cd:.2f} (Nemenyi, α=0.05)", ha="center", fontsize=8.5)
    # cliques: methods whose ranks differ < CD from best
    ax.set_title(f"Average rank over {S['N']} dataset×rate configurations "
                 f"(Friedman χ²={S['friedman_chi2']:.1f}, p={S['friedman_p']:.1e})", fontsize=10.5)
    save(fig, f"{FIG}/F4_critical_difference.png")


# ------------------------------------------------------------------ F5 rate curves
def fig_rates():
    df = bench()
    if df is None:
        return
    dss = [d for d in DSORDER if d in df.ds.unique()]
    fig, axs = plt.subplots(1, len(dss), figsize=(4.0 * len(dss), 3.8))
    axs = np.atleast_1d(axs)
    for ax, ds in zip(axs, dss):
        d = df[df.ds == ds]
        for m in KEY:
            dm = d[d.method == m]
            if dm.empty:
                continue
            g = dm.groupby("rate")["q"]
            mu = g.mean(); se = g.std() / np.sqrt(g.count())
            lw = 2.6 if m == "PC-LSB" else 1.6
            ax.errorbar(mu.index * 100, mu.values, yerr=1.96 * se.values, color=method_color(m) if m not in ("Cubic spline", "OMP-DCT") else ("#8a8985" if m == "Cubic spline" else "#e87ba4"),
                        lw=lw, marker="o", ms=5, capsize=2, label=m)
        if "PC-LSB+BS" in d.method.unique():
            g = d[d.method == "PC-LSB+BS"].groupby("rate")["q"].mean()
            ax.plot(g.index * 100, g.values, color="#0d366b", ls="--", marker="s", ms=5, label="PC-LSB+BS")
        ax.set_xscale("log"); ax.set_xlabel("sampling rate (%)")
        ax.set_ylabel("SNR (dB)" if ND[ds] == 1 else "PSNR (dB)")
        ax.set_title(NAMES[ds])
        rates = sorted(d.rate.unique()); ax.set_xticks([r * 100 for r in rates]); ax.set_xticklabels([f"{r*100:.3g}" for r in rates])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(min(rates) * 100 / 1.25, max(rates) * 100 * 1.25)
    axs[0].legend(fontsize=8, loc="upper left")
    save(fig, f"{FIG}/F5_rate_distortion.png")


# ------------------------------------------------------------------ F6 forest plots
def fig_forest():
    fs = sorted(glob.glob(f"{ROOT}/results/stats_*_r*.csv"))
    fs = [f for f in fs if not f.endswith("_BS.csv")]
    if not fs:
        return
    sel = {}
    for f in fs:
        ds, r = os.path.basename(f)[6:-4].rsplit("_r", 1)
        if ds not in sel or (ND[ds] == 1 and int(r) == 4) or (ND[ds] == 2 and int(r) == 2):
            sel[ds] = (f, int(r))
    dss = [d for d in DSORDER if d in sel]
    fig, axs = plt.subplots(1, len(dss), figsize=(3.3 * len(dss), 5.2), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, ds in zip(axs, dss):
        f, r = sel[ds]
        ours = "PC-LSB"
        if ds == "ecg" and os.path.exists(f.replace(".csv", "_BS.csv")):
            f = f.replace(".csv", "_BS.csv"); ours = "PC-LSB+BS"
        s = pd.read_csv(f)
        s = s[~s.baseline.isin(["Zero-fill", "Nearest"])].sort_values("mean_diff")
        y = np.arange(len(s))
        col = [PAL["ours"] if (p < 0.05 and m > 0) else ("#e34948" if (p < 0.05 and m < 0) else "#8a8985")
               for p, m in zip(s.p_holm, s.mean_diff)]
        ax.hlines(y, s.ci_lo, s.ci_hi, color=col, lw=2)
        ax.scatter(s.mean_diff, y, color=col, s=26, zorder=3)
        ax.axvline(0, color="#0b0b0b", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(s.baseline, fontsize=8)
        ax.set_xlabel(f"Δ dB ({ours} − baseline)")
        ax.set_title(f"{NAMES[ds]}\n{100 / r ** ND[ds]:.3g}% samples", fontsize=9.5)
    fig.suptitle("Paired differences with 95% bootstrap CIs — blue: ours significantly better, red: significantly worse, gray: n.s. (Holm-corrected Wilcoxon, α=0.05); trivial zero-fill/nearest omitted", fontsize=9.5)
    save(fig, f"{FIG}/F6_forest_paired_differences.png")


# ------------------------------------------------------------------ F7 qualitative
def fig_qual():
    for ds in ["mri", "cxr", "us"]:
        fs = sorted(glob.glob(f"{ROOT}/results/recon_{ds}_r*.npz"))
        if not fs:
            continue
        f = fs[-1]; r = int(f.split("_r")[-1][:-4])
        E = np.load(f)
        cols = ["gt", "MAC (RFT)", "Cubic spline", "CS-TV", "VDSR", "PC-LSB"]
        n = min(3, len(E["gt"]))
        fig, axs = plt.subplots(2 * n, len(cols) + 1, figsize=(2.15 * (len(cols) + 1), 4.3 * n))
        for i in range(n):
            gt = E["gt"][i]
            for j, c in enumerate(cols):
                im = E[c][i] if c != "gt" else gt
                axs[2 * i, j].imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
                if c != "gt":
                    ps = 10 * np.log10(1 / np.mean((gt - im) ** 2))
                    axs[2 * i, j].set_title(f"{c}\n{ps:.2f} dB", fontsize=8.5)
                    axs[2 * i + 1, j].imshow(np.abs(gt - im), cmap="magma", vmin=0, vmax=0.25)
                else:
                    axs[2 * i, j].set_title("ground truth", fontsize=8.5)
                    lo = np.zeros_like(gt); lo[::r, ::r] = gt[::r, ::r]
                    axs[2 * i + 1, j].imshow(np.kron(gt[::r, ::r], np.ones((r, r))), cmap="gray", vmin=0, vmax=1)
                    axs[2 * i + 1, j].set_title(f"{100/r**2:.3g}% samples", fontsize=8)
            axs[2 * i, -1].imshow(E["sigma"][i], cmap="magma", vmin=0, vmax=np.percentile(E["sigma"][i], 99.5))
            axs[2 * i, -1].set_title("PC-LSB predicted σ", fontsize=8.5)
            axs[2 * i + 1, -1].imshow(np.abs(gt - E["PC-LSB"][i]), cmap="magma", vmin=0, vmax=np.percentile(E["sigma"][i], 99.5) * 1.6)
            axs[2 * i + 1, -1].set_title("PC-LSB |error| (same scale)", fontsize=8)
        for a in axs.ravel():
            a.axis("off")
        fig.suptitle(f"{NAMES[ds]} — reconstructions (top) and |error| maps (bottom)", fontsize=11)
        save(fig, f"{FIG}/F7_qualitative_{ds}.png", dpi=130)
    for ds in ["ecg", "eeg"]:
        fs = sorted(glob.glob(f"{ROOT}/results/recon_{ds}_r*.npz"))
        if not fs:
            continue
        f = [x for x in fs if x.endswith("_r4.npz")] or fs
        E = np.load(f[0]); r = int(f[0].split("_r")[-1][:-4])
        fig, axs = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
        L = 400
        for i, ax in enumerate(axs):
            gt = E["gt"][i][:L]; t = np.arange(L)
            ax.plot(t, gt, color=PAL["gt"], lw=2.4, label="ground truth")
            for c, col in [("MAC (RFT)", PAL["mac"]), ("VDSR", PAL["deep"]), ("PC-LSB", PAL["ours"])]:
                ax.plot(t, E[c][i][:L], color=col, lw=1.3, label=c)
            if "PC-LSB+BS" in E.files:
                ax.plot(t, E["PC-LSB+BS"][i][:L], color=PAL["ours"], lw=1.3, ls="--", label="PC-LSB+BS")
            ax.plot(t[::r], gt[::r], "o", color="k", ms=3, label=f"{100/r:.0f}% samples")
            ax.set_ylabel("z-score")
        axs[0].legend(ncol=6, fontsize=8, loc="upper right")
        axs[-1].set_xlabel("sample (fine grid)")
        fig.suptitle(f"{NAMES[ds]}: reconstructions from {100/r:.0f}% samples", fontsize=11)
        save(fig, f"{FIG}/F7_qualitative_{ds}.png", dpi=130)


# ------------------------------------------------------------------ F8 k-space
def fig_kspace():
    fs = sorted(glob.glob(f"{ROOT}/results/kspace_*_a*.csv"))
    if not fs:
        return
    df = pd.concat([pd.read_csv(f) for f in fs])
    accs = sorted(df.accel.unique())
    masks = ["lowpass", "equispaced", "vardens"]
    meth = df.groupby("method")["q"].mean().sort_values().index.tolist()
    fig, axs = plt.subplots(len(accs), 2, figsize=(15, 4.2 * len(accs)), squeeze=False)
    mc = {"lowpass": PAL["mac"], "equispaced": "#8a8985", "vardens": PAL["cs"]}
    for ai, acc in enumerate(accs):
        d = df[df.accel == acc]
        for k, metric, lab in [(0, "q", "PSNR (dB)"), (1, "cls_ok", "tumour-type accuracy on reconstruction")]:
            ax = axs[ai, k]
            w = 0.27
            for mi, mk in enumerate(masks):
                if not (d["mask"] == mk).any():
                    continue
                g = d[d["mask"] == mk].groupby("method")[metric]
                mu = g.mean().reindex(meth); se = (g.std() / np.sqrt(g.count())).reindex(meth)
                ax.barh(np.arange(len(meth)) + (mi - 1) * w, mu.values, height=w - 0.03, xerr=1.96 * se.values,
                        color=mc[mk], label={"lowpass": "deterministic low-pass (slides' MAC setting)",
                                             "equispaced": "deterministic equispaced + centre",
                                             "vardens": "variable-density random (CS)"}[mk], error_kw=dict(lw=0.8))
            ax.set_yticks(range(len(meth))); ax.set_yticklabels(meth)
            ax.set_xlabel(lab)
            if metric == "q":
                lo = d.groupby(["mask", "method"])["q"].mean().min() - 1
                ax.set_xlim(lo, None)
            else:
                gt = pd.read_csv(f"{ROOT}/results/bench_mri_r2.csv")["gt_cls_ok"].iloc[0]
                ax.axvline(gt, color="k", ls=":", lw=1); ax.text(gt, -0.9, " fully sampled", fontsize=8)
                ax.set_xlim(0.3, max(0.7, gt + 0.05))
            ax.set_title(f"Accelerated MRI, R = {acc}: {lab.split(' (')[0]}")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{FIG}/F8_kspace_sampling_patterns.png", dpi=160, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ F9 uncertainty
def fig_unc():
    fs = sorted(glob.glob(f"{ROOT}/results/unc_*_r*.csv"))
    if not fs:
        return
    U = pd.concat([pd.read_csv(f) for f in fs])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    # (a) reliability from saved example images (pooled pixels)
    for ds, col in zip(["mri", "cxr", "us", "ecg", "eeg"], [PAL["ours"], PAL["mac"], PAL["deep"], "#e87ba4", PAL["cs"]]):
        fsr = sorted(glob.glob(f"{ROOT}/results/recon_{ds}_r*.npz"))
        if not fsr:
            continue
        E = np.load(fsr[0]); r = int(fsr[0].split("_r")[-1][:-4])
        e = np.abs(E["gt"] - E["PC-LSB"]).ravel(); s = E["sigma"].ravel()
        m = s > 0; e, s = e[m], s[m]
        qs = np.quantile(s, np.linspace(0, 1, 11))
        bx, by = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            mm = (s >= a) & (s <= b)
            bx.append(s[mm].mean()); by.append(np.sqrt((e[mm] ** 2).mean()))
        axs[0].plot(bx, by, "o-", color=col, label=NAMES[ds], ms=5)
        # (b) sparsification, per signal/image then averaged (AUSE-style)
        fr = np.linspace(0, 0.95, 20)
        SP, OR = [], []
        for gi, xi, si in zip(E["gt"], E["PC-LSB"], E["sigma"]):
            ei = np.abs(gi - xi).ravel(); si = si.ravel(); mm = si > 0; ei, si = ei[mm], si[mm]
            o = np.argsort(-si); oe = np.argsort(-ei)
            sp = np.array([np.sqrt(np.mean(ei[o][int(f * len(ei)):] ** 2)) for f in fr])
            orc = np.array([np.sqrt(np.mean(ei[oe][int(f * len(ei)):] ** 2)) for f in fr])
            SP.append(sp / sp[0]); OR.append(orc / orc[0])
        axs[1].plot(fr * 100, np.mean(SP, 0), color=col, lw=2, label=NAMES[ds])
        axs[1].plot(fr * 100, np.mean(OR, 0), color=col, lw=1, ls=":")
    lim = axs[0].get_xlim()[1]
    axs[0].plot([0, lim], [0, lim], color="k", ls=":", lw=1, label="ideal")
    axs[0].set_xlabel("predicted σ (bin mean)"); axs[0].set_ylabel("observed RMSE")
    axs[0].set_title("Reliability of PC-LSB uncertainty"); axs[0].legend(fontsize=8)
    axs[1].set_xlabel("% of most-uncertain samples removed"); axs[1].set_ylabel("remaining RMSE (relative)")
    axs[1].set_title("Sparsification (solid) vs oracle (dotted)"); axs[1].legend(fontsize=8)
    dss = [d for d in DSORDER if d in U.ds.unique()]
    data = [U[U.ds == d].spearman.dropna().values for d in dss]
    bp = axs[2].boxplot(data, tick_labels=[NAMES[d].split(" (")[0] for d in dss], patch_artist=True, widths=0.5, medianprops=dict(color="#0b0b0b"))
    for b in bp["boxes"]:
        b.set_facecolor(SEQ[2]); b.set_edgecolor(SEQ[5])
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].set_ylabel("Spearman ρ (σ vs |error|) per test unit"); axs[2].set_title("Error-ranking ability of σ")
    plt.setp(axs[2].get_xticklabels(), rotation=20)
    save(fig, f"{FIG}/F9_uncertainty_calibration.png")


# ------------------------------------------------------------------ F10 runtime
def fig_runtime():
    df = bench()
    if df is None:
        return
    d = df[(df.ds == "mri") & (df.r == 2)] if ((df.ds == "mri") & (df.r == 2)).any() else df
    g = d.groupby("method")[["q", "time"]].mean()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for m, row in g.iterrows():
        ax.scatter(row.time * 1000, row.q, s=60, color=method_color(m), zorder=3)
        ax.annotate(m, (row.time * 1000, row.q), xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("CPU inference time per image (ms, log)"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Accuracy vs. inference cost (Brain MRI, 25% samples, 2-core CPU)")
    save(fig, f"{FIG}/F10_runtime_accuracy.png")


# ------------------------------------------------------------------ F11 downstream
def fig_downstream():
    df = bench()
    if df is None:
        return
    panels = []
    if "r_f1" in df:
        panels.append(("ecg", "r_f1", "R-peak detection F1 (±150 ms)", True))
    if "bp_err" in df:
        panels.append(("eeg", "bp_err", "relative band-power error (lower=better)", False))
    if "cls_ok" in df:
        panels.append(("mri", "cls_ok", "tumour-type accuracy", True))
    if not panels:
        return
    fig, axs = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4))
    axs = np.atleast_1d(axs)
    for ax, (ds, col, lab, hib) in zip(axs, panels):
        d = df[(df.ds == ds) & df[col].notna()]
        meths = ["MAC (RFT)", "Cubic spline", "CS-TV", "VDSR", "PC-LSB"] + (["PC-LSB+BS"] if ds == "ecg" else [])
        rates = sorted(d.rate.unique(), reverse=True)
        w = 0.8 / len(meths)
        for mi, m in enumerate(meths):
            vals = [d[(d.method == m) & (d.rate == rt)][col].mean() for rt in rates]
            ax.bar(np.arange(len(rates)) + (mi - len(meths) / 2 + 0.5) * w, vals, w * 0.92,
                   color=method_color(m) if m not in ("Cubic spline",) else "#b9b8b3", label=m,
                   hatch="//" if m == "PC-LSB+BS" else None, edgecolor="white" if m != "PC-LSB+BS" else PAL["ours"])
        gt = d[f"gt_{col}"].iloc[0] if f"gt_{col}" in d else None
        if gt is not None:
            ax.axhline(gt, color="k", ls=":", lw=1, label="fully sampled")
        ax.set_xticks(range(len(rates))); ax.set_xticklabels([f"{rt*100:.3g}%" for rt in rates])
        ax.set_xlabel("sampling rate"); ax.set_title(f"{NAMES[ds]}: {lab}", fontsize=10)
        if hib and col != "r_f1":
            ax.set_ylim(max(0, d[col].mean() - 0.4), None)
        if col == "r_f1":
            ax.set_ylim(0.5, 1.0)
    axs[0].legend(fontsize=7.5)
    save(fig, f"{FIG}/F11_downstream_clinical_tasks.png")


# ------------------------------------------------------------------ F12 ablation / noise / ood / data-efficiency
def fig_extras():
    fs = sorted(glob.glob(f"{ROOT}/results/ablation_*.csv"))
    if fs:
        fig, axs = plt.subplots(1, len(fs), figsize=(6.8 * len(fs), 4.2))
        axs = np.atleast_1d(axs)
        for ax, f in zip(axs, fs):
            d = pd.read_csv(f); ds, r = d.ds.iloc[0], d.r.iloc[0]
            t = d.pivot_table(index="idx", columns="variant", values="q")
            full = t["PC-LSB (full)"]
            D = t.drop(columns=["PC-LSB (full)"]).sub(full, axis=0)   # variant - full (paired)
            mu = D.mean().sort_values(); se = D.std() / np.sqrt(len(D))
            col = [PAL["mac"] if v.startswith("MAC") else (SEQ[2] if mu[v] < 0 else PAL["deep"]) for v in mu.index]
            ax.barh(range(len(mu)), mu.values, xerr=1.96 * se[mu.index].values, color=col, error_kw=dict(lw=1))
            ax.axvline(0, color="#0b0b0b", lw=0.8)
            ax.set_yticks(range(len(mu))); ax.set_yticklabels(mu.index, fontsize=8.5)
            ax.set_xlabel("Δ dB vs full PC-LSB (paired, 95% CI)")
            ax.set_title(f"Ablation — {NAMES[ds]}, {100 / r ** ND[ds]:.3g}% samples (full = {full.mean():.2f} dB)", fontsize=10)
        save(fig, f"{FIG}/F12_ablation.png")
    fs = sorted(glob.glob(f"{ROOT}/results/noise_*.csv"))
    if fs:
        fig, axs = plt.subplots(1, len(fs), figsize=(6 * len(fs), 4.2))
        axs = np.atleast_1d(axs)
        for ax, f in zip(axs, fs):
            d = pd.read_csv(f); ds, r = d.ds.iloc[0], d.r.iloc[0]
            d["snrlab"] = d.meas_snr.replace(np.inf, 60)
            for m, g in d.groupby("method"):
                mu = g.groupby("snrlab")["q"].mean()
                ax.plot(mu.index, mu.values, "o-", label=m, color=method_color(m) if "clean" not in m else "#86b6ef",
                        lw=2.4 if "noise-aware" in m else 1.5)
            ax.set_xticks([20, 30, 40, 60]); ax.set_xticklabels(["20", "30", "40", "∞"])
            ax.set_xlabel("measurement SNR (dB)"); ax.set_ylabel("dB")
            ax.set_title(f"Noise robustness — {NAMES[ds]}, {100 / r ** ND[ds]:.3g}% samples")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(f"{FIG}/F13_noise_robustness.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    p = f"{ROOT}/results/ood.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        img = d[d.src != "ecg"]
        fig, axs = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [3, 1.2]})
        g = img[img.r == 2].groupby(["src", "tgt", "method"])["q"].mean().unstack()
        g = g.sub(g["MAC (RFT)"], axis=0)
        labs = [f"{s}→{t}" for s, t in g.index]
        w = 0.27
        for j, m in enumerate(["Cubic spline", "VDSR", "PC-LSB"]):
            axs[0].bar(np.arange(len(g)) + (j - 1) * w, g[m].values, w * 0.92, color=method_color(m) if m != "Cubic spline" else "#b9b8b3", label=m)
        axs[0].set_xticks(range(len(g))); axs[0].set_xticklabels(labs)
        axs[0].axhline(0, color="k", lw=0.8); axs[0].set_ylabel("Δ PSNR vs MAC (dB)")
        axs[0].set_title("Cross-modality transfer (train → test), 25% samples; diagonal = in-domain"); axs[0].legend(fontsize=8)
        e = d[d.src == "ecg"].groupby(["r", "method"])["q"].mean().unstack()
        for j, m in enumerate(e.columns):
            axs[1].bar(np.arange(len(e)) + (j - len(e.columns) / 2 + 0.5) * 0.13, e[m].values, 0.12,
                       color=method_color(m) if m not in ("Cubic spline",) else "#b9b8b3", label=m,
                       hatch="//" if m == "PC-LSB+BS" else None)
        axs[1].set_xticks(range(len(e))); axs[1].set_xticklabels([f"{100/r:.0f}%" for r in e.index])
        axs[1].set_ylabel("SNR (dB)"); axs[1].set_title("ECG: LTDB → MIT-BIH Arrhythmia"); axs[1].legend(fontsize=6.5)
        save(fig, f"{FIG}/F14_out_of_distribution.png")
    fs = sorted(glob.glob(f"{ROOT}/results/dataeff_*.csv"))
    if fs:
        d = pd.read_csv(fs[0]); ds, r = d.ds.iloc[0], d.r.iloc[0]
        fig, ax = plt.subplots(figsize=(7, 4.3))
        for m in ["PC-LSB", "VDSR"]:
            g = d[d.method == m].groupby("n")["q"]; mu = g.mean(); se = g.std() / np.sqrt(g.count())
            ax.errorbar(mu.index, mu.values, 1.96 * se.values, color=method_color(m), marker="o", lw=2, capsize=2, label=m)
        for m, ls in [("MAC (RFT)", "--"), ("Cubic spline", ":")]:
            v = d[d.method == m]["q"].mean(); ax.axhline(v, color=method_color(m) if m.startswith("MAC") else "#8a8985", ls=ls, label=f"{m} (no training)")
        ax.set_xscale("log"); ax.set_xlabel("# training images"); ax.set_ylabel("PSNR (dB)")
        ns = sorted(n for n in d.n.unique() if n > 0); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"Data efficiency — {NAMES[ds]}, {100 / r ** 2:.3g}% samples"); ax.legend(fontsize=8)
        save(fig, f"{FIG}/F15_data_efficiency.png")


if __name__ == "__main__":
    for f in [fig_heatmap, fig_cd, fig_rates, fig_forest, fig_qual, fig_kspace, fig_unc, fig_runtime,
              fig_downstream, fig_extras]:
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAILED", f.__name__, e)
