"""Animated results (GIF)."""
import io, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import setup, PAL
from common import ROOT, load, undump, metrics
from ops import Decimation, KSpace, kspace_mask
from baselines import spline3, cs_tv
from cnn import load_cnn

setup()
OUT = f"{ROOT}/gifs"
os.makedirs(OUT, exist_ok=True)


def fig2img(fig, dpi=90):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=dpi); plt.close(fig); b.seek(0)
    return np.array(Image.open(b).convert("RGB"))


def save_gif(frames, path, fps=6):
    h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
    imageio.mimsave(path, [f[:h, :w] for f in frames], duration=1 / fps, loop=0)
    print("wrote", path, len(frames))


# G1: ultrasound video, 25% samples
def g_us(r=2):
    D = load("us"); seqs = D["seq_test"]
    op = Decimation(r, 2)
    pc = undump(f"{ROOT}/results/models/us_r{r}_pclsb_s0.pkl")
    vd = load_cnn("vdsr", 2, undump(f"{ROOT}/results/models/us_r{r}_vdsr.pkl"))
    frames = []
    for si in range(min(2, len(seqs))):
        for x in seqs[si][:30].astype(np.float64):
            y = op.forward(x)
            outs = [("ground truth", x), (f"input {100/r**2:.0f}% (nearest)", np.kron(y, np.ones((r, r)))),
                    ("MAC (RFT)", op.mac(y)), ("VDSR", op.project(vd.predict(spline3(y, op)), y)),
                    ("PC-LSB (ours)", pc.predict(y, x.shape))]
            fig, axs = plt.subplots(1, 6, figsize=(13.2, 2.75))
            for ax, (t, im) in zip(axs, outs):
                ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
                lab = t if t in ("ground truth",) or t.startswith("input") else f"{t}\n{metrics(x, im, 2)['psnr']:.2f} dB"
                ax.set_title(lab, fontsize=8.5, color=PAL["ours"] if "ours" in t else "#0b0b0b"); ax.axis("off")
            axs[-1].imshow(np.abs(x - outs[-1][1]), cmap="magma", vmin=0, vmax=0.2); axs[-1].axis("off")
            axs[-1].set_title("|error| PC-LSB", fontsize=8.5)
            fig.suptitle("POCUS lung ultrasound video reconstructed frame-by-frame from 25% of pixels", fontsize=10)
            frames.append(fig2img(fig))
    save_gif(frames, f"{OUT}/G1_ultrasound_video_reconstruction.gif", fps=8)


# G2: sampling-pattern sweep for MRI (the slides' deterministic claim vs CS)
def g_kspace():
    x = load("mri")["X_test"][3].astype(np.float64)
    frames = []
    for acc in [2, 3, 4, 5, 6, 8, 10, 12]:
        row = []
        for kind in ["lowpass", "vardens"]:
            op = KSpace(kspace_mask(128, 128, acc, kind, seed=0))
            y = op.forward(x)
            row.append((kind, op, op.pinv(y), cs_tv(y, op, x.shape, iters=300)))
        fig, axs = plt.subplots(2, 3, figsize=(8.4, 5.9))
        for i, (kind, op, zf, tv) in enumerate(row):
            axs[i, 0].imshow(np.fft.fftshift(op.m), cmap="gray"); axs[i, 0].set_title(
                ("deterministic low-pass (MAC)" if kind == "lowpass" else "variable-density random") + f"\nmask {op.rate*100:.0f}%", fontsize=8.5)
            axs[i, 1].imshow(np.clip(zf, 0, 1), cmap="gray"); axs[i, 1].set_title(f"zero-fill / MAC  {metrics(x, zf, 2)['psnr']:.2f} dB", fontsize=8.5)
            axs[i, 2].imshow(np.clip(tv, 0, 1), cmap="gray"); axs[i, 2].set_title(f"CS-TV  {metrics(x, tv, 2)['psnr']:.2f} dB", fontsize=8.5)
        for a in axs.ravel():
            a.axis("off")
        fig.suptitle(f"Accelerated MRI, R = {acc}", fontsize=11)
        frames.append(fig2img(fig))
    save_gif(frames + frames[::-1], f"{OUT}/G2_mri_sampling_pattern_sweep.gif", fps=2)


# G3: boosting stages (MAC -> stage 1 -> stage 2), MRI 6.25%
def g_boost(ds="mri", r=4, idx=5):
    x = load(ds)["X_test"][idx].astype(np.float64)
    op = Decimation(r, 2); pc = undump(f"{ROOT}/results/models/{ds}_r{r}_pclsb_s0.pkl")
    from pclsb import anchor_input
    y = op.forward(x)
    states = [("MAC anchor (RFT zero-padding)", anchor_input(y, op, x.shape, pc.mirror, getattr(pc, "anchor", "mac")))]
    cur = states[0][1]
    for t, (st, gam) in enumerate(zip(pc.stages, pc.gammas)):
        cur, _ = pc._step(st, gam, cur, y, x.shape)
        states.append((f"after boosting stage {t+1}", cur))
    frames = []
    for k in range(len(states)):
        for a in np.linspace(0, 1, 6):
            im = states[k][1] if k == 0 else (1 - a) * states[k - 1][1] + a * states[k][1]
            fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.6))
            axs[0].imshow(x, cmap="gray", vmin=0, vmax=1); axs[0].set_title("ground truth", fontsize=9)
            axs[1].imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
            axs[1].set_title(f"{states[k][0]}\n{metrics(x, im, 2)['psnr']:.2f} dB", fontsize=9)
            axs[2].imshow(np.abs(x - im), cmap="magma", vmin=0, vmax=0.25); axs[2].set_title("|error|", fontsize=9)
            for q in axs:
                q.axis("off")
            fig.suptitle(f"PC-LSB boosting from {100/r**2:.2f}% samples", fontsize=10)
            frames.append(fig2img(fig))
        frames += [frames[-1]] * 4
    save_gif(frames, f"{OUT}/G3_boosting_stages.gif", fps=6)


# G4: beat-synchronous pooling on ECG
def g_bs(r=4, idx=373):
    from ecg_bs import bs_lsb, _nw
    D = load("ecg"); x = D["X_test"][idx].astype(np.float64)
    op = Decimation(r, 1); pc = undump(f"{ROOT}/results/models/ecg_r{r}_pclsb_s0.pkl")
    y = op.forward(x); x1 = pc.predict(y, x.shape)
    xh, info = bs_lsb(y, r, x1, eta=0.75, return_info=True)   # benchmark setting (eta tuned on validation)
    t = info["t"]; pos = np.arange(len(y)) * r; P, Q = 32, 51
    grid = np.arange(-P, Q + 1.)
    frames = []
    for nb in range(1, len(t) + 1):
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
        axs[0].plot(x, color=PAL["gt"], lw=1.6, label="ground truth")
        axs[0].plot(pos, y, "o", ms=2.5, color="k", label=f"{100/r:.0f}% samples")
        for j in range(nb):
            axs[0].axvspan(t[j] - P, t[j] + Q, color=PAL["ours"], alpha=0.10)
        axs[0].set_xlim(0, len(x)); axs[0].set_title("beats detected on the PC-LSB estimate (shaded)", fontsize=9.5)
        ta, va = [], []
        cmap = plt.get_cmap("Blues")
        for j in range(nb):
            m = (pos >= t[j] - P) & (pos <= t[j] + Q)
            axs[1].plot(pos[m] - t[j], y[m], "o", ms=3.5, color=cmap(0.35 + 0.6 * j / max(len(t) - 1, 1)))
            ta.append(pos[m] - t[j]); va.append(y[m])
        if nb >= 2:
            T, _ = _nw(np.concatenate(ta), np.concatenate(va), grid, 0.6)
            axs[1].plot(grid, T, color=PAL["ours"], lw=2, label="pooled template")
        axs[1].set_xlim(-P, Q); axs[1].set_title(f"{nb} beats pooled on a sub-sample time axis\n(effective rate ≈ {nb * 128 / r:.0f} Hz)", fontsize=9.5)
        axs[1].set_xlabel("time from R peak (fine samples)")
        frames.append(fig2img(fig))
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(x, color=PAL["gt"], lw=2.2, label="ground truth")
    ax.plot(x1, color="#86b6ef", lw=1.3, label=f"PC-LSB {metrics(x, x1, 1)['snr']:.2f} dB")
    ax.plot(xh, color=PAL["ours"], lw=1.3, label=f"PC-LSB+BS {metrics(x, xh, 1)['snr']:.2f} dB")
    ax.plot(pos, y, "o", ms=2.5, color="k"); ax.legend(fontsize=8, ncol=3); ax.set_xlim(0, len(x))
    ax.set_title("final beat-synchronous reconstruction", fontsize=10)
    last = fig2img(fig)
    frames += [last] * 8
    save_gif(frames, f"{OUT}/G4_ecg_beat_synchronous_pooling.gif", fps=2)


if __name__ == "__main__":
    for g in sys.argv[1:] or ["g_us", "g_kspace", "g_boost", "g_bs"]:
        try:
            globals()[g]()
        except Exception as e:
            import traceback; traceback.print_exc()
