"""
E3 -- Task B: accelerated MRI from undersampled Cartesian k-space.

Tests the slides' central sampling claim ("random sampling is forbidden;
deterministic low-frequency curtailment gives the maximum accuracy") by running
every method under three Hermitian-symmetric masks at the same acceleration:
  lowpass    -- deterministic high-frequency elimination (the slides' MAC setting)
  equispaced -- deterministic uniform lines + calibration centre
  vardens    -- variable-density random (standard CS-MRI)
python run_kspace.py <accels e.g. 4,8> [--cnn-steps 1500]
"""
import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load, metrics, dump, Timer
from ops import KSpace, kspace_mask
from baselines import tikhonov, cs_tv, cs_wavelet, LearnedLinear
from pclsb import PCLSB
from cnn import CNNBaseline
from downstream import make_hook


def hann_zf(y, op):
    H = op.m.shape[0]
    ky = np.abs(np.fft.fftfreq(H))[:, None]
    lines = np.where(op.m[:, 0])[0]
    kmax = np.abs(np.fft.fftfreq(H))[lines].max()
    w = np.clip(np.cos(np.pi * ky / (2 * kmax + 1e-9)) ** 2, 0, 1) * (ky <= kmax)
    return np.fft.ifft2(y * w, norm="ortho").real


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("accels")
    ap.add_argument("--cnn-steps", type=int, default=1500)
    ap.add_argument("--masks", default="lowpass,equispaced,vardens")
    a = ap.parse_args()
    D = load("mri")
    Xtr, Xva, Xte = [D[k].astype(np.float64) for k in ("X_train", "X_val", "X_test")]
    hook = make_hook("mri", D)
    lf = open(f"{ROOT}/logs/kspace.log", "a")

    def log(s):
        print(s, flush=True); lf.write(s + "\n"); lf.flush()

    for acc in [int(v) for v in a.accels.split(",")]:
        for kind in a.masks.split(","):
            if acc >= 8 and kind == "equispaced":   # compute budget: R=8 compares the two extremes only
                continue
            if os.path.exists(f"{ROOT}/results/kspace_{kind}_a{acc}.csv"):
                print("skip", kind, acc); continue
            m = kspace_mask(128, 128, acc, kind, seed=0)
            op = KSpace(m)
            log(f"=== accel {acc} mask {kind} rate {op.rate:.3f}")
            methods = {
                "Zero-fill / MAC": lambda y, s: op.pinv(y),
                "Hann-filtered ZF": lambda y, s: hann_zf(y, op),
                "Tikhonov": lambda y, s: tikhonov(y, op, s, lam=3e-3),
                "CS-TV": lambda y, s: cs_tv(y, op, s, iters=300, mu=1.0),
                "CS-Wavelet": lambda y, s: cs_wavelet(y, op, s),
            }
            t0 = time.time()
            ll = LearnedLinear(op, 9, n_train=250000).fit(list(Xtr))
            methods["LMMSE (learned linear)"] = ll.predict
            Xin = [op.pinv(op.forward(x)) for x in Xtr]
            for arch, name in [("srcnn", "SRCNN"), ("vdsr", "VDSR")]:
                cm = CNNBaseline(arch, 2, steps=a.cnn_steps, crop=48).fit(Xin, Xtr, log=log,
                    Vin=[op.pinv(op.forward(x)) for x in Xva], Vgt=Xva)
                methods[name] = (lambda mm: (lambda y, s: op.project(mm.predict(op.pinv(y)), y)))(cm)
            pc = PCLSB(op, 9, K=128, T=2, B=6, lam=1e-3, n_train=300000, mirror=False).fit(list(Xtr), list(Xva))
            dump(pc, f"{ROOT}/results/models/kspace_{kind}_a{acc}_pclsb.pkl")
            methods["PC-LSB"] = lambda y, s: pc.predict(y, s)
            log(f"  learned models fit in {time.time()-t0:.0f}s; gammas {[np.round(g,2).tolist() for g in pc.gammas]}")
            rows, ex = [], {"gt": [], "mask": m}
            for i, x in enumerate(Xte):
                y = op.forward(x)
                if i < 6:
                    ex["gt"].append(x)
                for name, f in methods.items():
                    with Timer() as tm:
                        xh = f(y, x.shape)
                    mt = metrics(x, xh, 2); mt.update(hook(i, x, xh))
                    rows.append(dict(accel=acc, mask=kind, rate=op.rate, idx=i, method=name, time=tm.dt, **mt))
                    if i < 6:
                        ex.setdefault(name, []).append(xh.astype(np.float32))
                if i % 60 == 0:
                    s = pd.DataFrame(rows).groupby("method")["q"].mean().sort_values(ascending=False)
                    log(f"  [{i}] " + " | ".join(f"{k}:{v:.2f}" for k, v in s.items()))
            df = pd.DataFrame(rows)
            df.to_csv(f"{ROOT}/results/kspace_{kind}_a{acc}.csv", index=False)
            np.savez_compressed(f"{ROOT}/results/recon_kspace_{kind}_a{acc}.npz", **{k: np.array(v) for k, v in ex.items()})
            log(df.groupby("method")[["q", "ssim", "cls_ok", "time"]].mean().sort_values("q", ascending=False).to_string())


if __name__ == "__main__":
    main()
