"""
Extra experiments
  ablation <ds> <r>   E4: component ablation of PC-LSB (test split, paired)
  noise    <ds> <r>   E5: robustness to measurement noise (sample SNR 40/30/20 dB)
  ood                 E6: cross-database / cross-modality generalisation
  dataeff  <ds> <r>   E8: data efficiency (PC-LSB vs VDSR vs #training images)
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, ND, load, metrics, undump, Timer
from ops import Decimation, mac_mirror
from baselines import spline3, cs_tv
from pclsb import PCLSB
from cnn import CNNBaseline, load_cnn
from run_benchmark import patch_size, cfg


def _split(ds):
    D = load(ds)
    return D, [D[k].astype(np.float64) for k in ("X_train", "X_val", "X_test")]


def ablation(ds, r, ntest=200):
    nd = ND[ds]; D, (Xtr, Xva, Xte) = _split(ds); Xte = Xte[:ntest]
    op = Decimation(r, nd); p = patch_size(nd, r); c = cfg(nd, r)
    variants = {
        "PC-LSB (full)": {},
        "w/o frequency gate": dict(gate=False),
        "w/o G-LSB consistency": dict(dc=False),
        "w/o boosting (T=1)": dict(T=1),
        "w/o pattern anchors (K=1)": dict(K=1),
        "w/o boundary-aware anchor": dict(mirror=False),
        "T=3 stages": dict(T=3),
    }
    rows = []
    fixed = {"MAC (RFT)": lambda y, s: op.mac(y), "MAC + mirror (no learning)": lambda y, s: mac_mirror(y, r, nd)}
    models = {}
    for name, kw in variants.items():
        cc = dict(c); cc.update(kw)
        t0 = time.time(); models[name] = PCLSB(op, p, seed=0, **cc).fit(list(Xtr), list(Xva))
        print(name, f"{time.time()-t0:.0f}s", flush=True)
    for i, x in enumerate(Xte):
        y = op.forward(x)
        for name, f in fixed.items():
            rows.append(dict(variant=name, idx=i, **metrics(x, f(y, x.shape), nd)))
        for name, m in models.items():
            rows.append(dict(variant=name, idx=i, **metrics(x, m.predict(y, x.shape), nd)))
    df = pd.DataFrame(rows); df["ds"] = ds; df["r"] = r
    df.to_csv(f"{ROOT}/results/ablation_{ds}_r{r}.csv", index=False)
    print(df.groupby("variant")["q"].mean().sort_values())


def noise(ds, r, ntest=150):
    nd = ND[ds]; D, (Xtr, Xva, Xte) = _split(ds); Xte = Xte[:ntest]
    op = Decimation(r, nd); p = patch_size(nd, r); c = cfg(nd, r)
    pc_clean = undump(f"{ROOT}/results/models/{ds}_r{r}_pclsb_s0.pkl")
    vd = load_cnn("vdsr", nd, undump(f"{ROOT}/results/models/{ds}_r{r}_vdsr.pkl"))
    rows = []
    for snr in [np.inf, 40, 30, 20]:
        rng = np.random.default_rng(7)

        def noisy(x):
            y = op.forward(x)
            if np.isfinite(snr):
                y = y + rng.standard_normal(y.shape) * np.sqrt(np.mean((y - y.mean()) ** 2) / 10 ** (snr / 10))
            return y
        Ytr = [noisy(x) for x in Xtr[:c.get("max_imgs", 400) if nd == 2 else 1500]]
        Yva = [noisy(x) for x in Xva]
        pc_na = PCLSB(op, p, seed=0, noise_aware=True, **c).fit(list(Xtr[:len(Ytr)]), list(Xva), Ytr, Yva)
        print(f"SNR {snr}: noise-aware mu={pc_na.mu}", flush=True)
        for i, x in enumerate(Xte):
            y = noisy(x)
            outs = {"MAC (RFT)": op.mac(y), "Cubic spline": spline3(y, op), "CS-TV": cs_tv(y, op, x.shape),
                    "VDSR": op.project(vd.predict(spline3(y, op)), y),
                    "PC-LSB (clean-trained)": pc_clean.predict(y, x.shape),
                    "PC-LSB (noise-aware)": pc_na.predict(y, x.shape)}
            for k, v in outs.items():
                rows.append(dict(meas_snr=snr, method=k, idx=i, **metrics(x, v, nd)))
        print(pd.DataFrame(rows)[lambda d: d.meas_snr == snr].groupby("method")["q"].mean().sort_values(), flush=True)
    df = pd.DataFrame(rows); df["ds"] = ds; df["r"] = r
    df.to_csv(f"{ROOT}/results/noise_{ds}_r{r}.csv", index=False)


def ood():
    rows = []
    # (a) ECG: trained on MIT-BIH LTDB, tested on MIT-BIH Arrhythmia
    from ecg_bs import bs_lsb
    X = load("mitdb")["X_test"].astype(np.float64)
    for r in [2, 4]:
        op = Decimation(r, 1)
        pc = undump(f"{ROOT}/results/models/ecg_r{r}_pclsb_s0.pkl")
        vd = load_cnn("vdsr", 1, undump(f"{ROOT}/results/models/ecg_r{r}_vdsr.pkl"))
        for i, x in enumerate(X):
            y = op.forward(x)
            px = pc.predict(y, x.shape)
            outs = {"MAC (RFT)": op.mac(y), "Cubic spline": spline3(y, op), "CS-TV": cs_tv(y, op, x.shape),
                    "VDSR": op.project(vd.predict(spline3(y, op)), y), "PC-LSB": px,
                    "PC-LSB+BS": bs_lsb(y, r, px, eta=0.75)}
            for k, v in outs.items():
                rows.append(dict(src="ecg", tgt="mitdb", r=r, method=k, idx=i, **metrics(x, v, 1)))
    # (b) imaging: trained on brain MRI, tested on CXR and ultrasound (and vice versa)
    for src in ["mri", "cxr", "us"]:
        for r in [2, 4]:
            op = Decimation(r, 2)
            pc = undump(f"{ROOT}/results/models/{src}_r{r}_pclsb_s0.pkl")
            vd = load_cnn("vdsr", 2, undump(f"{ROOT}/results/models/{src}_r{r}_vdsr.pkl"))
            for tgt in ["mri", "cxr", "us"]:
                Xt = load(tgt)["X_test"].astype(np.float64)[::3][:80]
                for i, x in enumerate(Xt):
                    y = op.forward(x)
                    outs = {"MAC (RFT)": op.mac(y), "Cubic spline": spline3(y, op),
                            "VDSR": op.project(vd.predict(spline3(y, op)), y), "PC-LSB": pc.predict(y, x.shape)}
                    for k, v in outs.items():
                        rows.append(dict(src=src, tgt=tgt, r=r, method=k, idx=i, **metrics(x, v, 2)))
            print(src, r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{ROOT}/results/ood.csv", index=False)
    print(df.groupby(["src", "tgt", "r", "method"])["q"].mean().unstack().round(2))


def dataeff(ds, r, sizes=(25, 50, 100, 200, 400), ntest=120, steps=1500):
    nd = ND[ds]; D, (Xtr, Xva, Xte) = _split(ds); Xte = Xte[:ntest]
    op = Decimation(r, nd); p = patch_size(nd, r); c = cfg(nd, r)
    rows = []
    rng = np.random.default_rng(0); perm = rng.permutation(len(Xtr))
    for n in sizes:
        sub = Xtr[perm[:n]]
        pc = PCLSB(op, p, seed=0, **dict(c, n_train=min(c["n_train"], n * 2000))).fit(list(sub), list(Xva))
        vd = CNNBaseline("vdsr", nd, steps=steps).fit([spline3(op.forward(x), op) for x in sub], sub)
        for i, x in enumerate(Xte):
            y = op.forward(x)
            rows.append(dict(n=n, method="PC-LSB", idx=i, **metrics(x, pc.predict(y, x.shape), nd)))
            rows.append(dict(n=n, method="VDSR", idx=i, **metrics(x, op.project(vd.predict(spline3(y, op)), y), nd)))
            if n == sizes[0]:
                rows.append(dict(n=0, method="MAC (RFT)", idx=i, **metrics(x, op.mac(y), nd)))
                rows.append(dict(n=0, method="Cubic spline", idx=i, **metrics(x, spline3(y, op), nd)))
        print(n, pd.DataFrame(rows)[lambda d: d.n == n].groupby("method")["q"].mean().to_dict(), flush=True)
    df = pd.DataFrame(rows); df["ds"] = ds; df["r"] = r
    df.to_csv(f"{ROOT}/results/dataeff_{ds}_r{r}.csv", index=False)




def plugin(ds, r, ntest=240):
    """E9: PC-LSB as a plug-in booster on top of a trained CNN (anchor = VDSR output)."""
    nd = ND[ds]; D, (Xtr, Xva, Xte) = _split(ds); Xte = Xte[:ntest]
    op = Decimation(r, nd); p = patch_size(nd, r); c = cfg(nd, r); c["anchor"] = "mac"
    vd = load_cnn("vdsr", nd, undump(f"{ROOT}/results/models/{ds}_r{r}_vdsr.pkl"))
    vfun = lambda y, s: op.project(vd.predict(spline3(y, op)), y)
    c2 = dict(c); c2["anchor"] = vfun
    pc_v = PCLSB(op, p, seed=0, **c2).fit(list(Xtr), list(Xva))
    pc = undump(f"{ROOT}/results/models/{ds}_r{r}_pclsb_s0.pkl")
    rows = []
    for i, x in enumerate(Xte):
        y = op.forward(x)
        for k, v in {"MAC (RFT)": op.mac(y), "VDSR": vfun(y, x.shape), "PC-LSB": pc.predict(y, x.shape),
                     "VDSR + PC-LSB (plug-in)": pc_v.predict(y, x.shape)}.items():
            rows.append(dict(method=k, idx=i, **metrics(x, v, nd)))
    df = pd.DataFrame(rows); df["ds"] = ds; df["r"] = r
    df.to_csv(f"{ROOT}/results/plugin_{ds}_r{r}.csv", index=False)
    print(df.groupby("method")["q"].mean().sort_values())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ood":
        ood()
    else:
        globals()[cmd](sys.argv[2], int(sys.argv[3]))
