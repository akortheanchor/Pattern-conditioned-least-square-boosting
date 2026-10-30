"""
E2 -- Main benchmark (Task A: deterministic uniform point sampling, the
acquisition model used by MAC in the slides).

python run_benchmark.py <dataset> <r1,r2,...> [--seeds 3] [--cnn-steps 2000]
writes results/bench_<ds>_r<r>.csv (per-sample, per-method metrics),
results/recon_<ds>_r<r>.npz (example reconstructions + uncertainty) and
results/models/<ds>_r<r>_*.pkl
"""
import argparse, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, ND, load, metrics, dump, undump, Timer
from ops import Decimation
from baselines import interp, spline3, tikhonov, cs_tv, cs_wavelet, omp_patch, LearnedLinear
from pclsb import PCLSB
from cnn import CNNBaseline
from downstream import make_hook


def patch_size(nd, r):
    # selected on the validation split (results/sweep_*.csv)
    return {1: {2: 17, 4: 17, 8: 25}, 2: {2: 7, 4: 7}}[nd][r]


def cfg(nd, r=4):
    K = {1: {2: 128, 4: 64, 8: 32}, 2: {2: 128, 4: 32}}[nd][r]
    return dict(K=K, T=2, B=6, lam=1e-3, n_train=300000, max_imgs=1500 if nd == 1 else 400,
                anchor="auto" if nd == 1 else "mac")


def log(msg, f):
    print(msg, flush=True); f.write(msg + "\n"); f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("rates")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cnn-steps", type=int, default=2000)
    ap.add_argument("--ntest", type=int, default=10000)
    a = ap.parse_args()
    ds, nd = a.ds, ND[a.ds]
    D = load(ds)
    Xtr, Xva, Xte = D["X_train"].astype(np.float64), D["X_val"].astype(np.float64), D["X_test"].astype(np.float64)
    if ds == "us":          # 6 evenly spaced frames per test video (units = videos for statistics)
        Xte = Xte[::2]
    Xte = Xte[: a.ntest]
    os.makedirs(f"{ROOT}/results/models", exist_ok=True)
    lf = open(f"{ROOT}/logs/bench_{ds}.log", "a")
    for r in [int(v) for v in a.rates.split(",")]:
        if os.path.exists(f"{ROOT}/results/bench_{ds}_r{r}.csv"):   # resumable after interruptions
            print(f"skip {ds} r={r} (done)"); continue
        op = Decimation(r, nd)
        p = patch_size(nd, r)
        log(f"=== {ds} r={r} (rate {op.rate:.4f}) train={len(Xtr)} val={len(Xva)} test={len(Xte)} p={p}", lf)
        fixed = {
            "Zero-fill": lambda y, s: op.pinv(y, s),
            "Nearest": lambda y, s: interp(y, op, "nearest"),
            "Linear": lambda y, s: interp(y, op, "linear"),
            "Cubic spline": lambda y, s: spline3(y, op),
            "Lanczos-3": lambda y, s: interp(y, op, "lanczos3"),
            "MAC (RFT)": lambda y, s: op.mac(y),
            "Tikhonov": lambda y, s: tikhonov(y, op, s),
            "CS-TV": lambda y, s: cs_tv(y, op, s),
            "CS-Wavelet": lambda y, s: cs_wavelet(y, op, s),
            "OMP-DCT": lambda y, s: omp_patch(y, op, s),
        }
        learned = {}
        # ---- learned linear (LMMSE)
        t0 = time.time()
        ll = LearnedLinear(op, p, n_train=cfg(nd, r)["n_train"]).fit(list(Xtr))
        learned["LMMSE (learned linear)"] = ll.predict
        log(f"  LMMSE fit {time.time()-t0:.1f}s", lf)
        # ---- CNNs on cubic-spline input
        Xin = [spline3(op.forward(x), op) for x in Xtr]
        for arch, name in [("srcnn", "SRCNN"), ("vdsr", "VDSR")]:
            t0 = time.time()
            mp = f"{ROOT}/results/models/{ds}_r{r}_{arch}.pkl"
            if os.path.exists(mp):
                from cnn import load_cnn
                m = load_cnn(arch, nd, undump(mp)); log(f"  {name} loaded from checkpoint", lf)
            else:
                m = CNNBaseline(arch, nd, steps=a.cnn_steps, seed=0).fit(Xin, Xtr, log=lambda s: log(s, lf),
                    Vin=[spline3(op.forward(x), op) for x in Xva], Vgt=Xva)
                dump([(np.asarray(w), np.asarray(b)) for w, b in m.params], mp)
            learned[name] = (lambda mm: (lambda y, s: op.project(mm.predict(spline3(y, op)), y)))(m)
            log(f"  {name} fit {time.time()-t0:.1f}s", lf)
        # ---- PC-LSB, several seeds
        pcs = []
        for sd in range(a.seeds):
            t0 = time.time()
            rng = np.random.default_rng(100 + sd)
            sub = rng.choice(len(Xtr), int(0.9 * len(Xtr)), replace=False)  # seed = different train subset
            mp = f"{ROOT}/results/models/{ds}_r{r}_pclsb_s{sd}.pkl"
            if os.path.exists(mp):
                m = undump(mp)
            else:
                m = PCLSB(op, p, seed=sd, **cfg(nd, r)).fit(list(Xtr[sub]), list(Xva))
                dump(m, mp)
            pcs.append(m)
            log(f"  PC-LSB seed {sd} fit {time.time()-t0:.1f}s gammas={[np.round(g,2).tolist() for g in m.gammas]} cal={m.cal:.3f} anchor={m.anchor}", lf)
        if ds in ("ecg",):
            from ecg_bs import bs_lsb, fit_eta
            fs = float(D.get("fs", 128))
            for sd, m in enumerate(pcs):
                eta, sc = fit_eta(list(Xva[:120]), op, (lambda mm: (lambda y, s: mm.predict(y, s)))(m))
                log(f"  PC-LSB+BS seed {sd}: eta={eta} val={ {k: round(v, 3) for k, v in sc.items()} }", lf)
                learned[f"PC-LSB+BS#s{sd}"] = (lambda mm, e: (lambda y, s: bs_lsb(y, op.r, mm.predict(y, s), fs=fs, eta=e)))(m, eta)
        hook = make_hook(ds, D)
        hook_gt = {}
        if hook is not None:   # downstream score of the fully-sampled ground truth (upper bound)
            g = pd.DataFrame([hook(i, x, x) for i, x in enumerate(Xte)]).mean()
            hook_gt = g.to_dict(); log(f"  downstream on ground truth: {hook_gt}", lf)
        rows, unc = [], []
        keep_idx = list(range(min(6, len(Xte))))
        ex = {"gt": [], "y": []}
        methods = dict(fixed); methods.update({k: v for k, v in learned.items() if "#s" not in k})
        methods.update({k: v for k, v in learned.items() if "#s" in k})
        for sd, m in enumerate(pcs):
            methods[f"PC-LSB#s{sd}"] = (lambda mm: (lambda y, s: mm.predict(y, s)))(m)
        for i, x in enumerate(Xte):
            y = op.forward(x)
            if i in keep_idx:
                ex["gt"].append(x); ex["y"].append(y)
            for name, f in methods.items():
                with Timer() as tm:
                    xh = f(y, x.shape)
                mt = metrics(x, xh, nd)
                if hook is not None and ("#s" not in name or name.endswith("#s0")):
                    mt.update(hook(i, x, xh))
                base, seed = (name.split("#s") + ["0"])[:2] if "#s" in name else (name, -1)
                rows.append(dict(ds=ds, r=r, rate=op.rate, idx=i, method=base, seed=int(seed), time=tm.dt, **mt))
                if i in keep_idx and (seed in (-1, "0")):
                    ex.setdefault(base, []).append(xh.astype(np.float32))
            # uncertainty (seed 0)
            xh, v = pcs[0].predict(y, x.shape, return_var=True)
            msk = op.phase(x.shape) != 0
            msk = np.broadcast_to(msk, x.shape)
            e = np.abs(x - xh)[msk]; sdv = np.sqrt(v[msk])
            rho = spearmanr(sdv, e).correlation
            cov = float(np.mean(e <= 1.645 * sdv))
            unc.append(dict(ds=ds, r=r, idx=i, spearman=rho, cov90=cov))
            if i in keep_idx:
                ex.setdefault("sigma", []).append(np.sqrt(v).astype(np.float32))
            if i % 50 == 0:
                df = pd.DataFrame(rows)
                s = df[df.idx <= i].groupby("method")["q"].mean().sort_values(ascending=False)
                log(f"  [{i}/{len(Xte)}] " + " | ".join(f"{k}:{v:.2f}" for k, v in s.items()), lf)
        df = pd.DataFrame(rows)
        for k, v in hook_gt.items():
            df["gt_" + k] = v
        df.to_csv(f"{ROOT}/results/bench_{ds}_r{r}.csv", index=False)
        pd.DataFrame(unc).to_csv(f"{ROOT}/results/unc_{ds}_r{r}.csv", index=False)
        np.savez_compressed(f"{ROOT}/results/recon_{ds}_r{r}.npz", **{k: np.array(v) for k, v in ex.items()})
        s = df.groupby("method")[["q", "time"]].mean().sort_values("q", ascending=False)
        log(s.to_string(), lf)


if __name__ == "__main__":
    main()
