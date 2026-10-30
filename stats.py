"""
Statistical validation.
 * per configuration: Friedman test over all methods (per-unit scores),
   PC-LSB vs every baseline: Wilcoxon signed-rank (two-sided), Holm-corrected,
   rank-biserial effect size, bootstrap 95% CI of the mean paired difference.
 * across configurations: average ranks, Friedman + Nemenyi critical difference.
Units: one test window/image; for ultrasound the unit is the video and for
CXR the patient (frames/images of one subject are averaged first).
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT

QA = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
      11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489}


def unit_table(df, ds, metric="q"):
    """samples x methods matrix of scores (PC-LSB averaged over seeds)."""
    d = df.groupby(["idx", "method"])[metric].mean().unstack()
    if ds in ("us", "cxr"):
        grp = pd.read_csv(f"{ROOT}/data/groups_{ds}.csv")["group"].values
        d["__g"] = grp[d.index.values]
        d = d.groupby("__g").mean()
    return d


def boot_ci(x, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (B, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(tab, ours="PC-LSB"):
    out = []
    for b in tab.columns:
        if b == ours:
            continue
        d = (tab[ours] - tab[b]).dropna().values
        if np.allclose(d, 0):
            continue
        st = wilcoxon(d)
        rk = rankdata(np.abs(d)); wp = rk[d > 0].sum(); wm = rk[d < 0].sum()
        lo, hi = boot_ci(d)
        out.append(dict(baseline=b, mean_diff=d.mean(), median_diff=np.median(d), ci_lo=lo, ci_hi=hi,
                        win_rate=float((d > 0).mean()), p=st.pvalue, rbc=(wp - wm) / (wp + wm), n=len(d)))
    o = pd.DataFrame(out)
    o["p_holm"] = multipletests(o.p, method="holm")[1]
    return o.sort_values("mean_diff")


def main():
    res = {}
    allcfg = []
    files = sorted(glob.glob(f"{ROOT}/results/bench_*_r*.csv"))
    for f in files:
        df = pd.read_csv(f)
        ds, r = df.ds.iloc[0], int(df.r.iloc[0])
        tab = unit_table(df, ds)
        fr = friedmanchisquare(*[tab[c].values for c in tab.columns])
        cmp_ = compare(tab)
        cmp_.to_csv(f"{ROOT}/results/stats_{ds}_r{r}.csv", index=False)
        seed_means = df[df.method == "PC-LSB"].groupby("seed")["q"].mean()
        res[f"{ds}_r{r}"] = dict(friedman_chi2=fr.statistic, friedman_p=fr.pvalue, n_units=len(tab),
                                 pclsb_seed_mean=seed_means.mean(), pclsb_seed_std=seed_means.std(ddof=1),
                                 n_sig_better=int(((cmp_.p_holm < 0.05) & (cmp_.mean_diff > 0)).sum()),
                                 n_sig_worse=int(((cmp_.p_holm < 0.05) & (cmp_.mean_diff < 0)).sum()),
                                 n_baselines=len(cmp_))
        m = tab.mean(); m.name = f"{ds}_r{r}"; allcfg.append(m)
        if "PC-LSB+BS" in tab.columns:
            c2 = compare(tab.drop(columns=["PC-LSB"]).rename(columns={"PC-LSB+BS": "PC-LSB"}))
            c2.to_csv(f"{ROOT}/results/stats_{ds}_r{r}_BS.csv", index=False)
    M = pd.DataFrame(allcfg)
    common = [c for c in M.columns if M[c].notna().all() and c != "PC-LSB+BS"]
    M2 = M[common]
    ranks = M2.rank(axis=1, ascending=False)
    avg = ranks.mean().sort_values()
    k, N = len(common), len(M2)
    fr = friedmanchisquare(*[M2[c].values for c in common])
    cd = QA.get(k, 3.5) * np.sqrt(k * (k + 1) / (6 * N))
    res["global"] = dict(k=k, N=N, friedman_chi2=fr.statistic, friedman_p=fr.pvalue, CD=cd,
                         avg_rank=avg.to_dict())
    M.to_csv(f"{ROOT}/results/summary_means.csv")
    with open(f"{ROOT}/results/stats_summary.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
