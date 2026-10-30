"""Hyper-parameter selection for PC-LSB on the VALIDATION split only.
python sweep.py <ds> <r> "<list of dicts>"  -> results/sweep_<ds>_r<r>.csv"""
import sys, os, time, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, ND, load, metrics
from ops import Decimation
from pclsb import PCLSB

ds, r = sys.argv[1], int(sys.argv[2]); grid = eval(sys.argv[3])
nd = ND[ds]; D = load(ds)
Xtr = D["X_train"].astype(float); Xva = D["X_val"].astype(float)
# inner split of validation: half to fit the gate, half to score
Xg, Xs = Xva[: len(Xva) // 2], Xva[len(Xva) // 2:]
op = Decimation(r, nd)
rows = []
for g in grid:
    t = time.time()
    kw = dict(K=64, T=2, B=6, lam=1e-3, n_train=150000); kw.update(g)
    p = kw.pop("p")
    m = PCLSB(op, p, **kw).fit(list(Xtr), list(Xg))
    q = np.mean([metrics(x, m.predict(op.forward(x), x.shape), nd)["q"] for x in Xs])
    rows.append(dict(p=p, **kw, q=q, sec=time.time() - t)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/results/sweep_{ds}_r{r}.csv", index=False)
