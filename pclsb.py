"""
PC-LSB: Pattern-Conditioned Least-Square Boost.

The slides' MAC predicts unknown data by zero-padding the transform of the
samples (prior = 0 for every unacquired coefficient) and claims this is the
"maximum accuracy". We generalise Least Square Boost to an arbitrary prior p:

        x_hat = argmin_x ||x - p||  s.t.  A x = y   =   p + A^+ (y - A p)       (G-LSB)

whose error is (I - A^+A)(x - p): only the null-space component of the prior
matters, and MAC is the special case p = MAC(y). PC-LSB builds p by
*learning the local data pattern* the slides talk about:

  1. anchor  x0 = MAC(y) (boundary-aware RFT zero-padding),
  2. every output sample gets a contrast-normalised, sign-folded patch
     pattern f~ = sigma * (f - mean) / ||f - mean|| from x0,
  3. f~ is matched to one of K pattern anchors (k-means codebook) and a
     per-(anchor, sampling-phase) ridge regressor predicts the null-space
     residual, rescaled by the patch contrast  (scale- and sign-equivariant),
  4. a frequency-band gate gamma (fit by least squares on validation data)
     shrinks the correction band-by-band  -> validation error <= MAC's,
  5. G-LSB projection restores exact consistency with the samples
     (non-expansive towards the truth), then the stage is repeated
     (boosting) on the new estimate.
Every step is closed-form (k-means + ridge); inference is non-iterative.
Each prediction also carries a per-sample uncertainty from the anchor's
residual variance.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ops import Decimation, KSpace, mac_mirror


def extract_patches(x, p, nd):
    h = p // 2
    if nd == 1:
        xp = np.pad(x, h, mode="reflect")
        return sliding_window_view(xp, p).reshape(-1, p)
    xp = np.pad(x, h, mode="reflect")
    return sliding_window_view(xp, (p, p)).reshape(-1, p * p)


def anchor_input(y, op, shape, mirror=True, anchor="mac"):
    if callable(anchor):               # any external estimator (e.g. a CNN) as the anchor
        return anchor(y, shape)
    if isinstance(op, Decimation):
        if anchor == "spline":
            from baselines import spline3
            return spline3(y, op)
        return mac_mirror(y, op.r, op.ndim) if mirror else op.mac(y)
    return op.pinv(y)


def _norm_feats(F, floor=1e-6):
    F = F - F.mean(1, keepdims=True)
    s = np.sqrt((F ** 2).sum(1) + floor ** 2)   # contrast floor avoids blow-up on flat patches
    return F / s[:, None], s


def sample_train(X_in, op, p, X_gt=None, n=200000, rng=None, normalize=True, Xs_in=None, floor=None, all_phases=False):
    """X_in: list/array of ground-truth signals (if Xs_in is None the anchor input is
    computed from simulated measurements). Returns features, targets, phases."""
    rng = rng or np.random.default_rng(0)
    N = len(X_in)
    per = max(1, n // N)
    Fs, Ts, Ps, Ss = [], [], [], []
    for i in range(N):
        x = X_in[i]
        x0 = Xs_in[i] if Xs_in is not None else anchor_input(op.forward(x), op, x.shape)
        F = extract_patches(x0, p, op.ndim)
        T = (x - x0).ravel()
        ph = np.broadcast_to(op.phase(x.shape), x.shape).ravel()
        keep = np.where(ph != 0)[0] if (isinstance(op, Decimation) and not all_phases) else np.arange(len(T))
        idx = rng.choice(keep, min(per, len(keep)), replace=False)
        Fs.append(F[idx]); Ts.append(T[idx]); Ps.append(ph[idx])
    F = np.concatenate(Fs).astype(np.float64); T = np.concatenate(Ts); PH = np.concatenate(Ps)
    if not normalize:
        return F, T, PH
    if floor is None:
        Fc = F - F.mean(1, keepdims=True)
        floor = 0.1 * np.median(np.linalg.norm(Fc, axis=1)) + 1e-6
    Fn, s = _norm_feats(F, floor)
    return Fn, T / s, PH, s, floor


def band_split(d, nd, B):
    """Split a correction map into B radial frequency bands (sums back to d)."""
    if nd == 1:
        D = np.fft.rfft(d)
        f = np.linspace(0, 1, D.shape[-1])
        edges = np.linspace(0, 1 + 1e-9, B + 1)
        return [np.fft.irfft(D * ((f >= edges[b]) & (f < edges[b + 1])), n=d.shape[-1]) for b in range(B)]
    D = np.fft.fft2(d)
    fy = np.fft.fftfreq(d.shape[0])[:, None]; fx = np.fft.fftfreq(d.shape[1])[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2) / np.sqrt(0.5)
    edges = np.linspace(0, 1 + 1e-9, B + 1)
    return [np.fft.ifft2(D * ((rad >= edges[b]) & (rad < edges[b + 1]))).real for b in range(B)]


class Stage:
    def __init__(self, op, p, K, lam, seed, all_phases=False, linear=True, lam_lin=1e-4):
        self.op, self.p, self.K, self.lam, self.seed, self.all_phases = op, p, K, lam, seed, all_phases
        self.linear, self.lam_lin = linear, lam_lin

    def _lin(self, Fraw, ph):
        out = np.zeros(len(Fraw))
        if not self.linear:
            return out
        for p_, W in self.Wlin.items():
            m = ph == p_
            out[m] = np.c_[Fraw[m], np.ones(m.sum())] @ W
        return out

    def fit(self, Xgt, Xin, n_train):
        rng = np.random.default_rng(self.seed)
        Fraw, Traw, PH = sample_train(Xgt, self.op, self.p, n=n_train, rng=rng, Xs_in=Xin,
                                      normalize=False, all_phases=self.all_phases)
        # (i) linear Wiener backbone (per sampling phase, exact LMMSE on raw patches)
        self.Wlin = {}
        if self.linear:
            for p_ in np.unique(PH):
                m = PH == p_
                A = np.c_[Fraw[m], np.ones(m.sum())]
                G = A.T @ A
                self.Wlin[p_] = np.linalg.solve(G + self.lam_lin * np.trace(G) / len(G) * np.eye(len(G)), A.T @ Traw[m])
        T2 = Traw - self._lin(Fraw, PH)
        # (ii) pattern-conditioned non-linear correction on contrast-normalised patterns
        Fc = Fraw - Fraw.mean(1, keepdims=True)
        self.floor = 0.1 * np.median(np.linalg.norm(Fc, axis=1)) + 1e-6
        F, s = _norm_feats(Fraw, self.floor)
        T = T2 / s
        sub = rng.choice(len(F), min(60000, len(F)), replace=False)
        Fk = F[sub]
        # sign-invariant spherical k-means: patterns f and -f share an anchor
        C = Fk[rng.choice(len(Fk), self.K, replace=False)].copy()
        for _ in range(15):
            S = Fk @ C.T
            a_ = np.abs(S).argmax(1); sg_ = np.sign(S[np.arange(len(Fk)), a_])
            for k in range(self.K):
                m = a_ == k
                if m.sum() > 0:
                    C[k] = (Fk[m] * sg_[m, None]).sum(0)
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        self.C = C
        a, sg = self._assign(F)
        self.W, self.var = {}, {}
        dim = F.shape[1] + 1
        self.Wph, self.varph = {}, {}
        for ph in np.unique(PH):
            m = PH == ph
            A = np.c_[F[m], sg[m]]
            t = T[m]
            Wg = np.linalg.solve(A.T @ A + self.lam * len(A) * np.eye(dim), A.T @ t)
            self.Wph[ph] = Wg; self.varph[ph] = np.mean((t - A @ Wg) ** 2)
            for k in range(len(self.C)):
                mk = m & (a == k)
                nk = mk.sum()
                if nk < 3 * dim:
                    continue
                Ak = np.c_[F[mk], sg[mk]]
                tk = T[mk]
                # shrink towards the phase-global model (hierarchical ridge)
                lam_k = self.lam * nk
                W = Wg + np.linalg.solve(Ak.T @ Ak + lam_k * np.eye(dim), Ak.T @ (tk - Ak @ Wg))
                self.W[(ph, k)] = W
                self.var[(ph, k)] = np.mean((tk - Ak @ W) ** 2)
        return self

    def _assign(self, F):
        S = F @ self.C.T
        a = np.abs(S).argmax(1)
        sg = np.sign(S[np.arange(len(F)), a]); sg[sg == 0] = 1
        return a, sg

    def predict(self, x0, shape):
        F = extract_patches(x0, self.p, self.op.ndim).astype(np.float64)
        Fn, s = _norm_feats(F, self.floor)
        a, sg = self._assign(Fn)
        ph = np.broadcast_to(self.op.phase(shape), shape).ravel()
        lin = self._lin(F, ph) if hasattr(self, "Wlin") else 0.0
        A = np.c_[Fn, sg]
        res = np.zeros(len(F)); var = np.zeros(len(F))
        for p_ in np.unique(ph):
            m = ph == p_
            if p_ not in self.Wph:
                continue
            res[m] = A[m] @ self.Wph[p_]; var[m] = self.varph[p_]
            for k in np.unique(a[m]):
                if (p_, k) in self.W:
                    mk = m & (a == k)
                    res[mk] = A[mk] @ self.W[(p_, k)]; var[mk] = self.var[(p_, k)]
        if hasattr(self, "Wlin") and self.linear:
            known = np.isin(ph, list(self.Wlin.keys()))
            lin = np.where(known, lin, 0.0)
        return (lin + res * s).reshape(shape), (var * s ** 2).reshape(shape)


class PCLSB:
    def __init__(self, op, p, K=64, T=2, B=6, lam=1e-3, gate=True, dc=True, mirror=True,
                 n_train=250000, seed=0, mu=1.0, max_imgs=400, noise_aware=False, anchor="mac", linear=True):
        self.op, self.p, self.K, self.T, self.B = op, p, K, T, B
        self.lam, self.gate, self.dc, self.mirror = lam, gate, dc, mirror
        self.n_train, self.seed, self.mu, self.max_imgs = n_train, seed, mu, max_imgs
        self.noise_aware = noise_aware
        self.anchor = anchor
        self.linear = linear

    def _proj(self, p_, y):
        return self.op.project(p_, y, self.mu) if self.dc else p_

    def fit(self, Xtr, Xval, Ytr=None, Yval=None):
        """Ytr/Yval optionally supply (noisy) measurements; default noiseless.
        anchor='auto' fits the MAC- and spline-anchored models and keeps the one with
        lower validation error (model selection on validation data only)."""
        if self.anchor == "auto":
            best = None
            for an in ("mac", "spline"):
                self.anchor = an
                self._fit(Xtr, Xval, Ytr, Yval)
                Yv = Yval if Yval is not None else [self.op.forward(x) for x in Xval]
                e = np.mean([np.mean((x - self.predict(y, x.shape)) ** 2) for x, y in zip(Xval, Yv)])
                if best is None or e < best[0]:
                    best = (e, an, self.stages, self.gammas, self.cal, self.mu)
            _, self.anchor, self.stages, self.gammas, self.cal, self.mu = best
            return self
        return self._fit(Xtr, Xval, Ytr, Yval)

    def _fit(self, Xtr, Xval, Ytr=None, Yval=None):
        op = self.op
        if len(Xtr) > self.max_imgs:   # subsample training signals (patch budget is fixed anyway)
            sel = np.random.default_rng(self.seed).choice(len(Xtr), self.max_imgs, replace=False)
            Xtr = [Xtr[i] for i in sel]
            Ytr = [Ytr[i] for i in sel] if Ytr is not None else None
        Ytr = Ytr if Ytr is not None else [op.forward(x) for x in Xtr]
        Yval = Yval if Yval is not None else [op.forward(x) for x in Xval]
        if self.noise_aware:
            self.mu = 0.0
        cur_tr = [anchor_input(y, op, x.shape, self.mirror, self.anchor) for y, x in zip(Ytr, Xtr)]
        cur_va = [anchor_input(y, op, x.shape, self.mirror, self.anchor) for y, x in zip(Yval, Xval)]
        self.stages, self.gammas = [], []
        for t in range(self.T):
            st = Stage(op, self.p, self.K, self.lam, self.seed + t, self.noise_aware, self.linear).fit(Xtr, cur_tr, self.n_train)
            # gate fit on validation
            if self.gate:
                G = np.zeros((self.B, self.B)); h = np.zeros(self.B)
                for x, x0 in zip(Xval, cur_va):
                    d, _ = st.predict(x0, x.shape)
                    bands = np.array([b.ravel() for b in band_split(d, op.ndim, self.B)])
                    G += bands @ bands.T; h += bands @ (x - x0).ravel()
                gam = np.linalg.lstsq(G + 1e-9 * np.eye(self.B), h, rcond=None)[0]
                gam = np.clip(gam, 0, 1.5)
            else:
                gam = np.ones(self.B)
            self.stages.append(st); self.gammas.append(gam)
            cur_tr = [self._step(st, gam, x0, y, x.shape)[0] for x0, y, x in zip(cur_tr, Ytr, Xtr)]
            cur_va = [self._step(st, gam, x0, y, x.shape)[0] for x0, y, x in zip(cur_va, Yval, Xval)]
        if self.noise_aware:   # data-consistency weight mu chosen on validation
            best = None
            for mu in (0.0, 0.25, 0.5, 0.75, 1.0):
                self.mu = mu
                e = np.mean([np.mean((x - self.predict(y, x.shape, calibrated=False)) ** 2) for x, y in zip(Xval, Yval)])
                if best is None or e < best[0]:
                    best = (e, mu)
            self.mu = best[1]
        # calibrate uncertainty on validation
        num = den = 0.0
        for x, y in zip(Xval, Yval):
            xh, v = self.predict(y, x.shape, return_var=True, calibrated=False)
            m = v > 0
            num += ((x - xh) ** 2)[m].sum(); den += v[m].sum()
        self.cal = num / max(den, 1e-12)
        return self

    def _step(self, st, gam, x0, y, shape):
        d, v = st.predict(x0, shape)
        if self.gate:
            bands = band_split(d, self.op.ndim, self.B)
            d = sum(g * b for g, b in zip(gam, bands))
        return self._proj(x0 + d, y), v

    def predict(self, y, shape, return_var=False, calibrated=True):
        x = anchor_input(y, self.op, shape, self.mirror, getattr(self, 'anchor', 'mac'))
        v = np.zeros(shape)
        for st, gam in zip(self.stages, self.gammas):
            x, v = self._step(st, gam, x, y, shape)
        if isinstance(self.op, Decimation) and self.dc and self.mu == 1.0:
            v = v * (self.op.phase(shape) != 0)
        if return_var:
            return x, v * (self.cal if calibrated else 1.0)
        return x
