"""
Beat-Synchronous Least-Square Boost (BS-LSB) for low-rate ECG.

Key observation: the cardiac cycle is the 'varying pattern' that repeats, and
because RR intervals are not multiples of the sampling interval, every beat is
sampled at a different sub-sample phase.  Pooling the samples of all beats in a
window on a common, sub-sample-aligned time axis gives a beat template sampled
far more densely than the acquisition itself (multi-frame super-resolution in
time). BS-LSB:
  1. start from any full-rate estimate x1 (here PC-LSB),
  2. detect R peaks on x1, refine them to sub-sample precision,
  3. alternate: kernel-regress a template on the pooled, aligned samples;
     per-beat amplitude/offset by least squares; per-beat sub-sample shift by
     template matching,
  4. leave-one-beat-out (LOBO) self-validation: a beat is only replaced in
     proportion to how well the OTHER beats predict its own samples
     (ectopic/noisy beats fall back to x1),
  5. G-LSB consistency: x = p + MAC(z - p[::r]) restores the samples exactly
     with a band-limited (ringing-free near samples) correction.
"""
import numpy as np
from scipy.signal import find_peaks
from ops import mac_mirror


def _nw(tau, val, grid, h):
    """Nadaraya-Watson kernel regression on a grid."""
    W = np.exp(-0.5 * ((grid[:, None] - tau[None, :]) / h) ** 2)
    s = W.sum(1)
    return (W @ val) / np.maximum(s, 1e-9), s


def bs_lsb(z, r, x1, fs=128, eta=1.0, pre=0.25, post=0.40, h=0.6, lo=0.15, hi=0.45,
           iters=3, shift=0.15, return_info=False):
    N = len(x1)
    pos = np.arange(len(z)) * r
    med = np.median(x1)
    sgn = 1.0 if np.percentile(x1 - med, 99.5) >= -np.percentile(x1 - med, 0.5) else -1.0
    yy = sgn * (x1 - med)
    pk, _ = find_peaks(yy, distance=int(0.33 * fs), prominence=0.45 * np.percentile(yy, 99.5))
    info = {"n_beats": len(pk), "weights": []}
    if len(pk) < 3:
        return (x1, info) if return_info else x1
    # sub-sample refinement by parabolic interpolation
    t = pk.astype(float)
    for i, k in enumerate(pk):
        if 0 < k < N - 1:
            a, b, c = yy[k - 1], yy[k], yy[k + 1]
            den = a - 2 * b + c
            if den < 0:
                t[i] = k + 0.5 * (a - c) / den
    P, Q = int(pre * fs), int(post * fs)
    grid = np.arange(-P, Q + 1, dtype=float)

    def beat_samples(tj):
        m = (pos >= tj - P) & (pos <= tj + Q)
        return pos[m] - tj, z[m]

    alpha = np.ones(len(t)); beta = np.zeros(len(t))
    for _ in range(iters):
        taus, vals = [], []
        for j, tj in enumerate(t):
            ta, va = beat_samples(tj)
            taus.append(ta); vals.append((va - beta[j]) / alpha[j])
        T, dens = _nw(np.concatenate(taus), np.concatenate(vals), grid, h)
        for j, tj in enumerate(t):
            # sub-sample shift refinement by template matching
            best, bd = np.inf, 0.0
            for d in (np.linspace(-shift * r, shift * r, 8 * r + 1) if shift > 0 else [0.0]):
                ta, va = beat_samples(tj + d)
                if len(ta) < 4:
                    continue
                Tv = np.interp(ta, grid, T)
                A = np.c_[Tv, np.ones_like(Tv)]
                coef, *_ = np.linalg.lstsq(A, va, rcond=None)
                res = np.sum((va - A @ coef) ** 2)
                if res < best:
                    best, bd, cbest = res, d, coef
            t[j] += bd
            if np.isfinite(best):
                alpha[j], beta[j] = max(cbest[0], 0.2), cbest[1]
    # LOBO self-validation and reconstruction
    p = x1.copy(); acc = np.zeros(N); wsum = np.zeros(N)
    tg = np.arange(N, dtype=float)
    for j, tj in enumerate(t):
        others = [k for k in range(len(t)) if k != j]
        ta_o = np.concatenate([beat_samples(t[k])[0] for k in others])
        va_o = np.concatenate([(beat_samples(t[k])[1] - beta[k]) / alpha[k] for k in others])
        To, _ = _nw(ta_o, va_o, grid, h)
        ta, va = beat_samples(tj)
        if len(ta) < 4:
            info["weights"].append(0.0); continue
        pred = alpha[j] * np.interp(ta, grid, To) + beta[j]
        nr = np.linalg.norm(va - pred) / (np.linalg.norm(va - va.mean()) + 1e-9)
        w = float(np.clip((hi - nr) / (hi - lo), 0, 1))
        info["weights"].append(w)
        if w <= 0:
            continue
        m = (tg >= tj - P) & (tg <= tj + Q)
        rel = tg[m] - tj
        est = alpha[j] * np.interp(rel, grid, T) + beta[j]
        taper = np.sin(np.pi * (rel + P) / (P + Q)) ** 0.5   # smooth edges
        acc[m] += w * taper * (est - x1[m]); wsum[m] += 1
    p = x1 + eta * acc / np.maximum(wsum, 1)
    xh = p + mac_mirror(z - p[::r], r, 1)
    if not np.all(np.isfinite(xh)):
        xh = x1
    info["t"] = t
    return (xh, info) if return_info else xh


def fit_eta(Xval, op, x1_fn, grid=(0, 0.25, 0.5, 0.75, 1.0), **kw):
    """Choose the blending weight eta on validation windows (eta=0 -> x1)."""
    def snr(x, xh):
        return 10 * np.log10(np.sum((x - x.mean()) ** 2) / max(np.sum((x - xh) ** 2), 1e-20))
    pre = []
    for x in Xval:
        y = op.forward(x); pre.append((x, y, x1_fn(y, x.shape)))
    scores = {e: np.mean([snr(x, bs_lsb(y, op.r, x1, eta=e, **kw)) for x, y, x1 in pre]) for e in grid}
    return max(scores, key=scores.get), scores
