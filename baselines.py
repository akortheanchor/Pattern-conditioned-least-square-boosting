"""
Classical, compressed-sensing and learned-linear baselines.
All functions take (y, op, shape) and return an estimate of x with shape `shape`
(single signal/image; batch handled by the runner).
"""
import numpy as np
import pywt
from scipy import ndimage
from scipy.fft import dctn, idctn
from sklearn.linear_model import orthogonal_mp

from ops import Decimation, KSpace


# ---------------------------------------------------------------- interpolators
def _kernel_matrix(n, r, kind):
    """Dense N x n interpolation matrix for samples at positions 0, r, 2r, ..."""
    N = n * r
    u = np.arange(N) / r                         # fine positions in sample units
    W = np.zeros((N, n))
    if kind == "nearest":
        W[np.arange(N), np.clip(np.round(u).astype(int), 0, n - 1)] = 1
        return W
    a = 3
    base = np.floor(u).astype(int)
    for off in range(-a + 1, a + 1):
        j = base + off
        d = u - j
        if kind == "linear":
            w = np.clip(1 - np.abs(d), 0, None)
        elif kind == "lanczos3":
            w = np.sinc(d) * np.sinc(d / a) * (np.abs(d) < a)
        jj = np.clip(j, 0, n - 1)
        np.add.at(W, (np.arange(N), jj), w)
    return W / W.sum(1, keepdims=True)


_KCACHE = {}


def interp(y, op, kind):
    mats = []
    for ax in range(op.ndim):
        n = y.shape[-op.ndim + ax]
        key = (n, op.r, kind)
        if key not in _KCACHE:
            _KCACHE[key] = _kernel_matrix(n, op.r, kind)
        mats.append(_KCACHE[key])
    if op.ndim == 1:
        return mats[0] @ y
    return mats[0] @ y @ mats[1].T


def spline3(y, op):
    """Cubic B-spline interpolation (exactly interpolating, mirror boundary)."""
    sh = [s * op.r for s in y.shape]
    grids = np.meshgrid(*[np.arange(s) / op.r for s in sh], indexing="ij")
    return ndimage.map_coordinates(y, grids, order=3, mode="mirror")


# ---------------------------------------------------------------- helpers
def _grad(x, ndim):
    if ndim == 1:
        return np.diff(x, append=x[-1:])[None]
    gx = np.diff(x, axis=0, append=x[-1:, :])
    gy = np.diff(x, axis=1, append=x[:, -1:])
    return np.stack([gx, gy])


def _div(p, ndim):
    if ndim == 1:
        q = p[0]
        d = np.empty_like(q); d[0] = q[0]; d[1:-1] = q[1:-1] - q[:-2]; d[-1] = -q[-2]
        return d
    px, py = p
    dx = np.empty_like(px); dx[0] = px[0]; dx[1:-1] = px[1:-1] - px[:-2]; dx[-1] = -px[-2]
    dy = np.empty_like(py); dy[:, 0] = py[:, 0]; dy[:, 1:-1] = py[:, 1:-1] - py[:, :-2]; dy[:, -1] = -py[:, -2]
    return dx + dy


def _x0(y, op, shape):
    return op.mac(y) if isinstance(op, Decimation) else op.pinv(y, shape)


# ---------------------------------------------------------------- Tikhonov
def tikhonov(y, op, shape, lam=1e-3, iters=150):
    """min ||A x - y||^2 + lam ||Lap x||^2 by conjugate gradients (soft data fit)."""
    nd = op.ndim

    def lap(x):
        return ndimage.laplace(x, mode="reflect")

    def AtA(x):
        if isinstance(op, Decimation):
            return op.pinv(op.forward(x), shape)
        return op.pinv(op.forward(x))

    def H(x):
        return AtA(x) + lam * lap(lap(x))

    b = op.pinv(y, shape)
    x = _x0(y, op, shape)
    rr = b - H(x); p = rr.copy(); rs = (rr * rr).sum()
    for _ in range(iters):
        Hp = H(p); a = rs / max((p * Hp).sum(), 1e-20)
        x += a * p; rr -= a * Hp
        rs2 = (rr * rr).sum()
        if rs2 < 1e-12:
            break
        p = rr + rs2 / rs * p; rs = rs2
    return x


# ---------------------------------------------------------------- CS-TV
def cs_tv(y, op, shape, iters=300, mu=1.0):
    """min TV(x) s.t. A x = y (exact data consistency), Chambolle-Pock PDHG."""
    nd = op.ndim
    L2 = 4.0 * nd
    tau = sigma = 0.99 / np.sqrt(L2)
    x = _x0(y, op, shape); xb = x.copy()
    p = np.zeros((nd,) + x.shape)
    for _ in range(iters):
        p = p + sigma * _grad(xb, nd)
        nrm = np.maximum(1.0, np.sqrt((p ** 2).sum(0)) / mu)
        p = p / nrm
        xn = op.project(x + tau * _div(p, nd), y)
        xb = 2 * xn - x; x = xn
    return x


# ---------------------------------------------------------------- CS-wavelet
def _wav(x, nd, wl="db4", lev=3):
    if nd == 1:
        c = pywt.wavedec(x, wl, mode="periodization", level=lev)
        arr, sl = pywt.coeffs_to_array(c)
    else:
        c = pywt.wavedec2(x, wl, mode="periodization", level=lev)
        arr, sl = pywt.coeffs_to_array(c)
    return arr, sl


def _iwav(arr, sl, nd, wl="db4"):
    if nd == 1:
        return pywt.waverec(pywt.array_to_coeffs(arr, sl, output_format="wavedec"), wl, mode="periodization")
    return pywt.waverec2(pywt.array_to_coeffs(arr, sl, output_format="wavedec2"), wl, mode="periodization")


def cs_wavelet(y, op, shape, iters=100):
    """l1-wavelet sparsity with hard data consistency: iterative soft-thresholding
    with continuation (Starck/Elad-style inpainting & CS-MRI)."""
    nd = op.ndim
    x = _x0(y, op, shape)
    arr, sl = _wav(x, nd)
    tmax = np.abs(arr).max() * 0.5
    for k in range(iters):
        t = tmax * (1 - k / iters) ** 2 + 1e-4
        arr, sl = _wav(x, nd)
        # keep approximation band untouched
        mask = np.ones_like(arr, bool)
        mask[tuple(sl[0])] = False
        arr = np.where(mask, np.sign(arr) * np.maximum(np.abs(arr) - t, 0), arr)
        x = op.project(_iwav(arr, sl, nd)[tuple(slice(0, s) for s in shape)], y)
    return x


# ---------------------------------------------------------------- OMP (patch)
def _dct_dict(p, atoms, nd):
    D1 = np.zeros((p, atoms))
    for k in range(atoms):
        v = np.cos(np.arange(p) * k * np.pi / atoms)
        if k > 0:
            v -= v.mean()
        D1[:, k] = v / np.linalg.norm(v)
    return D1 if nd == 1 else np.kron(D1, D1)


_DCACHE = {}


def omp_patch(y, op, shape, p=None, sparsity_frac=0.5):
    """Patch-wise OMP sparse coding in an overcomplete DCT dictionary using only the
    acquired samples of each patch (the slides' 'state of the art in CS')."""
    assert isinstance(op, Decimation)
    nd, r = op.ndim, op.r
    p = p or (8 if nd == 2 else 32)
    step = r if nd == 2 else 2 * r
    key = (p, nd)
    if key not in _DCACHE:
        _DCACHE[key] = _dct_dict(p, 2 * p if nd == 1 else int(1.5 * p), nd)
    D = _DCACHE[key]
    mloc = np.zeros((p,) * nd, bool)
    mloc[(slice(None, None, r),) * nd] = True
    mflat = mloc.ravel()
    Dm = D[mflat]
    nrm = np.linalg.norm(Dm, axis=0) + 1e-8
    k = max(1, int(sparsity_frac * mflat.sum()))
    pad = p                                   # multiple of r -> grid preserved
    ypad = np.pad(y, pad // r, mode="reflect")
    xp = np.zeros(tuple(s * r for s in ypad.shape))
    xp[(slice(None, None, r),) * nd] = ypad
    if nd == 1:
        starts = np.arange(0, xp.shape[0] - p + 1, step)
        P = np.stack([xp[s:s + p] for s in starts])
    else:
        si = np.arange(0, xp.shape[0] - p + 1, step); sj = np.arange(0, xp.shape[1] - p + 1, step)
        P = np.stack([xp[a:a + p, b:b + p].ravel() for a in si for b in sj])
    Y = P[:, mflat].T
    mu = Y.mean(0, keepdims=True)
    coef = orthogonal_mp(Dm / nrm, Y - mu, n_nonzero_coefs=k)
    rec = (D / nrm) @ coef + mu
    out = np.zeros_like(xp); cnt = np.zeros_like(xp)
    if nd == 1:
        for i, s in enumerate(starts):
            out[s:s + p] += rec[:, i]; cnt[s:s + p] += 1
        est = (out / np.maximum(cnt, 1))[pad:pad + shape[0]]
    else:
        i = 0
        for a in si:
            for b in sj:
                out[a:a + p, b:b + p] += rec[:, i].reshape(p, p); cnt[a:a + p, b:b + p] += 1; i += 1
        est = (out / np.maximum(cnt, 1))[pad:pad + shape[0], pad:pad + shape[1]]
    return op.project(est, y)


# ---------------------------------------------------------------- learned LMMSE (linear)
class LearnedLinear:
    """Learned linear MMSE estimator (Wiener filter): ridge regression from the MAC
    patch around each output sample to the MAC residual, one model per sampling
    phase. This is PC-LSB with a single anchor, no contrast normalisation."""

    def __init__(self, op, p, lam=1e-2, n_train=200000, seed=0):
        self.op, self.p, self.lam, self.n_train, self.seed = op, p, lam, n_train, seed

    def fit(self, X, Y_list_unused=None):
        from pclsb import extract_patches, sample_train
        rng = np.random.default_rng(self.seed)
        F, T, PH = sample_train(X, self.op, self.p, None, self.n_train, rng, normalize=False)
        self.W = {}
        for ph in np.unique(PH):
            m = PH == ph
            A = np.c_[F[m], np.ones(m.sum())]
            self.W[ph] = np.linalg.solve(A.T @ A + self.lam * len(A) * np.eye(A.shape[1]), A.T @ T[m])
        return self

    def predict(self, y, shape):
        from pclsb import extract_patches
        x0 = self.op.mac(y) if isinstance(self.op, Decimation) else self.op.pinv(y)
        F = extract_patches(x0, self.p, self.op.ndim).astype(np.float64)
        ph = self.op.phase(shape).ravel()
        res = np.zeros(F.shape[0])
        for k, W in self.W.items():
            m = ph == k
            res[m] = np.c_[F[m], np.ones(m.sum())] @ W
        return self.op.project(x0 + res.reshape(shape), y)
