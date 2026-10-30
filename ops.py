"""
Acquisition operators and the faithful re-implementation of MAC
(Maximum Accuracy Computing, X. Liu) as described in the slides:

  (1) samples z of x,  (2) learn the pattern d = P z with the real-valued
  Fourier transform (RFT), (3) Least Square Boost: zero-pad d to the full
  dimension (b = [d; 0]) and invert the transform.

For uniform point samples, (1)-(3) is exactly trigonometric interpolation:
z is kept intact inside x* (slide: "z ⊂ x*"), and x* is band-limited.
The RFT is the orthonormal real DFT (verified in audit_claims.py), so it is
implemented with numpy's FFT; Nyquist bins are split symmetrically so that
the samples remain intact for even lengths too.
"""
import numpy as np


# ----------------------------------------------------------------------------
# MAC: spectral zero-padding (RFT / Least Square Boost) along one axis
# ----------------------------------------------------------------------------
def _trig_interp_axis(z, r, axis):
    z = np.moveaxis(np.asarray(z, float), axis, -1)
    n = z.shape[-1]
    N = n * r
    Z = np.fft.fft(z, axis=-1)
    P = np.zeros(z.shape[:-1] + (N,), complex)
    h = n // 2
    if n % 2 == 1:
        P[..., : h + 1] = Z[..., : h + 1]
        P[..., N - h:] = Z[..., h + 1:]
    else:  # split the Nyquist coefficient so the result stays real & interpolating
        P[..., :h] = Z[..., :h]
        P[..., N - h + 1:] = Z[..., h + 1:]
        P[..., h] = Z[..., h] / 2
        P[..., N - h] = Z[..., h] / 2
    x = np.fft.ifft(P, axis=-1).real * r
    return np.moveaxis(x, -1, axis)


def mac_rft_1d(z, r, N=None):
    x = _trig_interp_axis(z, r, -1)
    return x if N is None else x[..., :N]


def mac_rft_2d(z, r):
    """Separable 2-D MAC for z = x[::r, ::r] (batch dims allowed in front)."""
    return _trig_interp_axis(_trig_interp_axis(z, r, -1), r, -2)


def mac_mirror(z, r, ndim, pad=8):
    """Boundary-aware MAC (ours, no learning): reflect-pad the samples before the
    transform so the periodic RFT does not wrap the two ends together."""
    if ndim == 1:
        zp = np.pad(z, [(0, 0)] * (z.ndim - 1) + [(pad, pad)], mode="reflect")
        x = mac_rft_1d(zp, r)
        return x[..., pad * r: pad * r + z.shape[-1] * r]
    zp = np.pad(z, [(0, 0)] * (z.ndim - 2) + [(pad, pad), (pad, pad)], mode="reflect")
    x = mac_rft_2d(zp, r)
    s = pad * r
    return x[..., s: s + z.shape[-2] * r, s: s + z.shape[-1] * r]


# ----------------------------------------------------------------------------
# Generic operators: forward, zero-filled pseudo-inverse, data-consistency projection
# ----------------------------------------------------------------------------
class Decimation:
    """Uniform deterministic point sampling x[::r] (1-D) or x[::r, ::r] (2-D)."""

    def __init__(self, r, ndim):
        self.r, self.ndim = r, ndim
        self.rate = 1.0 / r ** ndim

    def sl(self):
        return (Ellipsis, slice(None, None, self.r)) if self.ndim == 1 else \
            (Ellipsis, slice(None, None, self.r), slice(None, None, self.r))

    def mask(self, shape):
        m = np.zeros(shape[-self.ndim:], bool)
        m[self.sl()] = True
        return m

    def forward(self, x):
        return x[self.sl()].copy()

    def pinv(self, y, shape):  # zero filling == literal LSB on time-sample curtailment
        x = np.zeros(shape)
        x[self.sl()] = y
        return x

    def project(self, p, y, mu=1.0):
        out = p.copy()
        out[self.sl()] = (1 - mu) * p[self.sl()] + mu * y
        return out

    def mac(self, y):
        return mac_rft_1d(y, self.r) if self.ndim == 1 else mac_rft_2d(y, self.r)

    def phase(self, shape):
        """Index of each output sample relative to the sampling grid (0 = sampled)."""
        if self.ndim == 1:
            return np.arange(shape[-1]) % self.r
        i = np.arange(shape[-2]) % self.r
        j = np.arange(shape[-1]) % self.r
        return (i[:, None] * self.r + j[None, :])


class KSpace:
    """Cartesian MRI k-space sampling with a Hermitian-symmetric line mask
    (the 'symmetric curtailment' of the slides). Orthonormal 2-D DFT."""

    def __init__(self, mask):
        self.m = mask.astype(bool)
        self.rate = self.m.mean()
        self.ndim = 2

    def forward(self, x):
        return np.fft.fft2(x, norm="ortho") * self.m

    def pinv(self, y, shape=None):
        return np.fft.ifft2(y * self.m, norm="ortho").real

    def mac(self, y):  # zero-filled inverse == MAC/LSB for spectral curtailment
        return self.pinv(y)

    def project(self, p, y, mu=1.0):
        P = np.fft.fft2(p, norm="ortho")
        P = np.where(self.m, (1 - mu) * P + mu * y, P)
        return np.fft.ifft2(P, norm="ortho").real

    def phase(self, shape):
        return np.zeros(shape[-2:], int)


def kspace_mask(H, W, accel, kind, seed=0, center_frac=0.08):
    """Phase-encode (row) masks; every mask is made Hermitian-symmetric."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(H) * H            # integer frequencies in FFT order
    n_lines = int(round(H / accel))
    sel = np.zeros(H, bool)
    if kind == "lowpass":                 # PDF: deterministic high-frequency elimination
        order = np.argsort(np.abs(ky), kind="stable")
        sel[order[:n_lines]] = True
    elif kind == "equispaced":            # deterministic uniform + calibration centre
        nc = int(round(H * center_frac))
        sel[np.argsort(np.abs(ky), kind="stable")[:nc]] = True
        step = max(1, int(np.floor((H - nc) / max(n_lines - nc, 1))))
        sel[np.isin(np.abs(ky).astype(int) % step, [0])] = True
    elif kind == "vardens":               # CS-standard variable-density random
        nc = int(round(H * center_frac))
        sel[np.argsort(np.abs(ky), kind="stable")[:nc]] = True
        p = (1 - np.abs(ky) / (H / 2)) ** 3 + 1e-3
        p[sel] = 0
        half = np.where(ky > 0)[0]
        need = max(0, (n_lines - sel.sum()) // 2)
        pick = rng.choice(half, need, replace=False, p=p[half] / p[half].sum())
        sel[pick] = True
    sel = sel | sel[(-np.arange(H)) % H]  # Hermitian symmetry (ky <-> -ky)
    return np.repeat(sel[:, None], W, axis=1)
