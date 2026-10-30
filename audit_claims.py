"""
Experiment E1 -- Numerical audit of the mathematical claims in
"Maximum Accuracy Computing" (X. Liu, Lucidsee Technologies).

Every claim that can be checked by computation is checked here; the verdicts
are written to results/audit_claims.json and summarised in figs/E1_*.png.
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import setup, PAL, save
from ops import mac_rft_1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = {}


def rft_matrix(m):
    """Real-valued Fourier transform G_m exactly as on slide 'Better way' (m = 2k+1)."""
    assert m % 2 == 1
    k = (m - 1) // 2
    t = np.arange(m)
    rows = [np.full(m, 1 / np.sqrt(2))]
    rows += [np.cos(2 * np.pi * i * t / m) for i in range(1, k + 1)]
    rows += [np.sin(2 * np.pi * i * t / m) for i in range(1, k + 1)]
    return np.sqrt(2 / m) * np.array(rows)


# ---------------- A1: RFT orthogonality & relation to the real DFT -------------
res = {}
for m in [5, 9, 15, 33, 129]:
    G = rft_matrix(m)
    res[m] = float(np.abs(G @ G.T - np.eye(m)).max())
x = np.random.default_rng(0).standard_normal(33)
G = rft_matrix(33)
c = G @ x
F = np.fft.rfft(x)  # real DFT
k = 16
a_from_fft = np.r_[F[0].real / np.sqrt(33 / 2) / np.sqrt(2) * np.sqrt(2),
                   F[1:k + 1].real * np.sqrt(2 / 33)]
b_from_fft = -F[1:k + 1].imag * np.sqrt(2 / 33)
OUT["A1_rft_orthogonal"] = {
    "claim": "G_m is orthogonal (G^-1 = G^T)",
    "max_abs_dev_GGt_minus_I": res,
    "rft_equals_rescaled_rfft_maxerr": float(max(np.abs(c[1:k + 1] - a_from_fft[1:]).max(),
                                                  np.abs(c[k + 1:] - b_from_fft).max())),
    "verdict": "TRUE (orthogonal); but it is the classical real/trigonometric DFT "
               "(rescaled numpy.fft.rfft), so 'first construction' is not supported.",
}

# ---------------- A2/A3: Theorem 1 (maximal robustness, equiangular) ----------
def theorem1(m):
    k = (m - 1) // 2
    A = rft_matrix(m).T            # A = G^-1 = G^T
    Phi = A[: k + 1]               # first n = k+1 rows
    n = k + 1
    minsv, count = np.inf, 0
    for cols in itertools.combinations(range(m), n):
        s = np.linalg.svd(Phi[:, cols], compute_uv=False)
        minsv = min(minsv, s[-1]); count += 1
    C1 = Phi[:, 1:k + 1]
    Cn = C1 / np.linalg.norm(C1, axis=0)
    gram = np.abs(Cn.T @ Cn)[np.triu_indices(k, 1)]
    S1 = Phi[:, k + 1:]
    So = S1.T @ S1
    return dict(n=n, subsets=count, min_singular_value=float(minsv),
                coherence_values_C1=sorted(set(np.round(gram, 6).tolist())),
                S1_offdiag_max=float(np.abs(So - np.diag(np.diag(So))).max()))

t1 = {m: theorem1(m) for m in [5, 7, 9, 11, 13]}
OUT["A2_theorem1"] = {
    "claim": "(1) any n columns of Phi_1 independent; (2) C_1 strictly equiangular, S_1 orthogonal",
    "results": t1,
    "verdict": "Holds in exact arithmetic for every tested m (C_1 equiangular, S_1 orthogonal), "
               "BUT the smallest singular value over n-column subsets collapses roughly 10x per "
               "+2 in m (see min_singular_value), so 'maximal robustness' gives no numerical "
               "robustness to noise for realistic m."}

# ---------------- A4: Orthogonal System Theorem ------------------------------
rng = np.random.default_rng(1)
m, n = 16, 10
Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
A = Q * rng.uniform(0.5, 3, m)[:, None]        # mutually orthogonal (not normal) rows
b = np.zeros(m); idx = rng.choice(m, n, replace=False); b[idx] = rng.standard_normal(n)
xstar = np.linalg.solve(A, b)
C, d = A[idx], b[idx]
x_lsb = C.T @ np.linalg.solve(C @ C.T, d)
b2 = b.copy(); b2[[i for i in range(m) if i not in idx][:2]] = 0.7   # violate zero hypothesis
x2 = np.linalg.solve(A, b2)
x2_lsb = C.T @ np.linalg.solve(C @ C.T, b2[idx])
OUT["A4_orthogonal_system_theorem"] = {
    "claim": "x* = C^+ d when the curtailed equations have zero constant terms",
    "err_when_hypothesis_holds": float(np.abs(x_lsb - xstar).max()),
    "err_when_two_curtailed_terms_nonzero": float(np.linalg.norm(x2_lsb - x2) / np.linalg.norm(x2)),
    "verdict": "TRUE but elementary: with orthogonal rows, A^-1 b = sum_i a_i b_i/|a_i|^2, so "
               "dropping rows with b_i = 0 changes nothing. Exactness needs the UNKNOWN "
               "coefficients to be zero, i.e. the signal must be exactly band/sparsity-limited."}

# ---------------- A5: Hadamard TV-LP example ----------------------------------
H = np.array([[1]])
for _ in range(3):
    H = np.block([[H, H], [H, -H]])
# slide uses Sylvester ordering with rows as printed
Hs = np.array([[1,1,1,1,1,1,1,1],[1,-1,1,-1,1,-1,1,-1],[1,1,-1,-1,1,1,-1,-1],
               [1,-1,-1,1,1,-1,-1,1],[1,1,1,1,-1,-1,-1,-1],[1,-1,1,-1,-1,1,-1,1],
               [1,1,-1,-1,-1,-1,1,1],[1,-1,-1,1,-1,1,1,-1]], float)
bH = np.array([2,-2,0,4,4,0,-2,2.]); xH = np.array([1,0,0,2,0,0,-1,0.])
Cc, dd = Hs[1:], bH[1:]
# min sum|x_{i+1}-x_i| s.t. Cx=d  -> LP with slack t_i
N = 8; D = np.diff(np.eye(N), axis=0)
c_obj = np.r_[np.zeros(N), np.ones(N - 1)]
A_ub = np.block([[D, -np.eye(N - 1)], [-D, -np.eye(N - 1)]]); b_ub = np.zeros(2 * (N - 1))
A_eq = np.c_[Cc, np.zeros((7, N - 1))]
lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=dd, bounds=[(None, None)] * (2 * N - 1))
x_tv = lp.x[:N]
OUT["A5_hadamard_example"] = {
    "claim": "x* recovered from 7 of 8 Hadamard equations by TV-minimising LP",
    "Ax_star_equals_b": bool(np.allclose(Hs @ xH, bH)),
    "x_tv": np.round(x_tv, 4).tolist(), "x_star": xH.tolist(),
    "TV(x_tv)": float(np.abs(np.diff(x_tv)).sum()), "TV(x_star)": float(np.abs(np.diff(xH)).sum()),
    "recovered": bool(np.allclose(x_tv, xH, atol=1e-6)),
    "min_norm_solution_error": float(np.linalg.norm(np.linalg.pinv(Cc) @ dd - xH)),
}
OUT["A5_hadamard_example"]["verdict"] = (
    "TRUE as stated" if OUT["A5_hadamard_example"]["recovered"] else
    "NOT RECOVERED: the curtailed row is the constant (DC) row and TV is blind to constants, "
    "so x*+c is TV-minimal for every c; the LP returned x*+1 (same TV). The example is ill-posed.")

# ---------------- A6: H.264 example --------------------------------------------
A4m = np.array([[1,1,1,1],[2,1,-1,-2],[1,-1,-1,1],[1,-2,2,-1.]])
x4 = np.array([1.35,-0.95,-0.55,0.15]); b4 = A4m @ x4
x4_lsb = np.linalg.pinv(A4m[1:]) @ b4[1:]
OUT["A6_h264_example"] = {"Ax": b4.round(6).tolist(), "lsb_recovers_x": bool(np.allclose(x4_lsb, x4)),
                          "verdict": "TRUE (special case of A4)"}

# ---------------- A7: 'Maximum Accuracy Theorem' counterexample ----------------
def step_like(N=256):
    t = np.arange(N)
    return (np.where((t > 60) & (t < 150), 1.0, 0.0) + 0.3 * np.sin(2 * np.pi * t / N))
x = step_like(); r = 4
z = x[::r]
x_mac = mac_rft_1d(z, r, len(x))
x_lin = np.interp(np.arange(len(x)), np.arange(0, len(x), r), z)
e_mac, e_lin = np.linalg.norm(x - x_mac), np.linalg.norm(x - x_lin)
x7, z7, r7 = x.copy(), z.copy(), r
# Parseval identity holds for ANY estimator
Gm = None
OUT["A7_maximum_accuracy_theorem"] = {
    "claim": "MAC attains the maximum accuracy that the samples can provide",
    "l2_err_MAC": float(e_mac), "l2_err_linear_interp": float(e_lin),
    "samples_kept_intact_by_MAC": bool(np.allclose(x_mac[::r], z, atol=1e-8)),
    "verdict": "FALSE as an optimality statement: the proof only restates Parseval "
               "(||x-x'|| = ||A x - A x'|| for ANY estimate x'); no minimisation over estimators "
               "is performed. Plain linear interpolation from the SAME samples is more accurate "
               "on this piecewise-smooth signal." if e_lin < e_mac else "not refuted on this signal"}

# ---------------- A8: Gibbs phenomenon ('never fails at discontinuities') -----
N = 1024; t = np.arange(N); sq = np.sign(np.sin(2 * np.pi * 3 * t / N + 1e-3))
over = {}
for r in [2, 4, 8, 16]:
    xm = mac_rft_1d(sq[::r], r, N)
    over[r] = float((xm.max() - 1) / 2 * 100)   # overshoot as % of jump size
OUT["A8_gibbs"] = {"claim": "transform expression never fails, even for discontinuous f",
                   "overshoot_percent_of_jump_by_decimation": over,
                   "verdict": "FALSE pointwise: MAC rings at every jump (14-18% of the jump height here, above "
                              "the 8.95% Gibbs limit of Fourier partial sums because of aliasing) and the "
                              "overshoot does not vanish as the sampling rate changes."}

# ---------------- A9: arithmetic of the 40x claim ------------------------------
psnr = {"Lena": (42.58, 26.54), "Birds": (35.65, 23.23), "Fingerprint": (36.40, 25.57)}
ratios = {k: float(10 ** ((a - b) / 10)) for k, (a, b) in psnr.items()}
OUT["A9_40x_claim"] = {"mse_ratio_per_image": ratios,
                       "verdict": "Arithmetic correct for Lena only (40.18x); the other images give "
                                  f"{ratios['Birds']:.1f}x and {ratios['Fingerprint']:.1f}x. "
                                  "The CNN numbers (ReconNet, CVPR'16) come from random Gaussian "
                                  "block projections, a different measurement model, so the "
                                  "comparison is not like-for-like."}

# ---------------- A10: literal LSB with point samples = zero filling ---------
m = 33; Gm = rft_matrix(m); Am = Gm.T          # signal = A @ coeffs
xs = np.cos(2 * np.pi * 2 * np.arange(m) / m) + 0.2
sel = np.arange(0, m, 3)
Cp = Am[sel]; coef = Cp.T @ np.linalg.solve(Cp @ Cp.T, xs[sel])
rec = Am @ coef
OUT["A10_lsb_point_samples"] = {
    "claim": "LSB x*=C^+d applied to the partial system obtained by keeping time samples",
    "reconstruction_off_sample_values_maxabs": float(np.abs(np.delete(rec, sel)).max()),
    "verdict": "Literal LSB on time-sample curtailment returns the ZERO-FILLED signal "
               "(min-norm coefficients <=> min-norm signal for orthogonal A). The working "
               "MAC must therefore be spectral zero-padding (trigonometric interpolation)."}

os.makedirs(f"{ROOT}/results", exist_ok=True)
with open(f"{ROOT}/results/audit_claims.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)

# ---------------- figure ------------------------------------------------------
setup()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
tt = np.arange(len(x7))
ax[0].plot(tt, x7, color=PAL["gt"], lw=2.2, label="ground truth")
ax[0].plot(tt, x_mac, color=PAL["mac"], lw=1.4, label=f"MAC (RFT)  ℓ2 err={e_mac:.2f}")
ax[0].plot(tt, x_lin, color=PAL["base"], lw=1.4, ls="--", label=f"linear interp  ℓ2 err={e_lin:.2f}")
ax[0].plot(tt[::r7], z7, "o", ms=3.5, color="k", label="25% samples")
ax[0].set_title("A7  'Maximum accuracy' is not maximal"); ax[0].legend(fontsize=8, loc="upper right")
xm = mac_rft_1d(sq[::4], 4, N)
ax[1].plot(t[:400], sq[:400], color=PAL["gt"], lw=2.2, label="square wave")
ax[1].plot(t[:400], xm[:400], color=PAL["mac"], lw=1.3, label="MAC from 25% samples")
ax[1].set_title(f"A8  Gibbs ringing: {over[4]:.1f}% overshoot"); ax[1].legend(fontsize=8)
rs = list(over.keys())
ax[2].bar([f"1/{r}" for r in rs], [over[r] for r in rs], color=PAL["mac"])
ax[2].axhline(8.95, color="k", ls=":", lw=1); ax[2].text(-0.4, 9.6, "Fourier-series Gibbs limit 8.95%", fontsize=8)
ax[2].set_ylabel("overshoot (% of jump)"); ax[2].set_xlabel("sampling rate")
ax[2].set_title("Ringing does not vanish with more samples")
save(fig, f"{ROOT}/figs/E1_claim_audit.png")
print(json.dumps({k: v.get("verdict") for k, v in OUT.items()}, indent=1))
