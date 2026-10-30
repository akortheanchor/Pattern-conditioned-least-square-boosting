"""Downstream clinical-task probes evaluated on reconstructions."""
import numpy as np
from scipy.signal import find_peaks, butter, sosfiltfilt, welch


# ------------------------------------------------- MRI: tumour-type classification
class MRIClassifier:
    """HOG + logistic regression trained on FULLY-sampled training images only;
    a reconstruction is clinically useful if it keeps this classifier's decision."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))

    @staticmethod
    def feats(X):
        from skimage.feature import hog
        return np.array([hog(np.clip(x, 0, 1), orientations=9, pixels_per_cell=(12, 12),
                             cells_per_block=(2, 2)) for x in X])

    def fit(self, X, y):
        self.clf.fit(self.feats(X), y); return self

    def predict(self, X):
        return self.clf.predict(self.feats(X))


# ------------------------------------------------- ECG: R-peak detection
def detect_r(x, fs=128):
    """Fixed, simple QRS detector (band-pass + squared slope + adaptive peak picking)."""
    sos = butter(2, [5, min(30, fs / 2 - 1)], "bandpass", fs=fs, output="sos")
    f = sosfiltfilt(sos, x)
    e = np.convolve(np.gradient(f) ** 2, np.ones(int(0.1 * fs)) / int(0.1 * fs), "same")
    pk, _ = find_peaks(e, distance=int(0.3 * fs), height=0.3 * np.percentile(e, 99))
    # move to local extremum of the signal
    out = []
    for p in pk:
        a, b = max(0, p - int(0.06 * fs)), min(len(x), p + int(0.06 * fs) + 1)
        out.append(a + np.argmax(np.abs(x[a:b] - np.median(x))))
    return np.array(out)


def peak_f1(ref, det, tol):
    ref = np.asarray(ref); det = np.asarray(det)
    if len(ref) == 0 and len(det) == 0:
        return 1.0, 0.0
    used = np.zeros(len(det), bool); tp = 0; errs = []
    for rp in ref:
        if len(det) == 0:
            break
        d = np.abs(det - rp); d[used] = 1e9
        j = np.argmin(d)
        if d[j] <= tol:
            used[j] = True; tp += 1; errs.append(det[j] - rp)
    fp = len(det) - tp; fn = len(ref) - tp
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, float(np.mean(np.abs(errs))) if errs else np.nan


# ------------------------------------------------- EEG: relative band power
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def rel_bandpower(x, fs):
    f, P = welch(x, fs=fs, nperseg=min(256, len(x)))
    tot = P[(f >= 1) & (f <= 30)].sum() + 1e-20
    return np.array([P[(f >= a) & (f < b)].sum() / tot for a, b in BANDS.values()])


def make_hook(ds, D):
    """Returns f(i, x, xh) -> dict of downstream metrics for test sample i."""
    if ds == "ecg":
        fs = float(D["fs"]); peaks = D["peaks_test"]; tol = int(0.15 * fs)

        def hook(i, x, xh):
            f1, mae = peak_f1(peaks[i], detect_r(xh, fs), tol)
            return {"r_f1": f1, "r_mae_ms": mae * 1000 / fs if np.isfinite(mae) else np.nan}
        return hook
    if ds == "eeg":
        fs = float(D["fs"])

        def hook(i, x, xh):
            e = np.abs(rel_bandpower(xh, fs) - rel_bandpower(x, fs))
            return {"bp_err": float(e.mean()), **{f"bp_{k}": float(v) for k, v in zip(BANDS, e)}}
        return hook
    if ds == "mri":
        clf = MRIClassifier().fit(D["X_train"], D["y_train"])
        yt = D["y_test"]

        def hook(i, x, xh):
            return {"cls_ok": float(clf.predict([xh])[0] == yt[i])}
        return hook
    return None
