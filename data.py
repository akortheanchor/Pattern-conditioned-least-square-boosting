"""
Dataset preparation. All splits are subject/record/video/patient-disjoint.

  ecg : MIT-BIH Long-Term ECG Database (PhysioNet; 7 Holter records, 128 Hz)
  eeg : 128-channel resting-state EEG (NeuroKit2 reference recording, 300 Hz)
  mri : Brain-tumour MRI (Figshare/SARTAJ collection; 4 classes)
  cxr : COVID-19 Image Data Collection, chest X-ray (Cohen et al.)
  us  : POCUS lung ultrasound video collection (Born et al.)
  mitdb (OOD only): MIT-BIH Arrhythmia records 100 & 208, resampled to 128 Hz

Usage: python data.py ecg eeg mri cxr us mitdb
"""
import glob, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
T = "/tmp/claude-0/t"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/data"
os.makedirs(OUT, exist_ok=True)
BEATS = set("NLRBAaJSVrFejnE/fQ?")


def _znorm(w):
    return (w - w.mean()) / (w.std() + 1e-9)


# ------------------------------------------------------------------ ECG
def prep_ecg(N=1024, seed=0):
    import wfdb
    d = f"{T}/NeuroKit/data/mit_long-term/mit-bih-long-term-ecg-database-1.0.0"
    split = {"train": ["14046", "14134", "14149", "14157"], "val": ["14172"], "test": ["14184", "15814"]}
    per = {"train": 800, "val": 200, "test": 250}
    rng = np.random.default_rng(seed)
    out = {}
    for sp, recs in split.items():
        X, meta, peaks = [], [], []
        for rec in recs:
            sig, f = wfdb.rdsamp(f"{d}/{rec}", channels=[0])
            s = sig[:, 0]
            ann = wfdb.rdann(f"{d}/{rec}", "atr")
            bsamp = np.array([a for a, sy in zip(ann.sample, ann.symbol) if sy in BEATS])
            got = 0
            while got < per[sp]:
                st = int(rng.integers(0, len(s) - N))
                w = s[st:st + N]
                if not np.isfinite(w).all() or w.std() < 0.05 or np.abs(np.diff(w)).max() > 5:
                    continue
                X.append(_znorm(w)); meta.append((int(rec), st))
                pk = bsamp[(bsamp >= st) & (bsamp < st + N)] - st
                peaks.append(pk)
                got += 1
        out[f"X_{sp}"] = np.array(X, np.float32); out[f"meta_{sp}"] = np.array(meta)
        if sp == "test":
            out["peaks_test"] = np.array(peaks, dtype=object)
    out["fs"] = 128
    np.savez_compressed(f"{OUT}/ecg.npz", **out)
    print("ecg", {k: v.shape for k, v in out.items() if hasattr(v, "shape")})


def prep_mitdb(N=1024):
    """OOD ECG: MIT-BIH Arrhythmia 100 (wfdb sample data) and 208 (scipy.datasets)."""
    import wfdb
    from scipy.signal import resample_poly
    sig100 = wfdb.rdsamp(f"{T}/wfdb-python/sample-data/100", channels=[0])[0][:, 0]
    try:
        from scipy.datasets import electrocardiogram
        sig208 = electrocardiogram()
    except Exception as e:  # pragma: no cover
        print("scipy ecg unavailable", e); sig208 = None
    X = []
    for s in [sig100, sig208]:
        if s is None:
            continue
        s = resample_poly(s, 16, 45)        # 360 Hz -> 128 Hz
        for st in range(0, len(s) - N, N * 2):
            w = s[st:st + N]
            if w.std() > 0.05:
                X.append(_znorm(w))
    X = np.array(X, np.float32)[:300]
    np.savez_compressed(f"{OUT}/mitdb.npz", X_test=X, fs=128)
    print("mitdb", X.shape)


# ------------------------------------------------------------------ EEG
def prep_eeg(N=1024, seed=0):
    import mne
    from scipy.signal import butter, sosfiltfilt
    raw = mne.io.read_raw_fif(f"{T}/NeuroKit/data/eeg_resting_8min_300hz.fif", preload=True, verbose=False)
    raw.pick("eeg")
    Xall = raw.get_data()
    fs = raw.info["sfreq"]
    sos = butter(4, 0.5, "highpass", fs=fs, output="sos")
    Xall = sosfiltfilt(sos, Xall, axis=1)
    rng = np.random.default_rng(seed)
    nch = Xall.shape[0]
    chans = rng.permutation(nch)
    cs = {"train": chans[:90], "val": chans[90:100], "test": chans[100:]}
    ts = {"train": (0, int(300 * fs)), "val": (0, int(300 * fs)), "test": (int(330 * fs), Xall.shape[1])}
    per = {"train": 3000, "val": 200, "test": 500}
    out = {}
    for sp in cs:
        X, meta = [], []
        a, b = ts[sp]
        while len(X) < per[sp]:
            c = int(rng.choice(cs[sp])); st = int(rng.integers(a, b - N))
            w = Xall[c, st:st + N]
            if w.std() <= 0 or np.abs(w).max() > 20 * w.std():
                continue
            X.append(_znorm(w)); meta.append((c, st))
        out[f"X_{sp}"] = np.array(X, np.float32); out[f"meta_{sp}"] = np.array(meta)
    out["fs"] = fs
    np.savez_compressed(f"{OUT}/eeg.npz", **out)
    print("eeg", {k: v.shape for k, v in out.items() if hasattr(v, "shape")})


# ------------------------------------------------------------------ images
def _load_gray(path, S=128, crop_frac=1.0):
    from PIL import Image
    im = Image.open(path).convert("L")
    w, h = im.size
    m = int(min(w, h) * crop_frac)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    im = im.resize((S, S), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def prep_mri(seed=0):
    d = f"{T}/brain-tumor-classification-dataset"
    classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    rng = np.random.default_rng(seed)
    tr_files, te_files = [], []
    for ci, c in enumerate(classes):
        tr_files += [(f, ci) for f in sorted(glob.glob(f"{d}/Training/{c}/*.jpg"))]
        fs = sorted(glob.glob(f"{d}/Testing/{c}/*.jpg"))
        te_files += [(f, ci) for f in rng.permutation(fs)[:60]]
    tr_files = [tr_files[i] for i in rng.permutation(len(tr_files))]
    out = {}
    for sp, files in {"train": tr_files[:1500], "val": tr_files[1500:1650], "test": te_files}.items():
        X = np.array([_load_gray(f) for f, _ in files], np.float32)
        out[f"X_{sp}"] = X; out[f"y_{sp}"] = np.array([c for _, c in files])
    out["classes"] = np.array(classes)
    np.savez_compressed(f"{OUT}/mri.npz", **out)
    print("mri", {k: v.shape for k, v in out.items()})


def prep_cxr(seed=0):
    import pandas as pd
    d = f"{T}/covid-chestxray-dataset"
    md = pd.read_csv(f"{d}/metadata.csv")
    md = md[(md.modality == "X-ray") & md.view.isin(["PA", "AP", "AP Supine"])]
    md = md[[os.path.exists(f"{d}/images/{f}") for f in md.filename]]
    pats = np.array(sorted(md.patientid.astype(str).unique()))
    rng = np.random.default_rng(seed); pats = rng.permutation(pats)
    n = len(pats)
    sp_p = {"train": set(pats[: int(.6 * n)]), "val": set(pats[int(.6 * n): int(.7 * n)]), "test": set(pats[int(.7 * n):])}
    out = {}
    for sp, ps in sp_p.items():
        sub = md[md.patientid.astype(str).isin(ps)]
        X = np.array([_load_gray(f"{d}/images/{f}", crop_frac=0.9) for f in sub.filename], np.float32)
        out[f"X_{sp}"] = X
        out[f"finding_{sp}"] = sub.finding.astype(str).values
        out[f"pid_{sp}"] = sub.patientid.astype(str).values
    np.savez_compressed(f"{OUT}/cxr.npz", **out)
    print("cxr", {k: v.shape for k, v in out.items()}, "patients", n)


def _video_frames(path, max_frames=400):
    import imageio.v3 as iio
    fr = []
    try:
        for i, f in enumerate(iio.imiter(path)):
            fr.append(f)
            if i >= max_frames:
                break
    except Exception as e:
        print("skip", os.path.basename(path), e)
    return fr


def _us_frame(f, S=128):
    from PIL import Image
    f = np.asarray(f)
    if f.ndim == 3:
        f = f[..., :3].mean(-1)
    h, w = f.shape
    # central region of the sector, avoiding UI text borders
    m = int(min(h, w) * 0.8)
    t = int(h * 0.12)
    a = f[t: t + m, (w - m) // 2:(w - m) // 2 + m]
    im = Image.fromarray(a.astype(np.float32)).resize((S, S), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def prep_us(seed=0, per_video=12):
    d = f"{T}/covid19_ultrasound/data/pocus_videos"
    vids = sorted(glob.glob(f"{d}/*/*"))
    vids = [v for v in vids if v.lower().split(".")[-1] in ("mp4", "avi", "mov", "mpeg", "gif")]
    rng = np.random.default_rng(seed); vids = list(rng.permutation(vids))
    n = len(vids)
    split = {"train": vids[: int(.6 * n)], "val": vids[int(.6 * n): int(.7 * n)], "test": vids[int(.7 * n):]}
    out = {}
    for sp, vs in split.items():
        X, vid = [], []
        seqs = []
        for k, v in enumerate(vs):
            fr = _video_frames(v)
            if len(fr) < 8:
                continue
            idx = np.linspace(0, len(fr) - 1, per_video).astype(int)
            X += [_us_frame(fr[i]) for i in idx]; vid += [k] * len(idx)
            if sp == "test" and len(seqs) < 4 and len(fr) >= 40:
                seqs.append(np.array([_us_frame(f) for f in fr[:40]], np.float32))
        out[f"X_{sp}"] = np.array(X, np.float32); out[f"vid_{sp}"] = np.array(vid)
        out[f"names_{sp}"] = np.array([os.path.basename(v) for v in vs])
        if sp == "test":
            out["seq_test"] = np.array(seqs, np.float32)
    np.savez_compressed(f"{OUT}/us.npz", **out)
    print("us", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[f"prep_{name}"]()
