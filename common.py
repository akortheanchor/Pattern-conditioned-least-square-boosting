import os, pickle, time
import numpy as np
from skimage.metrics import structural_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND = {"ecg": 1, "eeg": 1, "mitdb": 1, "mri": 2, "cxr": 2, "us": 2}
NAMES = {"ecg": "MIT-BIH LTDB (ECG)", "eeg": "HD-EEG (128-ch)", "mri": "Brain MRI",
         "cxr": "COVID-19 CXR", "us": "POCUS lung US", "mitdb": "MIT-BIH Arrhythmia (OOD)"}


def load(ds):
    d = np.load(f"{ROOT}/data/{ds}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def metrics(x, xh, nd):
    err = x - xh
    mse = float(np.mean(err ** 2))
    out = {"nmse_db": 10 * np.log10(mse / max(np.mean(x ** 2), 1e-12) + 1e-20)}
    if nd == 1:
        out["snr"] = 10 * np.log10(np.sum((x - x.mean()) ** 2) / max(np.sum(err ** 2), 1e-20))
        out["prd"] = 100 * np.sqrt(np.sum(err ** 2) / np.sum((x - x.mean()) ** 2))
        out["q"] = out["snr"]
    else:
        out["psnr"] = 10 * np.log10(1.0 / max(mse, 1e-20))
        out["ssim"] = float(structural_similarity(x, np.clip(xh, 0, 1), data_range=1.0))
        out["q"] = out["psnr"]
    return out


def dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def undump(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
