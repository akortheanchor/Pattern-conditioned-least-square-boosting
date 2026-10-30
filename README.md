# PC-LSB — Pattern-Conditioned Least-Square Boost

Code, results, figures and animations for the study *"PC-LSB: Pattern-Conditioned Least-Square Boost for Under-Sampled Medical Signals and Images — with a reproducible audit of Maximum Accuracy Computing"*.

## Layout
```
src/            all code (NumPy / SciPy / scikit-learn / PyWavelets; JAX only for the CNN baselines)
  ops.py          sampling operators + faithful MAC (RFT spectral zero-padding / Least Square Boost)
  pclsb.py        PC-LSB (G-LSB projection, linear Wiener backbone, sign-invariant pattern codebook,
                  per-(anchor, phase) ridge, frequency gate, boosting, uncertainty)
  ecg_bs.py       beat-synchronous LSB for ECG (LOBO self-validated)
  baselines.py    zero-fill, nearest, linear, cubic spline, Lanczos-3, Tikhonov, CS-TV, CS-Wavelet, OMP-DCT, LMMSE
  cnn.py          SRCNN and VDSR-style CNN (JAX), early stopping on validation
  data.py         dataset download/preparation with subject-disjoint splits
  audit_claims.py E1  numerical audit of the MAC claims
  run_benchmark.py E2 main benchmark (5 datasets x rates, 14 methods, 3 seeds, downstream tasks)
  run_kspace.py   E3  accelerated MRI with deterministic vs random k-space masks
  run_extra.py    E4-E9 ablation, noise, out-of-distribution, data efficiency, CNN plug-in
  sweep.py        validation-only hyper-parameter selection
  stats.py        Friedman, Holm-corrected Wilcoxon, bootstrap CIs, Nemenyi CD
  figures.py      all PNG figures;  gifs.py  all GIF animations
results/        every per-sample CSV, statistics tables (stats_*.csv, stats_summary.json), audit JSON
figs/           PNG figures (E1, F2-F15)
gifs/           G1 ultrasound video, G2 k-space sweep, G3 boosting stages, G4 ECG beat pooling
```

## Datasets (public; fetched from GitHub mirrors)
| Dataset | Source |
|---|---|
| MIT-BIH Long-Term ECG (7 records) | PhysioNet `ltdb`, mirrored in github.com/neuropsychology/NeuroKit |
| 128-channel resting EEG (8 min, 300 Hz) | github.com/neuropsychology/NeuroKit (`data/eeg_resting_8min_300hz.fif`) |
| Brain-tumour MRI (4 classes) | github.com/sartajbhuvaji/brain-tumor-classification-dataset |
| COVID-19 Image Data Collection (CXR) | github.com/ieee8023/covid-chestxray-dataset |
| POCUS lung ultrasound videos | github.com/jannisborn/covid19_ultrasound |
| MIT-BIH Arrhythmia 100 / 208 (OOD) | github.com/MIT-LCP/wfdb-python sample data; `scipy.datasets.electrocardiogram` |

`src/data.py` expects the repositories cloned under `/tmp/claude-0/t/` (edit `T` at the top to change).

## Reproduce
```bash
pip install numpy scipy scikit-learn scikit-image pywavelets jax optax wfdb mne pandas statsmodels imageio matplotlib
python src/data.py ecg eeg mri cxr us mitdb
python src/audit_claims.py
python src/run_benchmark.py ecg 2,4,8 --seeds 3
python src/run_benchmark.py eeg 2,4,8 --seeds 3
python src/run_benchmark.py mri 2,4 --seeds 3
python src/run_benchmark.py cxr 2,4 --seeds 3
python src/run_benchmark.py us 2,4 --seeds 3
python src/run_kspace.py 4,8
python src/run_extra.py ablation mri 4 ; python src/run_extra.py ablation eeg 4
python src/run_extra.py noise mri 2   ; python src/run_extra.py noise ecg 4
python src/run_extra.py plugin mri 2  ; python src/run_extra.py plugin cxr 2
python src/run_extra.py dataeff mri 4 ; python src/run_extra.py ood
python src/stats.py && python src/figures.py && python src/gifs.py
```
Runs are resumable (finished configurations and trained models are reused). Full pipeline: about 8 CPU-hours on 2 cores.

## Headline results (from results/stats_summary.json)
* Average rank over 12 dataset x rate configurations: PC-LSB 2.08, VDSR 3.08, Tikhonov 4.42, ..., MAC 9.92 (Friedman p = 1e-19).
* MAC loses to cubic-spline interpolation in 12 / 12 configurations.
* Accelerated MRI: random variable-density sampling + CS-TV ties MAC at R = 4 and beats it by 0.51 dB at R = 8.
* PC-LSB as a plug-in on VDSR: +0.20 dB (MRI) and +0.36 dB (CXR) over VDSR.

Research code; not validated for clinical use.
