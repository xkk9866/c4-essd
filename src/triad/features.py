"""Light-weight window features for raw physiological streams (used for SenseCobot).

All functions return dicts of scalar features; NaN when a feature cannot be computed. The design goal is
robustness on short (60 s) windows rather than exhaustive feature sets: heart-rate variability from R peaks,
electrodermal tonic/phasic statistics, temperature, accelerometer energy, and camera-based AFFDEX emotion
statistics.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


# ----------------------------------------------------------------------------- heart rate variability
def detect_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """R-peak times (s) from a single ECG lead using neurokit2 if available, else a Pan-Tompkins-like fallback."""
    ecg = np.asarray(ecg, dtype=float)
    ecg = ecg - np.nanmean(ecg)
    ecg[~np.isfinite(ecg)] = 0.0
    try:
        import neurokit2 as nk
        clean = nk.ecg_clean(ecg, sampling_rate=int(fs), method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=int(fs), method="neurokit")
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
    except Exception:
        b, a = sps.butter(3, [5 / (fs / 2), 20 / (fs / 2)], btype="band")
        y = sps.filtfilt(b, a, ecg)
        y = np.gradient(y) ** 2
        y = np.convolve(y, np.ones(int(0.15 * fs)) / int(0.15 * fs), mode="same")
        peaks, _ = sps.find_peaks(y, distance=int(0.3 * fs), height=np.percentile(y, 90) * 0.3)
        peaks = peaks.astype(float)
    return peaks / fs


def clean_rr(t_peaks: np.ndarray):
    """Return (beat times, RR intervals in s) after removing physiologically implausible or ectopic beats."""
    t = np.asarray(t_peaks, dtype=float)
    if len(t) < 3:
        return t[1:], np.diff(t)
    rr = np.diff(t)
    ok = (rr > 0.3) & (rr < 2.0)
    med = np.median(rr[ok]) if ok.any() else np.nan
    if np.isfinite(med):
        ok &= np.abs(rr - med) < 0.3 * med
    return t[1:][ok], rr[ok]


def hrv_features(rr: np.ndarray, t_rr: np.ndarray, win: float, prefix: str = "hrv") -> dict:
    """Time- and frequency-domain HRV for one window (rr in s, t_rr = beat times in s)."""
    f = {}
    rr = np.asarray(rr, dtype=float)
    n = len(rr)
    f[f"{prefix}_n_beats"] = n
    if n < 6:
        for k in ("mean_nn", "sdnn", "rmssd", "pnn50", "hr_mean", "hr_std", "lf", "hf", "lfhf", "lf_n", "hf_n"):
            f[f"{prefix}_{k}"] = np.nan
        return f
    rr_ms = rr * 1000.0
    f[f"{prefix}_mean_nn"] = rr_ms.mean()
    f[f"{prefix}_sdnn"] = rr_ms.std(ddof=1)
    d = np.diff(rr_ms)
    f[f"{prefix}_rmssd"] = np.sqrt(np.mean(d ** 2)) if len(d) else np.nan
    f[f"{prefix}_pnn50"] = np.mean(np.abs(d) > 50) if len(d) else np.nan
    hr = 60.0 / rr
    f[f"{prefix}_hr_mean"] = hr.mean()
    f[f"{prefix}_hr_std"] = hr.std(ddof=1)
    # frequency domain on 4 Hz cubic-interpolated tachogram
    try:
        fs_i = 4.0
        tt = np.arange(t_rr[0], t_rr[-1], 1 / fs_i)
        if len(tt) >= 32:
            x = np.interp(tt, t_rr, rr_ms)
            x = x - x.mean()
            nper = min(len(x), 256)
            fr, p = sps.welch(x, fs=fs_i, nperseg=nper)
            lf = np.trapezoid(p[(fr >= 0.04) & (fr < 0.15)], fr[(fr >= 0.04) & (fr < 0.15)])
            hf = np.trapezoid(p[(fr >= 0.15) & (fr < 0.4)], fr[(fr >= 0.15) & (fr < 0.4)])
            f[f"{prefix}_lf"] = np.log1p(lf)
            f[f"{prefix}_hf"] = np.log1p(hf)
            f[f"{prefix}_lfhf"] = np.log((lf + 1e-6) / (hf + 1e-6))
            f[f"{prefix}_lf_n"] = lf / (lf + hf + 1e-9)
            f[f"{prefix}_hf_n"] = hf / (lf + hf + 1e-9)
        else:
            raise ValueError
    except Exception:
        for k in ("lf", "hf", "lfhf", "lf_n", "hf_n"):
            f[f"{prefix}_{k}"] = np.nan
    return f


# ----------------------------------------------------------------------------- electrodermal activity
def eda_decompose(x: np.ndarray, fs: float):
    """Tonic (SCL) / phasic (SCR) split by a 0.05 Hz low-pass; returns (tonic, phasic)."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nanmedian(x))
    if len(x) < int(5 * fs):
        return x, np.zeros_like(x)
    b, a = sps.butter(2, min(0.05 / (fs / 2), 0.99), btype="low")
    tonic = sps.filtfilt(b, a, x)
    return tonic, x - tonic


def eda_features(x: np.ndarray, fs: float, prefix: str = "eda") -> dict:
    f = {}
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < int(5 * fs):
        for k in ("mean", "std", "slope", "range", "scl_mean", "scr_std", "scr_n", "scr_amp_mean", "scr_amp_sum"):
            f[f"{prefix}_{k}"] = np.nan
        return f
    tonic, phasic = eda_decompose(x, fs)
    t = np.arange(len(x)) / fs
    f[f"{prefix}_mean"] = x.mean()
    f[f"{prefix}_std"] = x.std()
    f[f"{prefix}_slope"] = np.polyfit(t, x, 1)[0] if len(x) > 2 else np.nan
    f[f"{prefix}_range"] = x.max() - x.min()
    f[f"{prefix}_scl_mean"] = tonic.mean()
    f[f"{prefix}_scr_std"] = phasic.std()
    thr = max(0.01, 0.5 * phasic.std())
    peaks, props = sps.find_peaks(phasic, height=thr, distance=max(1, int(1.0 * fs)))
    f[f"{prefix}_scr_n"] = len(peaks) / (len(x) / fs) * 60.0  # per minute
    amps = props["peak_heights"] if len(peaks) else np.array([])
    f[f"{prefix}_scr_amp_mean"] = amps.mean() if len(amps) else 0.0
    f[f"{prefix}_scr_amp_sum"] = amps.sum() if len(amps) else 0.0
    return f


# ----------------------------------------------------------------------------- generic
def basic_stats(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_slope": np.nan, f"{prefix}_range": np.nan}
    t = np.arange(len(x))
    return {f"{prefix}_mean": x.mean(), f"{prefix}_std": x.std(),
            f"{prefix}_slope": np.polyfit(t, x, 1)[0] if len(x) > 2 else np.nan, f"{prefix}_range": x.max() - x.min()}


def acc_features(acc: np.ndarray, prefix: str = "acc") -> dict:
    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or len(acc) < 2:
        return {f"{prefix}_mag_mean": np.nan, f"{prefix}_mag_std": np.nan, f"{prefix}_energy": np.nan}
    mag = np.linalg.norm(acc, axis=1)
    return {f"{prefix}_mag_mean": np.nanmean(mag), f"{prefix}_mag_std": np.nanstd(mag),
            f"{prefix}_energy": np.nanmean(np.diff(mag) ** 2)}


def sliding_windows(t0: float, t1: float, win: float, step: float):
    s = t0
    while s + win <= t1 + 1e-9:
        yield s, s + win
        s += step
