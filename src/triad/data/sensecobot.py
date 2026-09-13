"""SenseCobot (Zenodo, CC-BY 4.0): 21 participants, baseline video + 5 cobot-programming tasks of increasing complexity.

Modalities used: Shimmer3 ECG (512 Hz, lead LL-RA), Shimmer3 GSR (128 Hz), Empatica E4 (1 Hz merged: EDA, HR, TEMP,
ACC), AFFDEX facial emotions (~15 Hz).  Self-report: NASA-TLX short form, two 7-point items per task
(A = mental/effort, B = physical).  The dataset also ships a per-task STRESS/NO-STRESS label that is derived from
the questionnaire (it varies within participant across tasks), which we keep as q_dslabel for reference.

Design condition: Baseline = 0 (relaxing/emotional videos), Task 1-5 = 1 (programming tasks).  `complexity`
stores the task index 0..5 so that TRIAD can estimate per-level manipulation efficacy.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from triad.data import raw_path
from triad.features import (acc_features, basic_stats, clean_rr, detect_rpeaks, eda_features, hrv_features,
                            sliding_windows)

TASKS = ["Baseline", "Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]
AFFDEX_COLS = ["Anger", "Contempt", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Engagement", "Valence",
               "Confusion", "Neutral", "Attention"]


def _find(root, sub, prefix, task, pid):
    pats = [os.path.join(root, sub, f"{prefix}_{task}_P_{pid}.csv"), os.path.join(root, sub, f"*{task}_P_{pid}.csv")]
    for p in pats:
        g = glob.glob(p)
        if g:
            return g[0]
    return None


def _load(path, cols):
    if path is None:
        return None
    try:
        df = pd.read_csv(path, usecols=lambda c: c == "Timestamp" or c in cols or c in ("Label", "SourceStimuliName"))
    except Exception as e:  # pragma: no cover
        print(f"  ! failed reading {path}: {e}")
        return None
    df["t"], df.attrs["time_kind"] = _parse_time(df["Timestamp"])
    return df


def _parse_time(s: pd.Series):
    """Return (seconds, kind).  kind = 'absolute' (seconds since epoch) or 'relative' for Excel-mangled 'MM:SS.f'
    stamps (about 40% of the Shimmer files), which are unwrapped into seconds within the recording hour and later
    anchored to an absolute reference modality by `_anchor`."""
    t = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    if t.isna().any():
        t = t.fillna(pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce"))
    if t.notna().mean() < 0.5:
        parts = s.astype(str).str.extract(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$")
        sec = parts[0].astype(float).to_numpy() * 60 + parts[1].astype(float).to_numpy()
        idx = np.arange(len(sec))
        ok = np.isfinite(sec)
        if ok.sum() < 2:
            return np.full(len(sec), np.nan), "relative"
        sec = np.interp(idx, idx[ok], sec[ok])
        wrap = np.concatenate([[0.0], np.cumsum(np.where(np.diff(sec) < -1800, 3600.0, 0.0))])
        return sec + wrap, "relative"
    v = t.astype("int64").to_numpy().astype(float) / 1e9
    v[t.isna().to_numpy()] = np.nan
    if np.isnan(v).any():
        idx = np.arange(len(v))
        ok = ~np.isnan(v)
        v = np.interp(idx, idx[ok], v[ok]) if ok.sum() > 1 else v
    return v, "absolute"


def _anchor(mods):
    """Convert 'relative' (MM:SS within hour) streams to absolute time using the earliest absolute stream as anchor."""
    abs_starts = [d["t"].iloc[0] for d in mods if d is not None and d.attrs.get("time_kind") == "absolute" and np.isfinite(d["t"].iloc[0])]
    for d in mods:
        if d is None or d.attrs.get("time_kind") != "relative" or not np.isfinite(d["t"].iloc[0]):
            continue
        sec = d["t"].to_numpy()
        if abs_starts:
            ref = min(abs_starts)
            hour0 = np.floor(ref / 3600.0) * 3600.0
            cands = [hour0 + sec[0] + k * 3600.0 for k in (-1, 0, 1)]
            start = min(cands, key=lambda c: abs(c - ref))
            d["t"] = start + (sec - sec[0])
        d.attrs["time_kind"] = "absolute" if abs_starts else "relative"


def read_nasa(root):
    p = os.path.join(root, "Additional_Information", "NASA_TLX.csv")
    df = pd.read_csv(p, index_col=0)
    df.columns = [c.strip().upper().replace("TASK ", "T") for c in df.columns]
    out = {}
    for pid, row in df.iterrows():
        pid = str(pid).strip().replace("P_", "")
        for k in range(1, 6):
            out[(pid, f"Task {k}")] = (float(row.get(f"A_T{k}", np.nan)), float(row.get(f"B_T{k}", np.nan)))
    return out


def build(root=None, out=None, win=60.0, step=30.0, verbose=True):
    root = root or raw_path("TRIAD_SENSECOBOT_DIR", "SenseCobot")
    nasa = read_nasa(root)
    pids = sorted({re.search(r"P_(\d+)\.csv", f).group(1) for f in glob.glob(os.path.join(root, "EDA_Empatica_Signals", "*.csv"))})
    rows = []
    for pid in pids:
        for ti, task in enumerate(TASKS):
            ecg = _load(_find(root, "ECG_Shimmer3_Signals", "ECG", task, pid), ["ECG LL-RA CAL"])
            gsr = _load(_find(root, "GSR_Shimmer3_Signals", "GSR", task, pid), ["GSR Conductance CAL"])
            e4 = _load(_find(root, "EDA_Empatica_Signals", "EDA_Empatica", task, pid), ["EDA", "HR", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"])
            aff = _load(_find(root, "Emotions_AFFDEX_Signals", "Emotions", task, pid), AFFDEX_COLS)
            _anchor([ecg, gsr, e4, aff])
            avail = [d for d in (ecg, gsr, e4, aff) if d is not None and len(d) > 10 and np.isfinite(d["t"].iloc[0])]
            if not avail:
                continue
            kinds = {d.attrs.get("time_kind") for d in avail}
            if len(kinds) > 1:  # cannot align relative and absolute streams: keep the absolute ones only
                avail = [d for d in avail if d.attrs.get("time_kind") == "absolute"]
                ecg, gsr, e4, aff = [d if (d is not None and d.attrs.get("time_kind") == "absolute") else None for d in (ecg, gsr, e4, aff)]
            t0 = min(d["t"].iloc[0] for d in avail)
            t1 = max(d["t"].iloc[-1] for d in avail)
            dur = t1 - t0
            ds_label = np.nan
            for d in (e4, ecg, gsr, aff):
                if d is not None and "Label" in d.columns and d["Label"].notna().any():
                    ds_label = 1.0 if str(d["Label"].dropna().iloc[0]).strip().upper() == "STRESS" else 0.0
                    break
            # R peaks once per recording
            rr_t = rr = None
            if ecg is not None and len(ecg) > 512 * 10:
                fs = 512.0
                tp = detect_rpeaks(ecg["ECG LL-RA CAL"].to_numpy(), fs) + ecg["t"].iloc[0]
                rr_t, rr = clean_rr(tp)
            gsr_ds = None
            if gsr is not None and len(gsr) > 128 * 10:
                # decimate 128 -> 8 Hz by block mean
                g = gsr["GSR Conductance CAL"].to_numpy(dtype=float)
                n = (len(g) // 16) * 16
                gsr_ds = (g[:n].reshape(-1, 16).mean(1), gsr["t"].to_numpy()[:n:16])
            wins = list(sliding_windows(t0, t1, win, step)) if dur >= win else [(t0, t1)]
            nA, nB = nasa.get((pid, task), (np.nan, np.nan))
            for (ws, we) in wins:
                f = {"dataset": "sensecobot", "subject": f"P{pid}", "session": f"P{pid}_{task.replace(' ', '')}",
                     "condition": task.replace(" ", "").lower(), "design": 0 if ti == 0 else 1, "complexity": ti,
                     "win_start": ws - t0, "q_nasaA": nA, "q_nasaB": nB,
                     "q_nasaAB": (nA + nB) / 2 if np.isfinite(nA) and np.isfinite(nB) else np.nan, "q_dslabel": ds_label}
                if rr is not None:
                    m = (rr_t >= ws) & (rr_t < we)
                    f.update(hrv_features(rr[m], rr_t[m], win, prefix="ecg"))
                if gsr_ds is not None:
                    gx, gt = gsr_ds
                    m = (gt >= ws) & (gt < we)
                    f.update(eda_features(gx[m], 8.0, prefix="gsr"))
                if e4 is not None:
                    m = (e4["t"] >= ws) & (e4["t"] < we)
                    seg = e4[m]
                    f.update(eda_features(seg["EDA"].to_numpy(), 1.0, prefix="e4eda"))
                    f.update(basic_stats(seg["HR"].to_numpy(), "e4hr"))
                    f.update(basic_stats(seg["TEMP"].to_numpy(), "e4temp"))
                    f.update(acc_features(seg[["ACC_X", "ACC_Y", "ACC_Z"]].to_numpy(), "e4acc"))
                if aff is not None:
                    m = (aff["t"] >= ws) & (aff["t"] < we)
                    seg = aff[m]
                    for c in AFFDEX_COLS:
                        v = seg[c].to_numpy(dtype=float)
                        v = v[np.isfinite(v)]
                        f[f"face_{c.lower()}_mean"] = v.mean() if len(v) else np.nan
                        f[f"face_{c.lower()}_std"] = v.std() if len(v) else np.nan
                rows.append(f)
            if verbose:
                print(f"P{pid} {task:9s} dur={dur/60:5.1f} min windows={len(wins)} beats={0 if rr is None else len(rr)} dslabel={ds_label} nasa=({nA},{nB})", flush=True)
    df = pd.DataFrame(rows)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print("saved", out, df.shape)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=raw_path("TRIAD_SENSECOBOT_DIR", "SenseCobot"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "sensecobot_windows.csv"))
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=30.0)
    a = ap.parse_args()
    build(a.root, os.path.abspath(a.out), a.win, a.step)
