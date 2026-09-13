"""MultiPhysio-HRC (Bussolan et al., 2025): 52 participants, day-1 cognitive stressors + VR, day-2 industrial tasks.

We use the official 60-s window bio features (HRV / EDA / EMG / RESP, 264 columns) and labels.csv (STAI-Y1, NASA-TLX
and its six sub-scales, SAM valence / arousal / dominance).  Two conventions in the released files are reconciled:
class names ("vr-job-simulator" vs "vr-job-sim", casing) and repetition numbering (day-1 tasks are 0 in bio and 1
in labels; rest_0 = day-1 rest = label rep 1, rest-1 = day-2 rest = label rep 2).  Duplicate label rows are averaged.

Design: rest / meditation = 0; stroopeasy / stroophard / n-back / mat / hanoi / vr-plank / vr-job-sim = 1 (day 1);
manual-task / cobot-task = 1 with day = 2 (industrial workload; excluded from the main protocol by default).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from triad.data import raw_path

ROOT = raw_path("TRIAD_MPHRC_DIR", "MultiPhysio-HRC", "features")
CALM = {"rest", "meditation"}
STRESS_D1 = {"stroopeasy", "stroophard", "n-back", "mat", "hanoi", "vr-plank", "vr-job-sim"}
DAY2 = {"manual-task", "cobot-task"}
QCOLS = {"STAI": "q_stai", "NASA": "q_nasa", "Mental_Demand": "q_mental", "Physical_Demand": "q_physical",
         "Temporal_Demand": "q_temporal", "Performance": "q_performance", "Effort": "q_effort",
         "Frustration": "q_frustration", "Valence": "q_valence", "Arousal": "q_arousal", "Dominance": "q_dominance"}


def norm_class(s):
    s = str(s).strip().lower().replace(" ", "-")
    return {"vr-job-simulator": "vr-job-sim"}.get(s, s)


def map_rep(c, r):
    if c in DAY2:
        return r
    if c == "rest":
        return {0: 1, 1: 2}.get(r, -1)
    return 1 if r == 0 else -1


def build(out=None, verbose=True):
    bio = pd.read_csv(os.path.join(ROOT, "bio_features_60s.csv"))
    lab = pd.read_csv(os.path.join(ROOT, "labels.csv"))
    bio["Class"] = bio["Class"].map(norm_class)
    lab["Class"] = lab["Class"].map(norm_class)
    bio["Repetition"] = [map_rep(c, r) for c, r in zip(bio["Class"], bio["Repetition"])]
    bio = bio[bio["Repetition"] > 0].copy()
    lab_num = [c for c in lab.columns if c not in ("ID", "Class", "Repetition")]
    lab = lab.groupby(["ID", "Class", "Repetition"])[lab_num].mean().reset_index()
    feat_cols = [c for c in bio.columns if c not in ("ID", "Class", "Repetition", "Window")]
    df = bio.merge(lab[["ID", "Class", "Repetition"] + list(QCOLS)], on=["ID", "Class", "Repetition"], how="inner")
    meta = pd.DataFrame({
        "dataset": "mphrc", "subject": "S" + df["ID"].astype(str), "session": "S" + df["ID"].astype(str) + "_" + df["Class"] + "_r" + df["Repetition"].astype(str),
        "condition": df["Class"], "design": np.where(df["Class"].isin(CALM), 0, 1),
        "day": np.where(df["Class"].isin(DAY2) | ((df["Class"] == "rest") & (df["Repetition"] == 2)), 2, 1),
        "win_start": df["Window"].astype(float) * 60.0})
    q = df[list(QCOLS)].rename(columns=QCOLS)
    outdf = pd.concat([meta.reset_index(drop=True), q.reset_index(drop=True), df[feat_cols].reset_index(drop=True)], axis=1)
    if verbose:
        print("windows", outdf.shape, "sessions", outdf["session"].nunique(), "subjects", outdf["subject"].nunique())
        print(outdf.groupby(["day", "condition"]).size())
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        outdf.to_csv(out, index=False)
        print("saved", out)
    return outdf


if __name__ == "__main__":
    build(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "mphrc_windows.csv")))
