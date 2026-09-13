"""WESAD (Schmidt et al., ICMI 2018): 15 subjects; Base, TSST (stress), Fun (amusement), Medi 1, Medi 2.

Physiology: chest RespiBAN (ECG, EDA, EMG, RESP, TEMP, ACC). Window features are 37 descriptors
(X [n,37]) with protocol labels in {1 base, 2 TSST, 3 fun, 4 medi}. Each window is aligned to its
protocol session using SX_quest.csv (ORDER / START / END, minutes).

Self-reports (one per session, from SX_quest.csv):  STAI-6 (items 1,4,6 reversed; 6-24), PANAS negative affect
(10 standard NA items), PANAS 'stressed' item, PANAS positive affect, SAM valence and arousal (1-9).

Design: TSST = 1, Base / Medi 1 / Medi 2 / Fun = 0 (the standard WESAD stress vs. non-stress split); the condition
column keeps the five-level protocol so TRIAD can estimate one manipulation-efficacy value per session type.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from triad.data import raw_path

NPZ = os.environ.get("TRIAD_WESAD_NPZ") or os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "processed", "wesad_chest.npz"))
RAW = raw_path("TRIAD_WESAD_DIR", "WESAD")
SESSIONS = ["Base", "TSST", "Medi 1", "Fun", "Medi 2"]
NA_IDX = [2, 5, 7, 8, 9, 12, 14, 16, 19, 20]  # 1-indexed PANAS negative-affect items
PA_IDX = [1, 3, 4, 6, 10, 11, 13, 15, 17, 18]
STRESSED_IDX = 21
Y2NAME = {1: "base", 2: "tsst", 3: "fun", 4: "medi"}


def parse_quest(path):
    rows = [l.rstrip("\n").split(";") for l in open(path, encoding="utf-8", errors="ignore")]
    get = lambda key: [r for r in rows if r and r[0].strip().lstrip("#").strip().upper() == key]
    order = [x.strip() for x in get("ORDER")[0][1:] if x.strip()]
    start = [float(x) for x in get("START")[0][1:] if x.strip()]
    end = [float(x) for x in get("END")[0][1:] if x.strip()]
    panas = [[float(x) for x in r[1:] if x.strip()] for r in get("PANAS")]
    stai = [[float(x) for x in r[1:] if x.strip()] for r in get("STAI")]
    dim = [[float(x) for x in r[1:] if x.strip()] for r in get("DIM")]
    sess = {}
    qi = 0
    for k, name in enumerate(order):
        if name not in SESSIONS:
            continue
        p, s, d = panas[qi], stai[qi], dim[qi]
        qi += 1
        stai6 = sum(5 - v if i in (1, 4, 6) else v for i, v in enumerate(s, start=1))
        sess[name] = dict(start=start[k] * 60, end=end[k] * 60, q_stai6=stai6,
                          q_panas_na=sum(p[i - 1] for i in NA_IDX), q_panas_pa=sum(p[i - 1] for i in PA_IDX),
                          q_stressed=p[STRESSED_IDX - 1], q_valence=d[0], q_arousal=d[1])
    return sess


def build(out=None, verbose=True):
    z = np.load(NPZ, allow_pickle=True)
    X, y, t, subj, names = z["X"], z["y"], z["t"], z["subject"], list(z["names"])
    rows = []
    for s in sorted(set(subj), key=lambda v: int(v[1:])):
        sess = parse_quest(os.path.join(RAW, s, f"{s}_quest.csv"))
        m = subj == s
        for xi, yi, ti in zip(X[m], y[m], t[m]):
            hit = [n for n, v in sess.items() if v["start"] - 1 <= ti < v["end"] + 1]
            if not hit:
                continue
            name = hit[0]
            expect = {"Base": 1, "TSST": 2, "Fun": 3, "Medi 1": 4, "Medi 2": 4}[name]
            if expect != yi:
                continue
            v = sess[name]
            r = {"dataset": "wesad", "subject": s, "session": f"{s}_{name.replace(' ', '')}",
                 "condition": name.replace(" ", "").lower(), "design": 1 if name == "TSST" else 0, "win_start": ti - v["start"]}
            r.update({k: v[k] for k in v if k.startswith("q_")})
            r.update(dict(zip(names, xi.astype(float))))
            rows.append(r)
        if verbose:
            print(s, {k: (round(v["q_stai6"], 1), v["q_arousal"]) for k, v in sess.items()})
    df = pd.DataFrame(rows)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print("saved", out, df.shape)
    return df


if __name__ == "__main__":
    build(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "wesad_windows.csv")))
