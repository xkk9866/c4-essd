"""Dev: held-out-view validation.  Scores of models trained with questionnaire q_train (cached npz) are evaluated
against the within-participant split of a questionnaire that was never used in training."""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from triad.evaluation import DATASETS, load_dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "results", "runs")
PAIRS = {"mphrc": [("q_stai", "q_nasa"), ("q_stai", "q_arousal")],
         "wesad": [("q_stai6", "q_panas_na"), ("q_stai6", "q_arousal")],
         "sensecobot": [("q_nasaA", "q_nasaB"), ("q_nasaB", "q_nasaA")],
         "mphrc_all": [("q_stai", "q_nasa"), ("q_stai", "q_arousal")]}
METHODS = ["cond", "report", "regress", "twohead", "filter", "ds", "raykar", "triad_cq", "triad", "triad_noanchor", "triad_mq"]
fam = sys.argv[1] if len(sys.argv) > 1 else "lr"


def auc(y, s):
    ok = np.isfinite(y) & np.isfinite(s)
    if ok.sum() < 3 or len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok].astype(int), s[ok])


for ds, pairs in PAIRS.items():
    d = load_dataset(ds)
    for q_train, q_eval in pairs:
        rows = []
        for m in METHODS:
            files = sorted(glob.glob(os.path.join(RUNS, ds, q_train, "task", fam, f"{m}_s*.npz")))
            if not files:
                continue
            vals = []
            for f in files:
                z = np.load(f, allow_pickle=True)
                assert (z["session"] == d.session).all()
                s = z["scores"]
                R = d.R[q_eval]
                Qc = d.Qc[q_eval]
                ok = np.isfinite(Qc) & np.isfinite(s)
                rho = spearmanr(Qc[ok], s[ok]).correlation if ok.sum() > 3 else np.nan
                # within-participant AUC against the held-out questionnaire
                w = []
                for sub in np.unique(d.subj):
                    mm = (d.subj == sub) & np.isfinite(R) & np.isfinite(s)
                    if mm.sum() >= 3 and len(np.unique(R[mm])) == 2:
                        w.append(roc_auc_score(R[mm].astype(int), s[mm]))
                vals.append((auc(R, s), rho, np.mean(w) if w else np.nan))
            v = np.nanmean(np.array(vals, float), 0)
            rows.append(dict(method=m, n=len(files), auc_heldout=v[0], rho_heldout=v[1], auc_within=v[2]))
        print(f"\n== {ds} {fam}: trained with {q_train}, evaluated on held-out {q_eval}")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
