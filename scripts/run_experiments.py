"""Run the LOSO benchmark grid and cache per-run scores + metrics under results/runs/.

Example:
  python scripts/run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --level window
  python scripts/run_experiments.py --datasets mphrc --families hgb --methods cond report filter twohead triad --seeds 0 1 2
Each run is stored as results/runs/<dataset>/<q>/<level>/<family>/<method>_s<seed>.npz (scores, window scores) +
.json (metrics, TRIAD by-products).  Existing runs are skipped unless --force.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from triad.evaluation import DATASETS, load_dataset, loso, metrics, metrics_windows

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "runs")

BASELINES = ["cond", "report", "regress", "filter", "twohead", "ds", "raykar"]
MLPS = ["mlp_ce", "mlp_gce", "mlp_sce", "mlp_forward", "mlp_coteach", "mlp_twohead"]
TRIADS = ["triad", "triad_cq", "triad_nocf", "triad_noalpha", "triad_noprior", "triad_noc", "triad_noq", "triad_mq"]
DETERMINISTIC = {"lr", "lr1"}


def method_kwargs(method):
    """Translate a method tag into (loso method name, kwargs)."""
    if method == "triad_mq":
        return "triad", dict(multi_q=True)
    if method.startswith("triad_a"):
        return "triad", dict(pi_prior=float(method[7:]))
    return method, {}


def run_one(ds, q, level, fam, method, seed, n_jobs, force=False):
    d_dir = os.path.join(OUT, ds, q, level, fam)
    os.makedirs(d_dir, exist_ok=True)
    stem = os.path.join(d_dir, f"{method}_s{seed}")
    if os.path.exists(stem + ".json") and not force:
        return json.load(open(stem + ".json"))
    data = DATA[ds]
    name, kw = method_kwargs(method)
    fam_used = None if method.startswith("mlp_") or method == "regress" else fam
    t0 = time.time()
    res = loso(name, data, q, clf_family=fam_used, seed=seed, n_jobs=n_jobs, level=level, **kw)
    scores, extras = res[0], res[1]
    m = metrics(scores, data, q)
    out = dict(dataset=ds, q=q, level=level, family=fam, method=method, seed=seed, seconds=time.time() - t0, **m)
    payload = dict(scores=scores, session=data.session, subject=data.subj)
    if level == "window":
        out.update(metrics_windows(res[2], data, q))
        payload["wscores"] = res[2]
    if extras and extras[0]:
        out["triad"] = dict(
            pi={str(k): float(np.mean([e["pi"][k] for e in extras])) for k in extras[0]["pi"]},
            beta=float(np.mean([e["beta"][0] for e in extras if e["beta"]])) if extras[0]["beta"] else None,
            sigma=float(np.mean([e["sigma"][0] for e in extras if e["sigma"]])) if extras[0]["sigma"] else None,
            eps_C=float(np.nanmean([e["eps_C"] for e in extras])), eps_R=float(np.nanmean([e["eps_R"] for e in extras])),
            n_iter=float(np.mean([e["n_iter"] for e in extras])),
            n_x=float(np.mean([e.get("n_x", np.nan) for e in extras])))
    np.savez_compressed(stem + ".npz", **payload)
    json.dump(out, open(stem + ".json", "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mphrc", "wesad", "sensecobot"])
    ap.add_argument("--families", nargs="+", default=["lr"])
    ap.add_argument("--methods", nargs="+", default=BASELINES + TRIADS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--level", default="window")
    ap.add_argument("--qs", default="primary", help="primary | all | explicit questionnaire name")
    ap.add_argument("--n_jobs", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    DATA = {ds: load_dataset(ds, keep_windows=True) for ds in a.datasets}
    for ds in a.datasets:
        qs = [DATASETS[ds]["primary"]] if a.qs == "primary" else (DATASETS[ds]["questionnaires"] if a.qs == "all" else [a.qs])
        for q in qs:
            for fam in a.families:
                for method in a.methods:
                    seeds = [0] if (fam in DETERMINISTIC and not method.startswith("mlp_")) else a.seeds
                    for seed in seeds:
                        r = run_one(ds, q, a.level, fam, method, seed, a.n_jobs, a.force)
                        keys = ["auc_C", "auc_R", "rho", "auc_agree", "auc_R_within"] + (["win_auc_C", "win_auc_R", "win_auc_agree"] if a.level == "window" else [])
                        print(f"{ds:10s} {q:10s} {fam:4s} {method:15s} s{seed} {r.get('seconds', 0):6.1f}s " + " ".join(f"{k}={r[k]:.3f}" for k in keys if k in r and r[k] is not None and np.isfinite(r[k])), flush=True)
