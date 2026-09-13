"""Quick smoke test of the harness on one dataset."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from triad.evaluation import load_dataset, loso, metrics

ds = sys.argv[1] if len(sys.argv) > 1 else "mphrc"
clf = sys.argv[2] if len(sys.argv) > 2 else "lr"
d = load_dataset(ds)
q = {"mphrc": "q_stai", "wesad": "q_stai6", "sensecobot": "q_nasaA"}[ds]
print(ds, "sessions", len(d.subj), "subjects", len(np.unique(d.subj)), "feats", d.X.shape[1], "conds", dict(zip(*np.unique(d.cond, return_counts=True))))
R, D = d.R[q], d.design.astype(float)
ok = np.isfinite(R) & (D >= 0)
print(f"agreement C vs R[{q}]: {(R[ok] == D[ok]).mean():.3f} on n={ok.sum()}")
for m in ["cond", "report", "filter", "ds", "raykar", "triad_cq", "triad"]:
    t0 = time.time()
    s, ex = loso(m, d, q, clf_family=clf, n_jobs=16)
    r = metrics(s, d, q)
    msg = " ".join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float))
    if ex and ex[0]:
        e = ex[0]
        msg += f" | pi={ {k: round(v, 2) for k, v in e['pi'].items()} } eps_C={e['eps_C']:.2f} eps_R={e['eps_R']:.2f} it={e['n_iter']}"
    print(f"{m:10s} {time.time()-t0:5.1f}s {msg}")
