"""Dev diagnostic: quick LOSO comparison for a few methods / TRIAD configurations.
usage: python scripts/dev_diag.py <dataset> <family> [n_jobs] [method[,method...]] [key=val ...]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from triad.evaluation import load_dataset, loso, metrics, DATASETS

ds = sys.argv[1] if len(sys.argv) > 1 else "mphrc"
fam = sys.argv[2] if len(sys.argv) > 2 else "lr"
nj = int(sys.argv[3]) if len(sys.argv) > 3 else 8
meths = sys.argv[4].split(",") if len(sys.argv) > 4 else ["cond", "twohead", "triad", "gtriad"]
extra_kw = {}
for a in sys.argv[5:]:
    k, v = a.split("=")
    try:
        v = float(v)
    except ValueError:
        pass
    extra_kw[k] = v
q = DATASETS[ds]["primary"]
d = load_dataset(ds)
keys = ("auc_C", "auc_R", "rho", "auc_agree", "auc_R_within")
heldout = [qq for qq in d.qnames if qq != q]


def _auc(y, s):
    ok = np.isfinite(y) & np.isfinite(s)
    if ok.sum() < 3 or len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok].astype(int), s[ok])


def run(name, **kw):
    t0 = time.time()
    res = loso(name, d, q, fam, n_jobs=nj, **kw)
    s = res[0]
    r = metrics(s, d, q)
    msg = " ".join(f"{k}={r[k]:.3f}" for k in keys)
    ho = " ".join(f"ho[{qq[2:]}]={_auc(d.R[qq], s):.3f}" for qq in heldout)
    ex = res[1]
    if ex and ex[0]:
        msg += f" | eps_C={np.mean([e['eps_C'] for e in ex]):.2f} eps_R={np.mean([e['eps_R'] for e in ex]):.2f} nx={np.mean([e.get('n_x', 0) for e in ex]):.1f}"
    tag = name + " " + " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"{ds} {fam} {tag:36s} {time.time()-t0:5.1f}s {msg} | {ho}", flush=True)


for m in meths:
    run(m, **extra_kw)
