"""Semi-synthetic validation with a known latent state.

Real physiology X (and participants) are kept; a teacher classifier fitted on the real design labels provides
p(S=1 | x), from which the latent state S is sampled.  Design labels C and self-reports Q are then generated from S with
controlled noise:  C = S flipped with prob eps_C;  Q = alpha_i + beta * S + N(0, 1) with per-participant alpha_i and
beta chosen so that the participant-median split of Q disagrees with S at rate ~ eps_R.  All methods are evaluated
LOSO against the true S, and TRIAD's recovered eps_C / eps_R are compared with the truth.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from triad.evaluation import TaskData, load_dataset, loso, make_clf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def beta_for_eps(eps_R, sigma=1.0):
    """Report sensitivity beta such that a within-person median split flips w.p. ~eps_R (balanced classes).
    For balanced S and threshold at the midpoint, P(flip) = Phi(-beta / (2 sigma))."""
    return -2 * sigma * norm.ppf(eps_R)


def _shift_to_base_rate(p, target, tol=1e-4):
    """Shift the teacher's log-odds by a constant so that the mean latent-state probability equals `target`.
    The teacher's ranking of windows (and hence the realism of X | S) is preserved."""
    lo, hi = -20.0, 20.0
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    for _ in range(100):
        b = 0.5 * (lo + hi)
        m = (1 / (1 + np.exp(-(z - b)))).mean()
        if abs(m - target) < tol:
            break
        lo, hi = (b, hi) if m > target else (lo, b)
    return 1 / (1 + np.exp(-(z - b)))


def make_synthetic(d: TaskData, eps_C, eps_R, rng, mode="uniform", pi_fail=0.3):
    """mode='uniform': C = S flipped independently w.p. eps_C (two synthetic conditions).
    mode='structured': six equiprobable synthetic conditions with condition-level efficacies pi_k = (0.9, 0.85,
    pi_fail | 0.1, 0.12, 0.15); the third 'stressor' is a failed manipulation.  The teacher's probabilities are shifted
    so that the base rate of S equals the mean efficacy, and the condition is drawn given S with P(k | S=1) = pi_k /
    sum(pi) and P(k | S=0) = (1-pi_k) / sum(1-pi).  Under this construction P(k) = 1/6 and P(S=1 | k) = pi_k exactly,
    i.e. the realised efficacy of every synthetic condition equals its nominal value; C is the design intent of the
    drawn condition.  eps_C is ignored in this mode."""
    teacher = make_clf("lr").fit(d.X, np.where(d.design == 1, 1, 0))
    p = teacher.predict_proba(d.X)[:, 1]
    if mode == "structured":
        pis = np.array([0.9, 0.85, pi_fail, 0.1, 0.12, 0.15])
        intent = np.array([1, 1, 1, 0, 0, 0])
        names = np.array(["stressA", "stressB", "stressFail", "calmA", "calmB", "calmC"])
        p = _shift_to_base_rate(p, pis.mean())
        S = (rng.random(len(p)) < p).astype(int)
        p1, p0 = pis / pis.sum(), (1 - pis) / (1 - pis).sum()
        k = np.where(S == 1, rng.choice(6, size=len(S), p=p1), rng.choice(6, size=len(S), p=p0))
        C = intent[k]
        cond_names = names[k]
    else:
        S = (rng.random(len(p)) < p).astype(int)
        C = np.where(rng.random(len(S)) < eps_C, 1 - S, S)
        cond_names = np.where(C == 1, "c_stress", "c_calm")
    subjects = np.unique(d.subj)
    alpha = dict(zip(subjects, rng.normal(0, 3.0, len(subjects))))
    beta = beta_for_eps(eps_R)
    q = np.array([alpha[s] for s in d.subj]) + beta * S + rng.normal(0, 1.0, len(S))
    dd = TaskData("synthetic", d.X, cond_names, C, q[:, None], ["q_syn"], d.subj, d.session,
                  feat_names=d.feat_names, win=d.win)
    R = np.full(len(q), np.nan)
    Qc = np.full(len(q), np.nan)
    for s in subjects:
        m = d.subj == s
        med = np.median(q[m])
        R[m] = np.where(q[m] > med, 1.0, np.where(q[m] < med, 0.0, np.nan))
        Qc[m] = q[m] - q[m].mean()
    dd.R["q_syn"], dd.Qc["q_syn"] = R, Qc
    ok = np.isfinite(R)
    pi_true = {str(c): float(S[cond_names == c].mean()) for c in np.unique(cond_names)}  # realised P(S=1 | condition)
    return dd, S, dict(eps_C_emp=float((C != S).mean()), eps_R_emp=float((R[ok] != S[ok]).mean()), beta=beta,
                       p_S=float(S.mean()), pi_true=pi_true)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mphrc")
    ap.add_argument("--eps_C", nargs="+", type=float, default=[0.05, 0.15, 0.30])
    ap.add_argument("--eps_R", nargs="+", type=float, default=[0.20, 0.35])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--family", default="lr")
    ap.add_argument("--level", default="task")
    ap.add_argument("--methods", nargs="+", default=["cond", "report", "filter", "twohead", "ds", "raykar", "triad", "triad_cq"])
    ap.add_argument("--mode", default="uniform", choices=["uniform", "structured"])
    ap.add_argument("--pi_fail", nargs="+", type=float, default=[0.5, 0.3])
    ap.add_argument("--n_jobs", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="recompute the listed methods (existing rows are dropped)")
    a = ap.parse_args()
    d = load_dataset(a.dataset, keep_windows=True)
    out = os.path.join(ROOT, "results", f"semi_synthetic_{a.mode}_{a.dataset}_{a.family}_{a.level}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = json.load(open(out)) if os.path.exists(out) else []
    if a.force:
        rows = [r for r in rows if r["method"] not in set(a.methods)]
    done = {(r["eps_C"], r["eps_R"], r["rep"], r["method"], r.get("pi_fail")) for r in rows}
    grid = [(eC, None) for eC in a.eps_C] if a.mode == "uniform" else [(np.nan, pf) for pf in a.pi_fail]
    for eC, pf in grid:
        for eR in a.eps_R:
            for rep in range(a.reps):
                seed = 1000 * rep + int(10 * eR) + (int(100 * eC) if pf is None else 7 + int(100 * pf))
                rng = np.random.default_rng(seed)
                dd, S, info = make_synthetic(d, eC, eR, rng, mode=a.mode, pi_fail=pf if pf is not None else 0.3)
                for m in a.methods:
                    key = (eC if pf is None else None, eR, rep, m, pf)
                    if key in done:
                        continue
                    res = loso(m, dd, "q_syn", clf_family=a.family, seed=rep, n_jobs=a.n_jobs, level=a.level)
                    auc = roc_auc_score(S, res[0])
                    row = dict(eps_C=eC if pf is None else None, pi_fail=pf, eps_R=eR, rep=rep, method=m, auc_S=auc, **info)
                    if res[1] and res[1][0]:
                        row["eps_C_hat"] = float(np.mean([e["eps_C"] for e in res[1]]))
                        row["eps_R_hat"] = float(np.mean([e["eps_R"] for e in res[1]]))
                        row["n_x"] = float(np.mean([e.get("n_x", np.nan) for e in res[1]]))
                        row["pi_hat"] = {k: float(np.mean([e["pi"][k] for e in res[1]])) for k in res[1][0]["pi"]}
                    rows.append(row)
                    json.dump(rows, open(out, "w"), indent=1)  # checkpoint after every run
                    tag = f"eps_C={eC:.2f}" if pf is None else f"pi_fail={pf:.2f} (eps_C_emp={info['eps_C_emp']:.2f})"
                    print(f"{tag} eps_R={eR:.2f} (emp {info['eps_R_emp']:.2f}) rep={rep} {m:10s} AUC_S={auc:.3f}"
                          + (f" eps_hat=({row['eps_C_hat']:.2f},{row['eps_R_hat']:.2f})" if "eps_C_hat" in row else ""), flush=True)
    print("saved", out)
