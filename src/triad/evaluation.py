"""Data assembly, leave-one-subject-out evaluation, metrics and subject-level bootstrap.

Evaluation is two-dimensional on purpose: every score vector is judged against the design condition (AUC_C), against
the binarised self-report (AUC_R, Spearman rho against the participant-centred score) and on the sub-set where the two
ground truths agree (AUC_agree, the closest thing to a gold standard the data offer).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .baselines import (AgreementFilter, ConditionOnly, DawidSkene, NoiseRobustMLP, Raykar, ReportOnly,
                        ReportRegression, TwoHead)
from .model import TRIAD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")

DATASETS = {
    "mphrc": dict(file="mphrc_windows.csv", questionnaires=["q_stai", "q_nasa", "q_arousal"], primary="q_stai", day=1),
    # all sessions incl. day-2 industrial manual / cobot tasks (design = task, but reports rarely agree)
    "mphrc_all": dict(file="mphrc_windows.csv", questionnaires=["q_stai", "q_nasa", "q_arousal"], primary="q_stai", day=None),
    "wesad": dict(file="wesad_windows.csv", questionnaires=["q_stai6", "q_panas_na", "q_arousal"], primary="q_stai6"),
    "sensecobot": dict(file="sensecobot_windows.csv", questionnaires=["q_nasaAB", "q_nasaA", "q_nasaB"], primary="q_nasaAB"),
}
NON_FEATURE = {"dataset", "subject", "session", "condition", "design", "win_start", "day", "complexity"}


# --------------------------------------------------------------------------- classifiers
class Scaled(BaseEstimator, ClassifierMixin):
    """StandardScaler + estimator that forwards sample_weight (Pipeline needs routing for that)."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.sc_ = StandardScaler().fit(X)
        self.est_ = clone(self.est)
        self.est_.fit(self.sc_.transform(X), y, sample_weight=sample_weight)
        self.classes_ = self.est_.classes_
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(self.sc_.transform(X))

    def predict(self, X):
        return self.est_.predict(self.sc_.transform(X))


def make_clf(family: str, seed: int = 0):
    if family == "lr":
        return Scaled(LogisticRegression(C=0.1, max_iter=5000))
    if family == "lr1":
        return Scaled(LogisticRegression(C=1.0, max_iter=5000))
    if family == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if family == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)
    if family == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             n_jobs=1, random_state=seed, verbosity=0)
    raise ValueError(family)


def make_reg(family: str, seed: int = 0, alpha: float = 10.0):
    """Regression counterpart of make_clf (deployment model of the graded latent-activation variant)."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from .graded import _ScaledRidge
    if family in ("lr", "lr1"):
        return _ScaledRidge(alpha=alpha)
    if family == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if family == "hgb":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)
    raise ValueError(family)


# --------------------------------------------------------------------------- data
@dataclass
class TaskData:
    name: str
    X: np.ndarray
    cond: np.ndarray
    design: np.ndarray
    Q: np.ndarray
    qnames: list
    subj: np.ndarray
    session: np.ndarray
    R: dict = field(default_factory=dict)  # qname -> median-split label (NaN ties)
    Qc: dict = field(default_factory=dict)  # qname -> participant-centred score
    feat_names: list = field(default_factory=list)
    win: dict = field(default_factory=dict)  # window-level arrays (X, session index) for deployment analysis

    def Q_for(self, qnames):
        idx = [self.qnames.index(q) for q in qnames]
        return self.Q[:, idx]


def _clean_features(df, feat_cols, max_nan=0.3):
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    keep = X.columns[X.isna().mean() < max_nan]
    X = X[keep]
    keep = X.columns[X.std(skipna=True) > 0]
    return X[keep]


def load_dataset(name, per_subject_norm=True, questionnaires=None, keep_windows=False):
    cfg = DATASETS[name]
    df = pd.read_csv(os.path.join(PROC, cfg["file"]))
    if cfg.get("day") is not None:
        df = df[df["day"] == cfg["day"]]
    if name == "sensecobot":
        # design manipulation = task complexity: levels 1-2 low (0), 3 ambiguous (-1), 4-5 high (1); the video
        # baseline carries no self-report and is excluded from the task-level analysis (but used for normalisation)
        df = df.copy()
        df["design"] = df["complexity"].map({0: 0, 1: 0, 2: 0, 3: -1, 4: 1, 5: 1})
        df = df[np.isfinite(df["q_nasaA"]) | (df["condition"] == "baseline")]
    qn = questionnaires or cfg["questionnaires"]
    qcols = [c for c in df.columns if c.startswith("q_")]
    feat_cols = [c for c in df.columns if c not in NON_FEATURE and c not in qcols]
    Xw = _clean_features(df, feat_cols)
    feat_names = list(Xw.columns)
    # drop windows with (almost) no valid features (e.g. recordings with corrupted timestamps)
    valid = Xw.notna().mean(1) > 0.2
    df, Xw = df[valid.to_numpy()], Xw[valid]
    Xw = Xw.to_numpy(float)
    subj_w = df["subject"].to_numpy()
    if per_subject_norm:  # label-free, uses only the participant's own windows
        for s in np.unique(subj_w):
            m = subj_w == s
            mu = np.nanmean(Xw[m], 0)
            sd = np.nanstd(Xw[m], 0) + 1e-6
            Xw[m] = (Xw[m] - mu) / sd
    med = np.nanmedian(Xw, 0)
    Xw = np.where(np.isfinite(Xw), Xw, med)
    Xw = np.where(np.isfinite(Xw), Xw, 0.0)
    if name == "sensecobot":
        keep = (df["condition"] != "baseline").to_numpy()
        df, Xw, subj_w = df[keep], Xw[keep], subj_w[keep]
    df = df.reset_index(drop=True)
    sess = df["session"].to_numpy()
    uniq, inv = np.unique(sess, return_inverse=True)
    X = np.zeros((len(uniq), Xw.shape[1]))
    for k in range(len(uniq)):
        X[k] = Xw[inv == k].mean(0)
    first = pd.Series(np.arange(len(df))).groupby(inv).first().to_numpy()
    meta = df.iloc[first]
    cond = meta["condition"].to_numpy()
    if name == "sensecobot":
        cond = np.array([f"level{int(c)}" for c in meta["complexity"]])
    design = meta["design"].to_numpy().astype(int)
    Q = meta[qn].to_numpy(float)
    subj = meta["subject"].to_numpy()
    data = TaskData(name, X, cond, design, Q, list(qn), subj, uniq, feat_names=feat_names)
    for j, q in enumerate(qn):
        v = Q[:, j]
        R = np.full(len(v), np.nan)
        Qc = np.full(len(v), np.nan)
        for s in np.unique(subj):
            m = (subj == s) & np.isfinite(v)
            if m.sum() < 2:
                continue
            medv = np.median(v[m])
            R[m] = np.where(v[m] > medv, 1.0, np.where(v[m] < medv, 0.0, np.nan))
            Qc[m] = v[m] - v[m].mean()
        data.R[q], data.Qc[q] = R, Qc
    if keep_windows:
        data.win = dict(X=Xw, sess_idx=inv, subj=subj_w)
    return data


# --------------------------------------------------------------------------- methods
def make_method(name: str, clf_family: str, seed: int = 0, tau: float = 1.0, **kw):
    """kw are forwarded to TRIAD variants (e.g. pi_prior, lam_alpha, n_folds)."""
    clf = make_clf(clf_family, seed) if clf_family else None
    if name == "cond":
        return ConditionOnly(clf)
    if name == "report":
        return ReportOnly(clf)
    if name == "regress":
        return ReportRegression()
    if name == "twohead":
        return TwoHead(clf)
    if name == "filter":
        return AgreementFilter(clf)
    if name == "ds":
        return DawidSkene(clf)
    if name == "raykar":
        return Raykar(clf)
    if name.startswith("mlp_"):
        return NoiseRobustMLP(loss=name.split("_", 1)[1], seed=seed)
    if name == "triad_cq":
        return TRIAD(clf, tau=0.0, **kw)
    if name == "triad":
        return TRIAD(clf, tau=tau, cross_fit=True, random_state=seed, **kw)
    if name == "triad_nocf":
        return TRIAD(clf, tau=tau, cross_fit=False, random_state=seed, **kw)
    if name == "triad_noalpha":
        return TRIAD(clf, tau=tau, subject_baseline=False, random_state=seed, **kw)
    if name == "triad_noc":
        return TRIAD(clf, tau=tau, use_c=False, random_state=seed, **kw)
    if name == "triad_noq":
        return TRIAD(clf, tau=tau, use_q=False, random_state=seed, **kw)
    if name == "triad_noprior":
        return TRIAD(clf, tau=tau, pi_prior=0.0, random_state=seed, **{k: v for k, v in kw.items() if k != "pi_prior"})
    if name == "triad_student":  # robustness: heavy-tailed (Student-t, nu=4) instead of Gaussian report likelihood
        return TRIAD(clf, tau=tau, q_dist="t", nu=4.0, random_state=seed, **kw)
    if name == "triad_eb":  # empirical-Bayes shrinkage of the participant baselines
        return TRIAD(clf, tau=tau, lam_alpha="auto", random_state=seed, **kw)
    if name.startswith("triad_t"):
        return TRIAD(clf, tau=float(name[7:]), random_state=seed, **kw)
    if name == "triad_noanchor":  # ablation: 16 physiological sweeps without report-anchored early stopping
        return TRIAD(clf, tau=tau, anchor=False, random_state=seed, **kw)
    if name.startswith("triad_b"):  # bounded physiological log-likelihood ratio, no early stopping
        return TRIAD(clf, tau=tau, anchor=False, x_bound=float(name[7:]), random_state=seed, **kw)
    if name.startswith("triad_ab"):  # bounded + anchored
        return TRIAD(clf, tau=tau, anchor=True, x_bound=float(name[8:]), random_state=seed, **kw)
    if name.startswith("triad_h_") or name == "triad_h" or name.startswith("triad_med"):
        # empirical-Bayes efficacy prior (h: method of moments; med: robust median centre, optional strength med50)
        if name.startswith("triad_med"):
            head = name.split("_")[1]
            prior = "median" if head == "med" else f"median:{head[3:]}"
        else:
            prior = "eb"
        opts = dict(tau=tau, random_state=seed, pi_prior=prior)
        for f in name.split("_")[2:]:
            if f == "noanchor":
                opts["anchor"] = False
            elif f == "cq":
                opts["tau"] = 0.0
            elif f == "rcal":
                opts["anchor"], opts["calibrate"] = False, "report"
            elif f.startswith("x"):
                opts["n_iter_x"], opts["anchor"] = int(f[1:]), False
            else:
                raise ValueError(name)
        opts.update({k: v for k, v in kw.items() if k != "pi_prior"})
        return TRIAD(clf, **opts)
    if name == "triad_samex":  # nuisance model of the same family as the deployment classifier
        return TRIAD(clf, tau=tau, x_model=None, random_state=seed, **kw)
    if name == "triad_samexcal":
        return TRIAD(clf, tau=tau, x_model=None, calibrate=True, random_state=seed, **kw)
    if name == "triad_hard":  # deployment classifier trained on the MAP labelling instead of soft posteriors
        return TRIAD(clf, tau=tau, deploy="hard", random_state=seed, **kw)
    if name == "triad_cqcal":  # physiological link calibrated on the design+report posterior, no early stopping
        return TRIAD(clf, tau=tau, anchor=False, calibrate="cq", random_state=seed, **kw)
    if name == "triad_cqcal_anchor":
        return TRIAD(clf, tau=tau, anchor=True, calibrate="cq", random_state=seed, **kw)
    if name == "triad_rcal":  # report-calibrated physiological link, no early stopping
        return TRIAD(clf, tau=tau, anchor=False, calibrate="report", random_state=seed, **kw)
    if name == "triad_rcal_anchor":
        return TRIAD(clf, tau=tau, anchor=True, calibrate="report", random_state=seed, **kw)
    if name.startswith("triad_x"):  # fixed number of physiological sweeps, no anchoring (sensitivity analysis)
        return TRIAD(clf, tau=tau, n_iter_x=int(name[7:]), anchor=False, random_state=seed, **kw)
    if name.startswith("gtriad"):  # graded latent-activation variant
        from .graded import GradedTRIAD
        reg = make_reg(clf_family, seed, alpha=kw.pop("ridge_alpha", 10.0)) if clf_family else None
        opts = dict(tau=tau, random_state=seed)
        flags = name.split("_")[1:]
        for f in flags:
            if f == "cq":
                opts["tau"] = 0.0
            elif f == "noanchor":
                opts["anchor"] = False
            elif f == "diag":
                opts["corr_q"] = False
            elif f == "w":
                opts["weight_x"] = True
            elif f == "c":  # deploy a soft-label classifier of the same family on P(z > 0)
                opts["deploy_clf"] = clf
            elif f.startswith("v"):
                opts["v0"] = float(f[1:])
            elif f == "nocf":
                opts["cross_fit"] = False
            elif f == "noalpha":
                opts["subject_baseline"] = False
            elif f == "noc":
                opts["use_c"] = False
            elif f == "noq":
                opts["use_q"] = False
            elif f == "noprior":
                opts["v_prior"] = 0.0
            elif f.startswith("t"):
                opts["tau"] = float(f[1:])
            elif f.startswith("x"):
                opts["n_iter_x"], opts["anchor"] = int(f[1:]), False
            elif f.startswith("a"):
                opts["v_prior"] = float(f[1:])
            else:
                raise ValueError(name)
        opts.update(kw)
        return GradedTRIAD(reg, **opts)
    raise ValueError(name)


def _fold(method_name, clf_family, seed, tau, data: TaskData, qnames_for_method, q_target, s, kw, level):
    te = data.subj == s
    tr = ~te
    m = make_method(method_name, clf_family, seed, tau, **kw)
    Q = data.Q_for(qnames_for_method)
    win_out = None
    if level == "window":
        ws = data.win["sess_idx"]
        wtr, wte = tr[ws], te[ws]
        pos_tr = np.searchsorted(np.where(tr)[0], ws[wtr])
        pos_te = np.searchsorted(np.where(te)[0], ws[wte])
        m.fit(data.X[tr], data.cond[tr], data.design[tr], Q[tr], data.R[q_target][tr], data.subj[tr],
              X_win=data.win["X"][wtr], win_sess=pos_tr)
        out = m.predict_logit(data.X[te], X_win=data.win["X"][wte], win_sess=pos_te)
        win_out = m.predict_logit_windows(data.win["X"][wte])
    else:
        m.fit(data.X[tr], data.cond[tr], data.design[tr], Q[tr], data.R[q_target][tr], data.subj[tr])
        out = m.predict_logit(data.X[te])
    extra = {}
    if isinstance(m, TRIAD):
        beta, sigma = getattr(m, "beta_", None), getattr(m, "sigma_", None)
        extra = dict(pi=m.pi_, beta=None if beta is None else np.asarray(beta).tolist(),
                     sigma=None if sigma is None else np.asarray(sigma).tolist(), eps_C=m.eps_C_,
                     eps_R=m.report_flip_rate(data.R[q_target][tr]), n_iter=m.n_iter_, n_x=m.n_x_used_)
        if hasattr(m, "v_"):
            extra["v"] = {str(k): float(v) for k, v in m.v_.items()}
            extra["x_cal"] = [float(v) for v in getattr(m, "x_cal_", (np.nan, np.nan, np.nan))]
            if getattr(m, "Sigma_", None) is not None:
                extra["Sigma"] = np.asarray(m.Sigma_).tolist()
    return s, out, win_out, extra


def loso(method_name, data: TaskData, q_target, clf_family="lr", seed=0, tau=1.0, multi_q=False, n_jobs=8,
         level="task", **kw):
    """Out-of-subject logits for every session (and every window if level == 'window').

    multi_q: TRIAD uses all questionnaires; otherwise only q_target.  Returns (scores, extras) or, for window level,
    (scores, extras, window_scores).
    """
    qnames = data.qnames if (multi_q and "triad" in method_name) else [q_target]
    subjects = np.unique(data.subj)
    if level == "window" and not data.win:
        raise ValueError("load_dataset(..., keep_windows=True) required for window-level evaluation")
    res = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_fold)(method_name, clf_family, seed, tau, data, qnames, q_target, s, kw, level) for s in subjects)
    scores = np.zeros(len(data.subj))
    wscores = np.zeros(len(data.win["sess_idx"])) if level == "window" else None
    extras = []
    for s, out, wout, extra in res:
        scores[data.subj == s] = out
        if wout is not None:
            wscores[data.win["subj"] == s] = wout
        extras.append(extra)
    if level == "window":
        return scores, extras, wscores
    return scores, extras


# --------------------------------------------------------------------------- metrics
def _auc(y, s):
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(s)
    if ok.sum() < 3 or len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok].astype(int), s[ok])


def metrics(scores, data: TaskData, q, thr=0.0):
    d = data.design.astype(float)
    d[d < 0] = np.nan
    R = data.R[q]
    out = {}
    out["auc_C"] = _auc(d, scores)
    out["auc_R"] = _auc(R, scores)
    qc = data.Qc[q]
    ok = np.isfinite(qc)
    out["rho"] = spearmanr(scores[ok], qc[ok]).correlation if ok.sum() > 3 else np.nan
    agree = np.isfinite(d) & np.isfinite(R) & (d == R)
    disagree = np.isfinite(d) & np.isfinite(R) & (d != R)
    out["auc_agree"] = _auc(np.where(agree, d, np.nan), scores)
    out["auc_disagree_design"] = _auc(np.where(disagree, d, np.nan), scores)
    out["n_agree"], out["n_disagree"] = int(agree.sum()), int(disagree.sum())
    pred = (scores > thr).astype(int)
    okC = np.isfinite(d)
    okR = np.isfinite(R)
    out["bacc_C"] = balanced_accuracy_score(d[okC].astype(int), pred[okC]) if okC.sum() else np.nan
    out["bacc_R"] = balanced_accuracy_score(R[okR].astype(int), pred[okR]) if okR.sum() else np.nan
    # per-subject mean AUC_R (each participant's own ranking) - robust to response style
    aucs = []
    for s in np.unique(data.subj):
        m = (data.subj == s) & np.isfinite(R)
        if m.sum() >= 3 and len(np.unique(R[m])) == 2:
            aucs.append(roc_auc_score(R[m].astype(int), scores[m]))
    out["auc_R_within"] = float(np.mean(aucs)) if aucs else np.nan
    return out


def metrics_windows(wscores, data: TaskData, q):
    """Window-level (real-time) AUCs against the session's design label and binarised self-report."""
    ws = data.win["sess_idx"]
    d = data.design.astype(float)
    d[d < 0] = np.nan
    R = data.R[q]
    agree = np.isfinite(d) & np.isfinite(R) & (d == R)
    return {"win_auc_C": _auc(d[ws], wscores), "win_auc_R": _auc(R[ws], wscores),
            "win_auc_agree": _auc(np.where(agree, d, np.nan)[ws], wscores)}


def _metric_labels(data: TaskData, q, metric):
    """Binary label vector (NaN = excluded) that defines an AUC-type metric."""
    d = data.design.astype(float)
    d[d < 0] = np.nan
    R = data.R[q]
    if metric == "auc_C":
        return d
    if metric == "auc_R":
        return R
    if metric == "auc_agree":
        agree = np.isfinite(d) & np.isfinite(R) & (d == R)
        return np.where(agree, d, np.nan)
    raise ValueError(metric)


def bootstrap_diff(scores_a, scores_b, data: TaskData, q, metric="auc_R", B=2000, seed=0):
    """Subject-level paired bootstrap of metric(a) - metric(b) for AUC-type metrics.
    Returns (mean diff, lo, hi, p two-sided)."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(data.subj)
    idx_by_s = {s: np.where(data.subj == s)[0] for s in subjects}
    y = _metric_labels(data, q, metric)
    diffs = []
    for _ in range(B):
        pick = rng.choice(subjects, len(subjects), replace=True)
        idx = np.concatenate([idx_by_s[s] for s in pick])
        ma, mb = _auc(y[idx], scores_a[idx]), _auc(y[idx], scores_b[idx])
        if np.isfinite(ma) and np.isfinite(mb):
            diffs.append(ma - mb)
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan, np.nan
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(min(1.0, p))
