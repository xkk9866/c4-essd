"""Baselines for learning a stress/activation detector from design conditions and/or self-reports.

All baselines expose  fit(X, cond, design, Q, R, subj)  and  predict_logit(X)  so that the evaluation harness can
treat them like TRIAD.  R is the per-participant median-split self-report (NaN for ties); Q the raw scores.

B1  ConditionOnly       classifier on design labels
B2  ReportOnly          classifier on binarised self-report
B3  ReportRegression    regressor on participant-centred self-report score
B4  TwoHead             average of B1 and B2 log-odds (shared features, two supervised heads)
B5  NoiseRobust         MLP on R with GCE / SCE / forward-correction / co-teaching losses (torch)
B6  AgreementFilter     classifier trained only where design and self-report agree
B7  DawidSkene          two-annotator Dawid-Skene EM (C and R as annotators) -> soft-label classifier
B8  Raykar              learning-from-crowds EM (Raykar et al. 2010) with classifier prior and two annotators
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .model import _predict_logit, _safe_logit, expand, fit_soft, pool


class _Base:
    """Common plumbing: hard-label fit on sessions or (if X_win/win_sess given) on windows with broadcast labels;
    prediction pools window log-odds back to sessions."""

    def __init__(self, clf):
        self.clf = clf

    def _fit_hard(self, X, y, mask=None, X_win=None, win_sess=None):
        m = np.ones(len(y), bool) if mask is None else mask
        m &= np.isfinite(y)
        if X_win is not None:
            X, y, m = X_win, expand(y, win_sess), expand(m, win_sess)
        if len(np.unique(y[m])) < 2:  # degenerate training set: constant score
            self.clf_ = None
            return self
        self.clf_ = clone(self.clf).fit(X[m], y[m].astype(int))
        return self

    def predict_logit_windows(self, X_win):
        if self.clf_ is None:
            return np.zeros(len(X_win))
        return _predict_logit(self.clf_, X_win)

    def predict_logit(self, X, X_win=None, win_sess=None):
        if X_win is not None:
            return pool(self.predict_logit_windows(X_win), np.asarray(win_sess), len(X))
        if self.clf_ is None:
            return np.zeros(len(X))
        return _predict_logit(self.clf_, X)


class ConditionOnly(_Base):
    def fit(self, X, cond, design, Q, R, subj, X_win=None, win_sess=None):
        d = np.asarray(design, float)
        d[d < 0] = np.nan
        return self._fit_hard(X, d, None, X_win, win_sess)


class ReportOnly(_Base):
    def fit(self, X, cond, design, Q, R, subj, X_win=None, win_sess=None):
        return self._fit_hard(X, np.asarray(R, float), None, X_win, win_sess)


class AgreementFilter(_Base):
    def fit(self, X, cond, design, Q, R, subj, X_win=None, win_sess=None):
        d = np.asarray(design, float)
        r = np.asarray(R, float)
        m = np.isfinite(r) & (d >= 0) & (d == r)
        return self._fit_hard(X, d, m, X_win, win_sess)


class ReportRegression:
    """Regress the participant-centred self-report score (dataset-paper style); the prediction is the score."""

    def __init__(self, reg=None):
        self.reg = reg or make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    def fit(self, X, cond, design, Q, R, subj, X_win=None, win_sess=None):
        q = np.asarray(Q, float)
        q = q[:, 0] if q.ndim == 2 else q
        subj = np.asarray(subj)
        qc = q.copy()
        for s in np.unique(subj):
            m = subj == s
            qc[m] = q[m] - np.nanmean(q[m])
        if X_win is not None:
            X, qc = X_win, expand(qc, win_sess)
        ok = np.isfinite(qc)
        self.reg_ = clone(self.reg).fit(X[ok], qc[ok])
        return self

    def predict_logit_windows(self, X_win):
        return self.reg_.predict(X_win)

    def predict_logit(self, X, X_win=None, win_sess=None):
        if X_win is not None:
            return pool(self.reg_.predict(X_win), np.asarray(win_sess), len(X))
        return self.reg_.predict(X)


class TwoHead:
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, cond, design, Q, R, subj, X_win=None, win_sess=None):
        self.a_ = ConditionOnly(self.clf).fit(X, cond, design, Q, R, subj, X_win, win_sess)
        self.b_ = ReportOnly(self.clf).fit(X, cond, design, Q, R, subj, X_win, win_sess)
        return self

    def predict_logit_windows(self, X_win):
        return 0.5 * (self.a_.predict_logit_windows(X_win) + self.b_.predict_logit_windows(X_win))

    def predict_logit(self, X, X_win=None, win_sess=None):
        return 0.5 * (self.a_.predict_logit(X, X_win, win_sess) + self.b_.predict_logit(X, X_win, win_sess))


class _SoftBase:
    """Soft-label classifier plumbing shared by the crowd-style EM baselines."""

    def _fit_final(self, X, mu, X_win=None, win_sess=None):
        if X_win is not None:
            self.clf_ = fit_soft(self.clf, X_win, expand(mu, win_sess))
        else:
            self.clf_ = fit_soft(self.clf, X, mu)

    def predict_logit_windows(self, X_win):
        return _predict_logit(self.clf_, X_win)

    def predict_logit(self, X, X_win=None, win_sess=None):
        if X_win is not None:
            return pool(_predict_logit(self.clf_, X_win), np.asarray(win_sess), len(X))
        return _predict_logit(self.clf_, X)


# --------------------------------------------------------------------------- crowd-style EM baselines
class DawidSkene(_SoftBase):
    """Two annotators (design, report) with class-conditional confusion; posterior -> soft-label classifier."""

    def __init__(self, clf, n_iter=50):
        self.clf, self.n_iter = clf, n_iter

    def fit(self, X, cond, design, Q, R, subj, X_win=None, win_sess=None):
        d = np.asarray(design, float)
        d[d < 0] = np.nan
        r = np.asarray(R, float)
        L = np.stack([d, r], 1)  # n x 2, NaN missing
        mu = np.nanmean(L, 1)
        mu = np.where(np.isfinite(mu), mu, 0.5)
        for _ in range(self.n_iter):
            p1 = np.clip(mu.mean(), 1e-3, 1 - 1e-3)
            sens = np.zeros(2)
            spec = np.zeros(2)
            for j in range(2):
                ok = np.isfinite(L[:, j])
                sens[j] = np.clip((mu[ok] * L[ok, j]).sum() / max(mu[ok].sum(), 1e-6), 1e-3, 1 - 1e-3)
                spec[j] = np.clip(((1 - mu[ok]) * (1 - L[ok, j])).sum() / max((1 - mu[ok]).sum(), 1e-6), 1e-3, 1 - 1e-3)
            lg = np.full(len(mu), _safe_logit(p1))
            for j in range(2):
                ok = np.isfinite(L[:, j])
                lg[ok] += np.where(L[ok, j] == 1, np.log(sens[j] / (1 - spec[j])), np.log((1 - sens[j]) / spec[j]))
            new = expit(lg)
            if np.abs(new - mu).mean() < 1e-5:
                mu = new
                break
            mu = new
        self.mu_, self.sens_, self.spec_ = mu, sens, spec
        self._fit_final(X, mu, X_win, win_sess)
        return self


class Raykar(_SoftBase):
    """Learning from crowds (Raykar et al., JMLR 2010): classifier prior + annotator sensitivity/specificity EM."""

    def __init__(self, clf, n_iter=10):
        self.clf, self.n_iter = clf, n_iter

    def fit(self, X, cond, design, Q, R, subj, X_win=None, win_sess=None):
        d = np.asarray(design, float)
        d[d < 0] = np.nan
        r = np.asarray(R, float)
        L = np.stack([d, r], 1)
        mu = np.nanmean(L, 1)
        mu = np.where(np.isfinite(mu), mu, 0.5)
        for _ in range(self.n_iter):
            sens = np.zeros(2)
            spec = np.zeros(2)
            for j in range(2):
                ok = np.isfinite(L[:, j])
                sens[j] = np.clip((mu[ok] * L[ok, j]).sum() / max(mu[ok].sum(), 1e-6), 1e-3, 1 - 1e-3)
                spec[j] = np.clip(((1 - mu[ok]) * (1 - L[ok, j])).sum() / max((1 - mu[ok]).sum(), 1e-6), 1e-3, 1 - 1e-3)
            if X_win is not None:
                clf = fit_soft(self.clf, X_win, expand(mu, win_sess))
                lg = pool(_predict_logit(clf, X_win), np.asarray(win_sess), len(mu))
            else:
                clf = fit_soft(self.clf, X, mu)
                lg = _predict_logit(clf, X)
            for j in range(2):
                ok = np.isfinite(L[:, j])
                lg[ok] += np.where(L[ok, j] == 1, np.log(sens[j] / (1 - spec[j])), np.log((1 - sens[j]) / spec[j]))
            new = expit(lg)
            conv = np.abs(new - mu).mean() < 1e-4
            mu = new
            if conv:
                break
        self.mu_, self.sens_, self.spec_ = mu, sens, spec
        self._fit_final(X, mu, X_win, win_sess)
        return self


# --------------------------------------------------------------------------- torch noise-robust MLPs
class NoiseRobustMLP:
    """MLP trained on binarised self-report with a label-noise-robust objective.

    loss in {"ce", "gce", "sce", "forward", "coteach", "twohead"}.  For "forward", the flip matrix is estimated from
    the design/report disagreement rate on the training set (a natural, label-free-for-S estimate).  "twohead" trains
    one trunk with a condition head and a report head and averages their logits at prediction time.
    """

    def __init__(self, loss="gce", hidden=64, epochs=200, lr=1e-3, wd=1e-3, q=0.7, seed=0, device=None):
        self.loss, self.hidden, self.epochs, self.lr, self.wd, self.q, self.seed = loss, hidden, epochs, lr, wd, q, seed
        self.device = device

    def _net(self, d):
        import torch.nn as nn
        return nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(self.hidden, self.hidden),
                             nn.ReLU(), nn.Linear(self.hidden, 2 if self.loss == "twohead" else 1))

    def fit(self, X, cond, design, Q, R, subj, X_win=None, win_sess=None):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(1)
        dev = self.device or "cpu"
        r = np.asarray(R, float)
        d = np.asarray(design, float)
        d[d < 0] = np.nan
        if X_win is not None:
            X, r, d = X_win, expand(r, win_sess), expand(d, win_sess)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        if self.loss == "twohead":
            ok = np.isfinite(r) | np.isfinite(d)
        else:
            ok = np.isfinite(r)
        Xt = torch.tensor(Xs[ok], dtype=torch.float32, device=dev)
        rt = torch.tensor(np.nan_to_num(r[ok], nan=0.5), dtype=torch.float32, device=dev)
        dt = torch.tensor(np.nan_to_num(d[ok], nan=0.5), dtype=torch.float32, device=dev)
        rmask = torch.tensor(np.isfinite(r[ok]), device=dev)
        dmask = torch.tensor(np.isfinite(d[ok]), device=dev)
        # forward-correction flip matrix from C/R disagreement
        both = np.isfinite(r) & np.isfinite(d)
        eps = float(np.clip((r[both] != d[both]).mean() / 2, 0.01, 0.45)) if both.any() else 0.1
        self.eps_ = eps
        nets = [self._net(Xs.shape[1]).to(dev)]
        if self.loss == "coteach":
            nets.append(self._net(Xs.shape[1]).to(dev))
        opts = [torch.optim.Adam(n.parameters(), lr=self.lr, weight_decay=self.wd) for n in nets]
        n = len(Xt)
        bs = min(64, n)
        forget = 0.0
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            if self.loss == "coteach":
                forget = min(2 * eps, 2 * eps * ep / max(1, self.epochs // 4))
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb = Xt[idx], rt[idx]
                if self.loss == "twohead":
                    out = nets[0](xb)
                    l = torch.zeros((), device=dev)
                    if dmask[idx].any():
                        l = l + F.binary_cross_entropy_with_logits(out[dmask[idx], 0], dt[idx][dmask[idx]])
                    if rmask[idx].any():
                        l = l + F.binary_cross_entropy_with_logits(out[rmask[idx], 1], yb[rmask[idx]])
                    opts[0].zero_grad(); l.backward(); opts[0].step()
                    continue
                if self.loss == "coteach":
                    losses = []
                    for net in nets:
                        z = net(xb).squeeze(1)
                        losses.append(F.binary_cross_entropy_with_logits(z, yb, reduction="none"))
                    k = int(round((1 - forget) * len(idx)))
                    sel = [torch.argsort(l_)[:k] for l_ in losses]
                    for j, (net, opt) in enumerate(zip(nets, opts)):
                        other = sel[1 - j]
                        z = net(xb[other]).squeeze(1)
                        l = F.binary_cross_entropy_with_logits(z, yb[other])
                        opt.zero_grad(); l.backward(); opt.step()
                    continue
                z = nets[0](xb).squeeze(1)
                p = torch.sigmoid(z).clamp(1e-4, 1 - 1e-4)
                py = torch.where(yb > 0.5, p, 1 - p)
                if self.loss == "ce":
                    l = -torch.log(py).mean()
                elif self.loss == "gce":
                    l = ((1 - py ** self.q) / self.q).mean()
                elif self.loss == "sce":
                    # symmetric CE (Wang et al. 2019): CE + reverse CE with log(0) clipped to A = -4
                    l = (-torch.log(py) + 4.0 * (1 - py)).mean()
                elif self.loss == "forward":
                    p_noisy = (1 - eps) * p + eps * (1 - p)
                    py_n = torch.where(yb > 0.5, p_noisy, 1 - p_noisy)
                    l = -torch.log(py_n.clamp(1e-6)).mean()
                else:
                    raise ValueError(self.loss)
                opts[0].zero_grad(); l.backward(); opts[0].step()
        self.nets_ = [n.eval() for n in nets]
        self.dev_ = dev
        return self

    def predict_logit_windows(self, X_win):
        import torch
        Xs = torch.tensor(self.scaler_.transform(X_win), dtype=torch.float32, device=self.dev_)
        with torch.no_grad():
            if self.loss == "twohead":
                return self.nets_[0](Xs).mean(1).cpu().numpy()
            return np.mean([n(Xs).squeeze(1).cpu().numpy() for n in self.nets_], 0)

    def predict_logit(self, X, X_win=None, win_sess=None):
        if X_win is not None:
            return pool(self.predict_logit_windows(X_win), np.asarray(win_sess), len(X))
        return self.predict_logit_windows(X)
