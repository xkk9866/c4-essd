"""Graded TRIAD: a latent *activation* model with the design condition as an instrument.

    z_ij ~ N(v_{c_ij}, 1)                                   condition c shifts the latent activation by v_c
    q^m_ij | z ~ N(alpha^m_i + beta^m z_ij, .),  Cov(q | z) = Sigma   (full residual covariance = method variance)
    g(x_ij) | z ~ N(a + b z_ij, s^2)                        cross-fitted physiological regressor, tempered by tau

The binary state of the base model is the sign of z (P(S=1 | c) = Phi(v_c)).  Because every view is Gaussian in z,
the E-step is a precision-weighted average with a closed form -- design intent, de-baselined reports and physiology
are combined on one interval scale -- and the M-step for beta is an instrumental-variable regression of the report
on the design contrast.  Deployment uses a regressor g trained on the posterior mean activation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .model import TRIAD, pool


class _ScaledRidge:
    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **p):
        self.alpha = p.get("alpha", self.alpha)
        return self

    def _t(self, X):
        # clipping the standardised features bounds the influence of a feature that is nearly constant in the
        # training fold but not in the test fold (a linear regressor, unlike a logistic one, is unbounded)
        return np.clip(self.sc_.transform(X), -self.clip, self.clip)

    def fit(self, X, y, sample_weight=None):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        self.clip = 5.0
        self.sc_ = StandardScaler().fit(X)
        self.r_ = Ridge(alpha=self.alpha).fit(self._t(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.r_.predict(self._t(X))


def _fit_w(reg, X, y, w=None):
    m = clone(reg)
    try:
        m.fit(X, y, sample_weight=w)
    except TypeError:
        m.fit(X, y)
    return m


class GradedTRIAD(TRIAD):
    """Parameters follow TRIAD; `reg` is the deployment regressor prototype (None disables the X view),
    `x_model` the nuisance regressor used inside the E-step ("auto" = standardised ridge), `v0` the prior centre of
    the activation shift for designed stressors (+v0) and calm conditions (-v0), `v_prior` its strength in
    pseudo-sessions, `corr_q` whether the residual covariance between questionnaires is estimated (method variance)
    or fixed diagonal, `weight_x` whether the nuisance/deployment regressions are precision-weighted."""

    def __init__(self, reg=None, tau=1.0, cross_fit=True, n_folds=5, subject_baseline=True, use_c=True, use_q=True,
                 lam_alpha=1.0, v_prior=20.0, v0=1.0, n_iter_cq=50, n_iter_x=16, tol=1e-4, x_model="auto",
                 corr_q=True, anchor=True, anchor_tol=0.005, weight_x=False, deploy_clf=None, random_state=0,
                 verbose=False):
        self.reg, self.clf = reg, reg
        self.deploy_clf = deploy_clf  # optional: deploy a soft-label classifier on P(z > 0) instead of a regressor
        self.tau, self.cross_fit, self.n_folds = tau, cross_fit, n_folds
        self.subject_baseline, self.use_c, self.use_q = subject_baseline, use_c, use_q
        self.lam_alpha, self.v_prior, self.v0 = lam_alpha, v_prior, v0
        self.n_iter_cq, self.n_iter_x, self.tol = n_iter_cq, n_iter_x, tol
        self.corr_q, self.anchor, self.anchor_tol, self.weight_x = corr_q, anchor, anchor_tol, weight_x
        self.random_state, self.verbose = random_state, verbose
        self.x_model = _ScaledRidge() if x_model == "auto" else x_model
        self.pi_prior, self.pi_clip, self.q_dist, self.nu, self.calibrate = v_prior, (0.02, 0.98), "gauss", 4.0, False

    # ------------------------------------------------------------------ M-step
    def _m_v(self, mz, cond):
        v = {}
        a = self.v_prior
        for c in np.unique(cond):
            m = cond == c
            v[c] = float((mz[m].sum() + a * self.v0_[c]) / (m.sum() + a))
        return v

    def _m_q(self, mz, Vz, Q, sidx):
        n, M = Q.shape
        alpha = np.zeros((self.n_subj_, M))
        beta = np.zeros(M)
        mu = np.nanmean(Q, 0)
        for m in range(M):
            q = Q[:, m]
            ok = np.isfinite(q)
            if ok.sum() < 5:
                continue
            qq, zz, vv, ss = q[ok], mz[ok], Vz[ok], sidx[ok]
            a = np.full(self.n_subj_, mu[m])
            b = float(self.beta_[m]) if hasattr(self, "beta_") else max(np.std(qq), 1e-3)
            lam = self.lam_alpha
            for _ in range(5):
                if self.subject_baseline:
                    num = np.bincount(ss, weights=qq - b * zz, minlength=self.n_subj_) + lam * mu[m]
                    den = np.bincount(ss, minlength=self.n_subj_).astype(float) + lam
                    a = num / den
                else:
                    a = np.full(self.n_subj_, float(np.mean(qq - b * zz)))
                r = qq - a[ss]
                b = float((r * zz).sum() / max((zz ** 2 + vv).sum(), 1e-9))
            alpha[:, m], beta[m] = a, b
        # residual covariance (expected under the posterior), pairwise-complete, shrunk toward its diagonal
        res = Q - alpha[sidx] - beta[None, :] * mz[:, None]
        Sigma = np.eye(M)
        for i in range(M):
            for j in range(M):
                ok = np.isfinite(res[:, i]) & np.isfinite(res[:, j])
                if ok.sum() < 5:
                    Sigma[i, j] = 1.0 if i == j else 0.0
                    continue
                Sigma[i, j] = float(np.mean(res[ok, i] * res[ok, j] + beta[i] * beta[j] * Vz[ok]))
        d = np.sqrt(np.maximum(np.diag(Sigma), 1e-6))
        if not self.corr_q:
            Sigma = np.diag(d ** 2)
        else:
            R = Sigma / np.outer(d, d)
            R = 0.9 * R + 0.1 * np.eye(M)  # mild shrinkage keeps Sigma well conditioned
            Sigma = R * np.outer(d, d)
        return alpha, beta, Sigma

    def _q_terms(self, Q, sidx):
        """Per-session precision and precision-weighted mean contribution of the observed questionnaires."""
        n, M = Q.shape
        prec = np.zeros(n)
        pm = np.zeros(n)
        obs = np.isfinite(Q)
        pats = {}
        for i in range(n):
            pats.setdefault(tuple(obs[i]), []).append(i)
        for pat, idx in pats.items():
            o = np.array(pat)
            if not o.any():
                continue
            idx = np.array(idx)
            S = self.Sigma_[np.ix_(o, o)]
            b = self.beta_[o]
            w = np.linalg.solve(S, b)  # Sigma_oo^{-1} beta_o
            prec[idx] = float(b @ w)
            r = Q[np.ix_(idx, np.where(o)[0])] - self.alpha_[sidx[idx]][:, o]
            pm[idx] = r @ w
        return prec, pm

    def _x_terms(self, X, mz, Vz, subj):
        """Cross-fitted regressor g(x) of the posterior mean; calibrated as g ~ N(a + b z, s^2)."""
        Xw, ws = self._Xw, self._ws
        n = len(mz)
        if Xw is None:
            Xw, ws = X, np.arange(n)
        yw, sw = mz[ws], subj[ws]
        ww = (1.0 / Vz[ws]) if self.weight_x else None
        reg = self.x_model if self.x_model is not None else self.reg
        if not self.cross_fit:
            g = _fit_w(reg, Xw, yw, ww).predict(Xw)
        else:
            g = np.zeros(len(yw))
            gkf = GroupKFold(n_splits=min(self.n_folds, len(np.unique(subj))))
            for tr, te in gkf.split(Xw, yw, sw):
                g[te] = _fit_w(reg, Xw[tr], yw[tr], None if ww is None else ww[tr]).predict(Xw[te])
        g = pool(g, ws, n)
        # errors-in-variables calibration of g on z:  b = Cov(g, z) / Var(z),  Var(z) = Var(m) + E[V]
        gm, zm = g.mean(), mz.mean()
        b = float(((g - gm) * (mz - zm)).sum() / max(((mz - zm) ** 2 + Vz).sum(), 1e-9))
        b = max(b, 1e-3)
        a = gm - b * zm
        s2 = float(np.mean((g - a - b * mz) ** 2 + b * b * Vz))
        s2 = max(s2, 1e-3)
        self.x_cal_ = (a, b, s2)
        return b * b / s2, b * (g - a) / s2  # precision, precision-weighted mean contribution

    # ------------------------------------------------------------------ fit
    def fit(self, X, cond, design, Q, R=None, subj=None, X_win=None, win_sess=None):
        if subj is None:
            subj, R = R, None
        self._Xw = None if X_win is None else np.asarray(X_win, dtype=float)
        self._ws = None if win_sess is None else np.asarray(win_sess)
        X = np.asarray(X, dtype=float)
        cond, design = np.asarray(cond), np.asarray(design)
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 1:
            Q = Q[:, None]
        subj = np.asarray(subj)
        self.subj_index_ = {s: i for i, s in enumerate(np.unique(subj))}
        sidx = np.array([self.subj_index_[s] for s in subj])
        self.n_subj_ = len(self.subj_index_)
        n = len(cond)
        v_int = np.where(design == 1, self.v0, np.where(design == 0, -self.v0, 0.0)).astype(float)
        self.v0_ = {c: float(v_int[cond == c].mean()) for c in np.unique(cond)}
        mz = v_int.copy()
        Vz = np.ones(n)
        use_x = self.reg is not None and self.tau > 0
        Ranchor = np.asarray(R, dtype=float) if R is not None else self._binarise(Q[:, 0], sidx)
        if not np.isfinite(Ranchor).any() or not self.use_q:
            Ranchor = None
        xprec, xpm = np.zeros(n), np.zeros(n)
        it, n_x_done, with_x, best = 0, 0, False, None
        self.history_ = []
        while True:
            if not with_x and it >= self.n_iter_cq:
                with_x = use_x
                if not with_x:
                    break
            if with_x and n_x_done >= self.n_iter_x:
                break
            # M-step
            self.v_ = self._m_v(mz, cond) if self.use_c else {c: float(mz.mean()) for c in np.unique(cond)}
            if self.use_q:
                self.alpha_, self.beta_, self.Sigma_ = self._m_q(mz, Vz, Q, sidx)
            if with_x:
                xprec, xpm = self._x_terms(X, mz, Vz, subj)
            # E-step: precision-weighted combination of the views on the activation scale
            prec = np.ones(n)
            pm = np.array([self.v_[c] for c in cond])
            if self.use_q:
                qp, qm = self._q_terms(Q, sidx)
                prec, pm = prec + qp, pm + qm
            if with_x:
                prec, pm = prec + self.tau * xprec, pm + self.tau * xpm
            new_mz, new_Vz = pm / prec, 1.0 / prec
            delta = float(np.abs(new_mz - mz).mean())
            mz, Vz = new_mz, new_Vz
            gamma = norm.cdf(mz / np.sqrt(Vz))
            self.history_.append((with_x, delta))
            it += 1
            if with_x:
                n_x_done += 1
                if self.anchor and Ranchor is not None:
                    eps_now = self._flip_rate(gamma, Ranchor)
                    if best is None or eps_now <= best[0] + self.anchor_tol:
                        snap = dict(mz=mz.copy(), Vz=Vz.copy(), v=dict(self.v_), alpha=self.alpha_.copy() if self.use_q else None,
                                    beta=self.beta_.copy() if self.use_q else None, Sigma=self.Sigma_.copy() if self.use_q else None)
                        best = (min(eps_now, best[0]) if best else eps_now, snap)
                    else:
                        s = best[1]
                        mz, Vz, self.v_ = s["mz"], s["Vz"], s["v"]
                        if self.use_q:
                            self.alpha_, self.beta_, self.Sigma_ = s["alpha"], s["beta"], s["Sigma"]
                        gamma = norm.cdf(mz / np.sqrt(Vz))
                        break
            if delta < self.tol:
                if with_x or not use_x:
                    break
                with_x = True
        self.n_iter_, self.n_x_used_ = it, n_x_done
        self.mz_, self.Vz_, self.gamma_ = mz, Vz, gamma
        self.pi_ = {c: float(norm.cdf(v)) for c, v in self.v_.items()}
        self.sigma_ = np.sqrt(np.diag(self.Sigma_)) if self.use_q else None
        self.cond_, self.design_, self.sidx_ = cond, design, sidx
        self.reg_ = self.clf_ = None
        if self.deploy_clf is not None:
            from .model import fit_soft
            if self._Xw is not None:
                self.clf_ = fit_soft(self.deploy_clf, self._Xw, gamma[self._ws])
            else:
                self.clf_ = fit_soft(self.deploy_clf, X, gamma)
        elif self.reg is not None:
            w = (1.0 / Vz) if self.weight_x else None
            if self._Xw is not None:
                self.reg_ = _fit_w(self.reg, self._Xw, mz[self._ws], None if w is None else w[self._ws])
            else:
                self.reg_ = _fit_w(self.reg, X, mz, w)
        self._Xw = self._ws = None
        self.eps_C_ = self._flip_rate(gamma, np.where(design >= 0, design, np.nan))
        return self

    def predict_proba(self, X):
        return norm.cdf(self.predict_logit(X))

    def _score(self, X):
        if self.clf_ is not None:
            from .model import _predict_logit
            return _predict_logit(self.clf_, X)
        return self.reg_.predict(X)

    def predict_logit(self, X, X_win=None, win_sess=None):
        if X_win is None:
            return self._score(np.asarray(X, dtype=float))
        return pool(self._score(np.asarray(X_win, dtype=float)), np.asarray(win_sess), len(X))

    def predict_logit_windows(self, X_win):
        return self._score(np.asarray(X_win, dtype=float))
