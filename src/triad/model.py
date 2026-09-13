"""TRIAD: a latent-state model that triangulates design condition (C), self-report (Q) and physiology (X).

Generative structure  C -> S -> (Q, X):
    S_ij ~ Bernoulli(pi_{c_ij})                        manipulation efficacy of each design condition
    Q^m_ij | S_ij ~ N(alpha^m_i + beta^m S_ij, sigma_m^2)  per-participant response baseline, shared sensitivity
    X_ij  | S_ij  ~ p(x | s)  represented discriminatively by a classifier f_theta(x) ~ p(S=1 | x)

Inference is an EM whose E-step combines the three views on the logit scale and whose M-step has closed forms for
(pi, alpha, beta, sigma) and a soft-label refit for f_theta.  The X view enters through *cross-fitted* (out-of-fold,
grouped by participant) predictions so the classifier cannot confirm its own labels.  After fitting, only f_theta is
needed at deployment; (pi, alpha, beta, sigma, gamma) are interpretable by-products: how effective each stimulus was,
how each person scales the questionnaire, how noisy self-reports are relative to the latent state.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit as _logit
from sklearn.base import clone
from sklearn.model_selection import GroupKFold


def _safe_logit(p, eps=1e-4):
    return _logit(np.clip(p, eps, 1 - eps))


def fit_soft(clf, X, gamma, sample_weight=None):
    """Fit a sklearn classifier on soft labels gamma in [0,1] by duplicating rows with weights (gamma, 1-gamma)."""
    n = len(gamma)
    Xd = np.vstack([X, X])
    yd = np.concatenate([np.ones(n), np.zeros(n)])
    w = np.concatenate([gamma, 1 - gamma])
    if sample_weight is not None:
        w = w * np.concatenate([sample_weight, sample_weight])
    keep = w > 1e-6
    m = clone(clf)
    m.fit(Xd[keep], yd[keep], sample_weight=w[keep])
    return m


def _predict_logit(m, X):
    if hasattr(m, "predict_proba"):
        return _safe_logit(m.predict_proba(X)[:, 1])
    return m.decision_function(X)


def expand(y, win_sess):
    """Broadcast a session-level array to windows."""
    return None if y is None else np.asarray(y)[np.asarray(win_sess)]


def pool(v, win_sess, n_sess):
    """Average window-level values per session (sessions without windows get NaN)."""
    s = np.bincount(win_sess, weights=v, minlength=n_sess)
    c = np.bincount(win_sess, minlength=n_sess)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


class _ScaledLR:
    """Standardised L2 logistic regression with sample weights (the default E-step nuisance model)."""

    def __init__(self, C=0.1):
        self.C = C

    def get_params(self, deep=True):
        return {"C": self.C}

    def set_params(self, **p):
        self.C = p.get("C", self.C)
        return self

    def fit(self, X, y, sample_weight=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.sc_ = StandardScaler().fit(X)
        self.lr_ = LogisticRegression(C=self.C, max_iter=5000).fit(self.sc_.transform(X), y, sample_weight=sample_weight)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.sc_.transform(X))


class TRIAD:
    """Three-view latent-state EM.

    Parameters
    ----------
    clf : sklearn classifier prototype (must accept sample_weight).  None disables the X view (tau = 0).
    tau : temperature on the X likelihood-ratio term (0 = ignore physiology in the E-step).
    cross_fit : use out-of-fold classifier predictions in the E-step (grouped by participant).
    subject_baseline : per-participant alpha_i (True) or a shared alpha (False).
    use_c / use_q : ablation switches for the condition prior and the self-report likelihood.
    lam_alpha : ridge shrinkage of alpha_i toward the grand mean (in units of pseudo-observations).
    n_iter_cq : EM iterations run without the classifier (cheap warm start); n_iter_x : iterations with X view.
    pi_init : dict condition -> initial P(S=1 | c); defaults derived from the binary design intent (0.85 / 0.15 / 0.5).
    """

    def __init__(self, clf=None, tau=1.0, cross_fit=True, n_folds=5, subject_baseline=True, use_c=True, use_q=True,
                 lam_alpha=1.0, pi_prior=20.0, n_iter_cq=50, n_iter_x=16, tol=1e-4, pi_init=None, pi_clip=(0.02, 0.98),
                 x_model="auto", calibrate=False, q_dist="gauss", nu=4.0, anchor=True, anchor_tol=0.005,
                 x_bound=None, deploy="soft", random_state=0, verbose=False):
        """clf: deployment classifier trained on the final posterior.  x_model: nuisance classifier used inside the
        E-step ("auto" = standardised logistic regression, a well-calibrated choice; None = same as clf).
        q_dist: "gauss" or "t" (Student-t with nu degrees of freedom; a heavy-tailed report likelihood whose
        log-likelihood ratio is bounded, so a single extreme questionnaire score cannot override the other views).
        anchor: report-anchored early stopping of the physiological sweeps (stop once the posterior's expected
        disagreement with the binarised self-report rises more than anchor_tol above its best value); with
        anchor=False exactly n_iter_x sweeps are run."""
        self.clf, self.tau, self.cross_fit, self.n_folds = clf, tau, cross_fit, n_folds
        self.subject_baseline, self.use_c, self.use_q = subject_baseline, use_c, use_q
        self.lam_alpha, self.pi_prior, self.n_iter_cq, self.n_iter_x, self.tol = lam_alpha, pi_prior, n_iter_cq, n_iter_x, tol
        self.pi_init, self.pi_clip, self.random_state, self.verbose = pi_init, pi_clip, random_state, verbose
        self.calibrate, self.q_dist, self.nu = calibrate, q_dist, nu
        self.anchor, self.anchor_tol, self.x_bound, self.deploy = anchor, anchor_tol, x_bound, deploy
        self.x_model = _ScaledLR() if x_model == "auto" else x_model

    # ------------------------------------------------------------------ M-step pieces
    def _m_pi(self, gamma, cond):
        """MAP update of the manipulation efficacy under a Beta prior centred on the design intent p0_c.

        pi_c = (sum_j gamma_cj + a * p0_c) / (n_c + a), with a = pi_prior pseudo-sessions: the experimenter's intent
        is worth `a` sessions of evidence.  Small conditions therefore stay close to the intent, while a condition with
        many sessions can override it when reports and physiology consistently disagree (e.g. tasks that were designed
        as stressors but did not raise anxiety).  a -> inf recovers the condition-only labelling; a = 0 is fully
        data-driven and, with few sessions per condition, may drift to degenerate solutions.
        """
        pi = {}
        conds = np.unique(cond)
        if isinstance(self.pi_prior, str) and self.pi_prior.startswith("median"):
            # robust empirical-Bayes centre: conditions with the same design intent share a Beta prior centred on the
            # *median* posterior efficacy of that group (a failed stressor is the exception the median ignores; when
            # every stressor is nearly always effective the centre moves to ~0.95 and the prior no longer biases
            # eps_C toward 0.15).  Strength a in pseudo-sessions as before (default 20; "median:50" etc.).
            a = float(self.pi_prior.split(":")[1]) if ":" in self.pi_prior else 20.0
            phat = {c: float(gamma[cond == c].mean()) for c in conds}
            self.eb_ = {}
            for g in set(self.group_.values()):
                cs = [c for c in conds if self.group_[c] == g]
                m0 = float(np.mean([self.p0_[c] for c in cs]))
                # median over conditions, with the design intent counted as one extra (pseudo-)condition
                m = float(np.median([phat[c] for c in cs] + [m0]))
                m = float(np.clip(m, 0.05, 0.95))
                self.eb_[g] = (m, a)
                for c in cs:
                    mm = cond == c
                    pi[c] = float(np.clip((gamma[mm].sum() + a * m) / (mm.sum() + a), *self.pi_clip))
            return pi
        if self.pi_prior == "eb":
            # hierarchical (empirical-Bayes) prior: conditions with the same design intent share a Beta(a m, a (1-m))
            # prior whose centre m and strength a are estimated from the current posterior by the method of moments.
            # If all stressors are (nearly) equally effective the prior pools them; if one fails, the between-
            # condition variance grows, a shrinks and that condition is free to deviate.  Nothing is fixed at 0.85.
            cnt = {c: float((cond == c).sum()) for c in conds}
            phat = {c: float(gamma[cond == c].mean()) for c in conds}
            self.eb_ = {}
            for g in set(self.group_.values()):
                cs = [c for c in conds if self.group_[c] == g]
                n_g = np.array([cnt[c] for c in cs])
                p_g = np.array([phat[c] for c in cs])
                # weak anchor toward the design intent keeps the centre identified with very few conditions
                a0, m0 = 5.0, float(np.mean([self.p0_[c] for c in cs]))
                m = float((np.sum(n_g * p_g) + a0 * m0) / (n_g.sum() + a0))
                m = float(np.clip(m, 0.05, 0.95))
                if len(cs) >= 2:
                    between = float(np.average((p_g - m) ** 2, weights=n_g))
                    within = float(np.average(m * (1 - m) / n_g, weights=n_g))
                    V = between - within
                    a = m * (1 - m) / V - 1 if V > 1e-6 else 200.0
                    a = float(np.clip(a, 2.0, 200.0))
                else:
                    a = 20.0
                self.eb_[g] = (m, a)
                for c in cs:
                    mm = cond == c
                    pi[c] = float(np.clip((gamma[mm].sum() + a * m) / (mm.sum() + a), *self.pi_clip))
            return pi
        a = self.pi_prior
        for c in conds:
            m = cond == c
            pi[c] = float(np.clip((gamma[m].sum() + a * self.p0_[c]) / (m.sum() + a), *self.pi_clip))
        return pi

    def _t_weights(self, r2, s2):
        """Scale-mixture weights of the Student-t: E[lambda | r] = (nu+1)/(nu + r^2/s^2); 1 for the Gaussian."""
        if self.q_dist != "t":
            return np.ones_like(r2)
        return (self.nu + 1.0) / (self.nu + r2 / s2)

    def _m_q(self, gamma, Q, subj):
        """Weighted least squares for alpha_i, beta, sigma per questionnaire column (NaN allowed).

        For the Student-t report model the update is the usual scale-mixture EM: every (session, state) residual is
        down-weighted by its posterior precision multiplier w = (nu+1)/(nu + r^2/sigma^2), so far-outlying reports pull
        the baseline, the sensitivity and the scale less than under a Gaussian.  lam_alpha="auto" uses the empirical-
        Bayes random-effects estimate lam = sigma^2 / tau_alpha^2 (tau_alpha^2 = between-participant variance of the
        de-activated report residuals), otherwise the given pseudo-observation count.
        """
        n, M = Q.shape
        alpha = np.zeros((self.n_subj_, M))
        beta = np.zeros(M)
        sigma = np.ones(M)
        lam_used = np.zeros(M)
        for m in range(M):
            q = Q[:, m]
            ok = np.isfinite(q)
            if ok.sum() < 5:
                beta[m], sigma[m] = 0.0, 1.0
                continue
            g, qq, ss = gamma[ok], q[ok], subj[ok]
            mu = qq.mean()
            a = np.full(self.n_subj_, mu)
            b = float(self.beta_[m]) if hasattr(self, "beta_") else max(qq.std(), 1e-3)
            s2 = float(self.sigma_[m] ** 2) if hasattr(self, "sigma_") else max(qq.var(), 1e-6)
            lam = self.lam_alpha if self.lam_alpha != "auto" else 1.0
            for _ in range(5):
                r1, r0 = qq - a[ss] - b, qq - a[ss]
                w1, w0 = g * self._t_weights(r1 ** 2, s2), (1 - g) * self._t_weights(r0 ** 2, s2)
                if self.lam_alpha == "auto" and self.subject_baseline:
                    # empirical Bayes: alpha_i ~ N(mu, tau^2); tau^2 = var of per-participant means of the
                    # de-activated report (q - gamma*b) minus its sampling variance sigma^2 / n_i
                    cnt = np.bincount(ss, minlength=self.n_subj_)
                    means = np.bincount(ss, weights=qq - g * b, minlength=self.n_subj_) / np.maximum(cnt, 1)
                    has = cnt > 0
                    tau2 = max(means[has].var() - s2 * np.mean(1.0 / cnt[has]), 0.05 * s2)
                    lam = s2 / tau2
                # alpha_i | beta: minimise sum w1 (q-a-b)^2 + w0 (q-a)^2 + lam (a-mu)^2
                if self.subject_baseline:
                    num = np.bincount(ss, weights=w1 * (qq - b) + w0 * qq, minlength=self.n_subj_) + lam * mu
                    den = np.bincount(ss, weights=w1 + w0, minlength=self.n_subj_) + lam
                    a = num / den
                else:
                    a = np.full(self.n_subj_, (w1 * (qq - b) + w0 * qq).sum() / (w1 + w0).sum())
                # beta | alpha
                r = qq - a[ss]
                b = float((w1 * r).sum() / max(w1.sum(), 1e-6))
                # sigma^2 | alpha, beta (scale-mixture EM update; reduces to the weighted residual variance)
                r1, r0 = qq - a[ss] - b, qq - a[ss]
                s2 = float(max((w1 * r1 ** 2 + w0 * r0 ** 2).sum() / len(qq), 1e-6))
            alpha[:, m], beta[m], sigma[m], lam_used[m] = a, b, float(np.sqrt(s2)), lam
        self.lam_used_ = lam_used
        return alpha, beta, sigma

    def _q_loglr(self, Q, subj):
        """sum_m log p(q | S=1) - log p(q | S=0) under the Gaussian or Student-t report model (NaN-safe)."""
        n, M = Q.shape
        out = np.zeros(n)
        for m in range(M):
            q = Q[:, m]
            ok = np.isfinite(q)
            a = self.alpha_[subj[ok], m]
            b, s = self.beta_[m], self.sigma_[m]
            r1, r0 = q[ok] - a - b, q[ok] - a
            if self.q_dist == "t":
                # log t_nu(r1; 0, s) - log t_nu(r0; 0, s): bounded in |q|, unlike the Gaussian ratio
                out[ok] += -0.5 * (self.nu + 1) * (np.log1p(r1 ** 2 / (self.nu * s ** 2)) - np.log1p(r0 ** 2 / (self.nu * s ** 2)))
            else:
                out[ok] += (r0 ** 2 - r1 ** 2) / (2 * s ** 2)
        return out

    def _x_loglr(self, X, gamma, subj):
        """Cross-fitted classifier log-odds minus prior log-odds = log p(x|1)/p(x|0) (tempered later).

        When window-level features were supplied to fit(), the classifier is trained on windows with the session
        posterior broadcast as soft label and window log-odds are averaged back to sessions.
        """
        prior = _safe_logit(gamma.mean())
        Xw, ws = self._Xw, self._ws
        n = len(gamma)
        if Xw is None:
            Xw, ws = X, np.arange(n)
        gw, sw = gamma[ws], subj[ws]
        clf = self.x_model if self.x_model is not None else self.clf
        if not self.cross_fit:
            m = fit_soft(clf, Xw, gw)
            lg = _predict_logit(m, Xw)
        else:
            lg = np.zeros(len(gw))
            n_groups = len(np.unique(subj))
            gkf = GroupKFold(n_splits=min(self.n_folds, n_groups))
            for tr, te in gkf.split(Xw, gw, sw):
                m = fit_soft(clf, Xw[tr], gw[tr])
                lg[te] = _predict_logit(m, Xw[te])
        if self.calibrate == "cq":
            # the physiological score enters through a one-dimensional logistic link fitted to the posterior computed
            # from design and reports alone (never from physiology), so its evidence scale is set by its out-of-fold
            # validity against the independent views and cannot reinforce itself across sweeps
            from sklearn.linear_model import LogisticRegression
            ls = pool(lg, ws, n)
            cal = fit_soft(LogisticRegression(C=1.0), ls[:, None], self._gamma_cq)
            return _predict_logit(cal, ls[:, None]) - _safe_logit(self._gamma_cq.mean())
        if self.calibrate == "report" and self._Ranchor is not None:
            # report-calibrated link: the physiological score enters only through a one-dimensional logistic link
            # fitted to the *independent* binarised report, de-attenuated by the report's estimated flip rates
            # (P(R=1|x) = e0 + (1-e0-e1) P(S=1|x)).  A score that does not predict the reports (a task confound
            # such as moving vs. sitting) therefore gets a flat link and contributes nothing to the E-step.
            from sklearn.linear_model import LogisticRegression
            ls = pool(lg, ws, n)
            R = self._Ranchor
            ok = np.isfinite(R)
            cal = LogisticRegression(C=1.0).fit(ls[ok, None], R[ok].astype(int))
            pR = cal.predict_proba(ls[:, None])[:, 1]
            nz = self.label_noise_from(gamma, R)
            den = max(1.0 - nz["e0"] - nz["e1"], 0.2)
            pS = np.clip((pR - nz["e0"]) / den, 1e-3, 1 - 1e-3)
            return _safe_logit(pS) - prior
        if self.calibrate:
            # Platt scaling of the (out-of-fold) log-odds against the current posterior: a nuisance classifier's
            # scores are only used through a calibrated one-dimensional link, so tree ensembles cannot dominate the
            # E-step with over-confident probabilities
            from sklearn.linear_model import LogisticRegression
            cal = fit_soft(LogisticRegression(C=1.0), lg[:, None], gw)
            lg = _predict_logit(cal, lg[:, None])
        return pool(lg, ws, n) - prior

    # ------------------------------------------------------------------ fit
    def fit(self, X, cond, design, Q, R=None, subj=None, X_win=None, win_sess=None):
        """X [n,d]; cond [n] hashable condition ids; design [n] in {0,1,-1}; Q [n,M] (NaN ok); subj [n] ids.

        R (binarised report, NaN for ties) is used only as the anchor of the early-stopping rule for the physiological
        sweeps; when omitted it is derived from Q by a participant median split.
        X_win [n_w,d] / win_sess [n_w] optionally provide window-level features (index into sessions) for the X view.
        """
        if subj is None:  # called positionally as fit(X, cond, design, Q, subj)
            subj, R = R, None
        self._Xw = None if X_win is None else np.asarray(X_win, dtype=float)
        self._ws = None if win_sess is None else np.asarray(win_sess)
        X = np.asarray(X, dtype=float)
        cond = np.asarray(cond)
        design = np.asarray(design)
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 1:
            Q = Q[:, None]
        subj = np.asarray(subj)
        self.subj_index_ = {s: i for i, s in enumerate(np.unique(subj))}
        sidx = np.array([self.subj_index_[s] for s in subj])
        self.n_subj_ = len(self.subj_index_)
        n = len(cond)
        # initial posterior / prior centre from design intent
        gamma = np.where(design == 1, 0.85, np.where(design == 0, 0.15, 0.5)).astype(float)
        if self.pi_init:
            gamma = np.array([self.pi_init.get(c, g) for c, g in zip(cond, gamma)])
        self.p0_ = {c: float(gamma[cond == c].mean()) for c in np.unique(cond)}
        self.group_ = {c: int(np.round(np.mean(design[cond == c]))) for c in np.unique(cond)}
        self.history_ = []
        use_x = self.clf is not None and self.tau > 0
        # anchor label for the physiological sweeps: the binarised self-report (participant median split), computed
        # from Q if not supplied, so that the anchor never depends on the design condition
        Ranchor = np.asarray(R, dtype=float) if R is not None else self._binarise(Q[:, 0], sidx)
        if not np.isfinite(Ranchor).any() or not self.use_q:
            Ranchor = None
        self._Ranchor = Ranchor
        self._gamma_cq = gamma.copy()
        xterm = np.zeros(n)
        it, n_x_done = 0, 0
        with_x = False
        best = None  # (eps_R, state) of the best physiological sweep so far
        self.n_x_used_ = 0
        while True:
            if not with_x and it >= self.n_iter_cq:
                with_x = use_x
                if not with_x:
                    break
            if with_x and n_x_done >= self.n_iter_x:
                break
            # M-step
            self.pi_ = self._m_pi(gamma, cond) if self.use_c else {c: float(gamma.mean()) for c in np.unique(cond)}
            if self.use_q:
                self.alpha_, self.beta_, self.sigma_ = self._m_q(gamma, Q, sidx)
            if with_x:
                xterm = self._x_loglr(X, gamma, subj)
            # E-step
            lg = _safe_logit(np.array([self.pi_[c] for c in cond]))
            if self.use_q:
                lg = lg + self._q_loglr(Q, sidx)
            self._gamma_cq = expit(lg)
            if with_x:
                xt = xterm if self.x_bound is None else np.clip(xterm, -self.x_bound, self.x_bound)
                lg = lg + self.tau * xt
            new_gamma = expit(lg)
            delta = np.abs(new_gamma - gamma).mean()
            gamma = new_gamma
            self.history_.append((with_x, delta))
            if self.verbose:
                print(f"  it {it:2d} x={with_x!s:5s} delta={delta:.5f} pi={ {k: round(v, 2) for k, v in self.pi_.items()} }")
            it += 1
            if with_x:
                n_x_done += 1
                if self.anchor and Ranchor is not None:
                    # report-anchored early stopping: a physiological sweep is accepted only while the posterior keeps
                    # (or improves) its agreement with the self-report view.  The classifier view is flexible enough to
                    # represent any partition of the physiology, so left alone it can redefine the state (e.g. moving
                    # vs. sitting); the reports, which are independent of both design and sensors, anchor it.
                    eps_now = self._flip_rate(gamma, Ranchor)
                    if best is None or eps_now <= best[0] + self.anchor_tol:
                        if best is None or eps_now < best[0]:
                            best = (eps_now, self._snapshot(gamma))
                        else:
                            best = (best[0], self._snapshot(gamma))
                    else:
                        gamma = self._restore(best[1])
                        break
            if delta < self.tol:
                if with_x or not use_x:
                    break
                with_x = True  # C+Q phase converged: move on to the X iterations
        self.n_iter_ = it
        self.n_x_used_ = n_x_done
        self.gamma_ = gamma
        self.cond_ = cond
        self.design_ = design
        self.sidx_ = sidx
        # final deployable classifier on all data (windows if available)
        if self.clf is not None:
            g_dep = gamma if self.deploy == "soft" else (gamma > 0.5).astype(float)
            if self._Xw is not None:
                self.clf_ = fit_soft(self.clf, self._Xw, g_dep[self._ws])
            else:
                self.clf_ = fit_soft(self.clf, X, g_dep)
        else:
            self.clf_ = None
        self._Xw = self._ws = None  # do not keep training windows around
        # by-products
        self.eps_C_ = self._flip_rate(gamma, np.where(design >= 0, design, np.nan))
        return self

    @staticmethod
    def _flip_rate(gamma, lab):
        ok = np.isfinite(lab)
        if ok.sum() == 0:
            return np.nan
        g, l = gamma[ok], lab[ok]
        return float((g * (l == 0) + (1 - g) * (l == 1)).mean())

    @staticmethod
    def _binarise(q, sidx):
        """Participant median split of a report column (NaN for ties / participants with < 2 reports)."""
        R = np.full(len(q), np.nan)
        for s in np.unique(sidx):
            m = (sidx == s) & np.isfinite(q)
            if m.sum() < 2:
                continue
            med = np.median(q[m])
            R[m] = np.where(q[m] > med, 1.0, np.where(q[m] < med, 0.0, np.nan))
        return R

    def _snapshot(self, gamma):
        return dict(gamma=gamma.copy(), pi=dict(self.pi_), alpha=getattr(self, "alpha_", None),
                    beta=getattr(self, "beta_", None), sigma=getattr(self, "sigma_", None))

    def _restore(self, snap):
        self.pi_ = dict(snap["pi"])
        if snap["alpha"] is not None:
            self.alpha_, self.beta_, self.sigma_ = snap["alpha"], snap["beta"], snap["sigma"]
        return snap["gamma"].copy()

    def report_flip_rate(self, R):
        """Posterior-expected P(R != S) for a binarised self-report R (NaN for ties)."""
        return self._flip_rate(self.gamma_, np.asarray(R, dtype=float))

    def label_noise(self, lab):
        """Posterior-expected noise structure of a binary label relative to S: dict(p, r1, e0, e1, eps, kappa, ceiling).

        e1 = P(lab=0 | S=1) (miss rate), e0 = P(lab=1 | S=0) (false-alarm rate), p = P(S=1), r1 = P(lab=1),
        kappa = attenuation factor of Proposition 2 (general form), ceiling = 1/2 + kappa/2 = max AUC against lab.
        """
        return self.label_noise_from(self.gamma_, lab)

    @staticmethod
    def label_noise_from(gamma, lab):
        lab = np.asarray(lab, dtype=float)
        ok = np.isfinite(lab)
        g, l = gamma[ok], lab[ok]
        p = float(g.mean())
        r1 = float(l.mean())
        e1 = float((g * (l == 0)).sum() / max(g.sum(), 1e-9))
        e0 = float(((1 - g) * (l == 1)).sum() / max((1 - g).sum(), 1e-9))
        kappa = attenuation(p, e0, e1)
        return dict(p=p, r1=r1, e0=e0, e1=e1, eps=float((g * (l == 0) + (1 - g) * (l == 1)).mean()), kappa=kappa,
                    ceiling=0.5 + 0.5 * kappa, n=int(ok.sum()))

    def predict_proba(self, X):
        return self.clf_.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict_logit(self, X, X_win=None, win_sess=None):
        """Session-level log-odds; with windows, the mean window log-odds per session."""
        if X_win is None:
            return _predict_logit(self.clf_, np.asarray(X, dtype=float))
        return pool(_predict_logit(self.clf_, np.asarray(X_win, dtype=float)), np.asarray(win_sess), len(X))

    def predict_logit_windows(self, X_win):
        return _predict_logit(self.clf_, np.asarray(X_win, dtype=float))


def attenuation(p, e0, e1):
    """Proposition 2 (general form).  For a score g independent of the noisy label L given S,
        AUC_L(g) - 1/2 = kappa * (AUC_S(g) - 1/2),   kappa = p(1-p)(1 - e0 - e1) / (r1 (1 - r1)),
    with p = P(S=1), e1 = P(L=0|S=1), e0 = P(L=1|S=0), r1 = P(L=1) = p(1-e1) + (1-p)e0.
    Symmetric balanced noise (p=1/2, e0=e1=eps) gives kappa = 1 - 2 eps."""
    r1 = p * (1 - e1) + (1 - p) * e0
    if r1 <= 0 or r1 >= 1:
        return np.nan
    return p * (1 - p) * (1 - e0 - e1) / (r1 * (1 - r1))


def denoised_auc(auc_obs, eps=None, kappa=None):
    """AUC against the latent state implied by an observed AUC against a noisy label (Proposition 2)."""
    if kappa is None:
        if eps is None or not np.isfinite(eps) or eps >= 0.5:
            return np.nan
        kappa = 1 - 2 * eps
    if not np.isfinite(kappa) or kappa <= 0.05:
        return np.nan
    return 0.5 + (auc_obs - 0.5) / kappa
