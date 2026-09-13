"""Aggregate results/runs into LaTeX tables, figures and a summary JSON for the paper.

Outputs (paper/tables/*.tex, paper/figures/*.pdf, results/summary.json):
  main_<family>.tex      benchmark table per classifier family (datasets x methods x metrics)
  ablation.tex           TRIAD ablations on MultiPhysio-HRC
  byproducts.tex         manipulation efficacy, report/design flip rates, ceilings, denoised AUC
  case_day2.tex          MultiPhysio-HRC day-2 case study
  semi_synthetic.tex     AUC against the known latent state and recovered noise rates
  significance.tex       subject-level bootstrap CIs for TRIAD minus best baseline
  fig_agreement.pdf, fig_pi.pdf, fig_semi.pdf, fig_sensitivity.pdf, fig_ceiling.pdf, fig_cross.pdf
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from triad.evaluation import DATASETS, bootstrap_diff, load_dataset, metrics
from triad.model import denoised_auc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "results", "runs")
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

NAMES = {
    "cond": "Condition-only (B1)", "report": "Self-report-only (B2)", "regress": "Report regression (B3)",
    "twohead": "Two-head ensemble (B4)", "mlp_ce": "MLP, CE on report", "mlp_gce": "MLP, GCE (B5)", "mlp_sce": "MLP, SCE (B5)",
    "mlp_forward": "MLP, forward corr. (B5)", "mlp_coteach": "MLP, co-teaching (B5)", "mlp_twohead": "MLP, two-head (B4)",
    "filter": "Agreement filter (B6)", "ds": "Dawid--Skene (B7)", "raykar": "Raykar LFC (B8)",
    "triad_cq": "TRIAD (C+Q, $\\tau=0$)", "triad": "\\textbf{TRIAD} (C+Q+X)", "triad_nocf": "TRIAD w/o cross-fitting",
    "triad_noalpha": "TRIAD w/o person baselines $\\alpha_i$", "triad_noprior": "TRIAD w/o design prior ($a=0$)",
    "triad_noc": "TRIAD w/o condition view", "triad_noq": "TRIAD w/o self-report view", "triad_mq": "TRIAD, all questionnaires",
    "triad_noanchor": "TRIAD w/o report-validity constraint (16 sweeps)", "triad_student": "TRIAD, Student-$t$ reports ($\\nu=4$)",
    "triad_eb": "TRIAD, empirical-Bayes $\\lambda_\\alpha$", "triad_x1": "TRIAD, 1 sweep", "triad_x2": "TRIAD, 2 sweeps",
    "triad_x8": "TRIAD, 8 sweeps", "triad_x16": "TRIAD, 16 sweeps",
    "triad_rcal": "TRIAD, report-calibrated link (unconstrained)", "triad_hard": "TRIAD, MAP labels for deployment",
    "triad_samex": "TRIAD, nuisance = deployment family", "triad_med": "TRIAD, empirical-Bayes prior centre",
    "gtriad": "Graded activation (ridge deployment)", "gtriad_c": "Graded activation (classifier deployment)",
}
DSN = {"mphrc": "MultiPhysio-HRC", "wesad": "WESAD", "sensecobot": "SenseCobot", "mphrc_all": "MultiPhysio-HRC (day 1+2)"}
ORDER = ["cond", "report", "regress", "twohead", "mlp_gce", "mlp_sce", "mlp_forward", "mlp_coteach", "filter", "ds", "raykar", "triad_cq", "triad"]
METRICS = ["auc_C", "auc_R", "rho", "auc_agree"]
MNAMES = {"auc_C": "AUC$_C$", "auc_R": "AUC$_R$", "rho": "$\\rho$", "auc_agree": "AUC$_{\\cap}$", "auc_R_within": "AUC$_R^{\\text{within}}$",
          "win_auc_C": "win-AUC$_C$", "win_auc_R": "win-AUC$_R$", "win_auc_agree": "win-AUC$_\\cap$"}


def load_runs():
    rows = []
    for f in glob.glob(os.path.join(RUNS, "**", "*.json"), recursive=True):
        r = json.load(open(f))
        r["file"] = f
        rows.append(r)
    df = pd.DataFrame(rows)
    return df


def agg(df, ds, q, level, fam, method):
    sub = df[(df.dataset == ds) & (df.q == q) & (df.level == level) & (df.family == fam) & (df.method == method)]
    if len(sub) == 0:
        return None
    out = {m: sub[m].mean() for m in METRICS + ["auc_R_within"] if m in sub}
    out.update({m + "_sd": sub[m].std(ddof=0) for m in METRICS if m in sub})
    for m in ["win_auc_C", "win_auc_R", "win_auc_agree"]:
        if m in sub and sub[m].notna().any():
            out[m] = sub[m].mean()
    out["n_seeds"] = len(sub)
    if "triad" in sub and sub["triad"].notna().any():
        out["triad"] = sub["triad"].dropna().iloc[0]
    return out


def fmt(v, best=False, sd=None):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.3f}"
    if sd is not None and np.isfinite(sd) and sd > 0:
        s += f"\\tiny$\\pm${sd:.2f}"
    return f"\\textbf{{{s}}}" if best else s


def table_main(df, fam, level="task", methods=ORDER, datasets=("mphrc", "wesad", "sensecobot"), fname=None, caption=None, label=None):
    lines = []
    cols = "l" + "".join("cccc" for _ in datasets)
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{DSN[d]}}}" for d in datasets) + " \\\\")
    lines.append(" ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(datasets))))
    lines.append(" & " + " & ".join(" & ".join(MNAMES[m] for m in METRICS) for _ in datasets) + " \\\\")
    lines.append("\\midrule")
    cells = {}
    for d in datasets:
        q = DATASETS[d]["primary"]
        for meth in methods:
            f = fam
            if meth.startswith("mlp_") or meth == "regress":
                f = "lr"  # family-independent methods are stored under lr
            cells[(d, meth)] = agg(df, d, q, level, f, meth)
    for d in datasets:
        for m in METRICS:
            vals = {meth: cells[(d, meth)][m] for meth in methods if cells[(d, meth)] and np.isfinite(cells[(d, meth)].get(m, np.nan))}
            best = max(vals, key=vals.get) if vals else None
            for meth in methods:
                if cells[(d, meth)]:
                    cells[(d, meth)][m + "_best"] = (meth == best) or (best is not None and abs(vals.get(meth, -9) - vals[best]) < 5e-4)
    for meth in methods:
        if all(cells[(d, meth)] is None for d in datasets):
            continue
        row = [NAMES.get(meth, meth)]
        for d in datasets:
            c = cells[(d, meth)]
            if c is None:
                row += ["--"] * 4
            else:
                row += [fmt(c[m], c.get(m + "_best", False), c.get(m + "_sd") if c.get("n_seeds", 1) > 1 else None) for m in METRICS]
        if meth == "triad_cq":
            lines.append("\\midrule")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(TAB, fname or f"main_{fam}.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return cells


ABL_METRICS = ["auc_C", "auc_R", "auc_agree"]


def table_ablation(df, fam="lr", level="task", datasets=("mphrc", "sensecobot", "mphrc_all")):
    meths = ["triad", "triad_noc", "triad_noq", "triad_cq", "triad_nocf", "triad_noalpha", "triad_noprior", "triad_noanchor",
             "triad_student", "triad_eb", "triad_mq"]
    lines = ["\\begin{tabular}{l" + "ccc" * len(datasets) + "}", "\\toprule",
             "Variant & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DSN[d]}}}" for d in datasets) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(datasets))),
             " & " + " & ".join(" & ".join(MNAMES[m] for m in ABL_METRICS) for _ in datasets) + " \\\\", "\\midrule"]
    for meth in meths:
        row = [NAMES.get(meth, meth)]
        any_ = False
        for d in datasets:
            c = agg(df, d, DATASETS[d]["primary"], level, fam, meth)
            if c is None:
                row += ["--"] * 3
            else:
                any_ = True
                row += [fmt(c[m]) for m in ABL_METRICS]
        if any_:
            lines.append(" & ".join(row) + " \\\\")
        if meth == "triad":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "ablation.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def table_alternatives(df, fam="lr", level="task", datasets=("mphrc", "sensecobot", "mphrc_all")):
    """Design alternatives that were explored and rejected (Appendix): same layout as the ablation table."""
    meths = ["triad", "triad_x1", "triad_noanchor", "triad_rcal", "triad_med", "triad_hard", "gtriad_c", "gtriad"]
    lines = ["\\begin{tabular}{l" + "ccc" * len(datasets) + "}", "\\toprule",
             "Variant & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DSN[d]}}}" for d in datasets) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(datasets))),
             " & " + " & ".join(" & ".join(MNAMES[m] for m in ABL_METRICS) for _ in datasets) + " \\\\", "\\midrule"]
    for meth in meths:
        row = [NAMES.get(meth, meth)]
        any_ = False
        for d in datasets:
            c = agg(df, d, DATASETS[d]["primary"], level, fam, meth)
            if c is None:
                row += ["--"] * 3
            else:
                any_ = True
                row += [fmt(c[m]) for m in ABL_METRICS]
        if any_:
            lines.append(" & ".join(row) + " \\\\")
        if meth == "triad":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "alternatives.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def table_regret(df, level="task", datasets=("mphrc", "wesad", "sensecobot"), fams=("lr", "hgb", "rf"),
                 methods=("cond", "report", "twohead", "filter", "ds", "raykar", "triad_cq", "triad")):
    """Mean and maximum regret (best method minus method) over datasets x classifier families x {AUC_C, AUC_R, AUC_agree}."""
    reg = {m: [] for m in methods}
    wins = {m: 0 for m in methods}
    for d in datasets:
        q = DATASETS[d]["primary"]
        for fam in fams:
            cells = {m: agg(df, d, q, level, fam, m) for m in methods}
            cells = {m: c for m, c in cells.items() if c is not None}
            if not cells:
                continue
            for metric in ABL_METRICS:
                vals = {m: c[metric] for m, c in cells.items() if np.isfinite(c.get(metric, np.nan))}
                best = max(vals.values())
                for m, v in vals.items():
                    reg[m].append(best - v)
                    if best - v < 5e-4:
                        wins[m] += 1
    n_cells = max(len(v) for v in reg.values())
    lines = ["\\begin{tabular}{lccc}", "\\toprule", f"Method & Mean regret & Max regret & Best-or-tied (of {n_cells}) \\\\", "\\midrule"]
    order = sorted(methods, key=lambda m: np.mean(reg[m]) if reg[m] else 9)
    for m in order:
        if not reg[m]:
            continue
        lines.append(f"{NAMES.get(m, m)} & {np.mean(reg[m]):.3f} & {np.max(reg[m]):.3f} & {wins[m]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "regret.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return {m: dict(mean=float(np.mean(reg[m])), max=float(np.max(reg[m])), wins=wins[m]) for m in methods if reg[m]}


def table_byproducts(df, level="task", fam="lr"):
    """Manipulation efficacy, label-noise structure and AUC ceilings per dataset.

    The noise structure is taken from a TRIAD fit on all sessions (a description of the dataset, not a prediction);
    the AUCs are the LOSO numbers of the benchmark runs."""
    from triad.evaluation import make_clf
    from triad.model import TRIAD
    rows = []
    summary = {}
    for ds in ["mphrc", "wesad", "sensecobot", "mphrc_all"]:
        q = DATASETS[ds]["primary"]
        c = agg(df, ds, q, level, fam, "triad")
        b = agg(df, ds, q, level, fam, "cond")
        r = agg(df, ds, q, level, fam, "report")
        if c is None:
            continue
        d = load_dataset(ds)
        R, D = d.R[q], d.design.astype(float)
        ok = np.isfinite(R) & (D >= 0)
        agree = float((R[ok] == D[ok]).mean())
        m = TRIAD(make_clf("lr"), tau=1.0).fit(d.X, d.cond, d.design, d.Q_for([q]), None, d.subj)
        nR = m.label_noise(R)
        Dn = D.copy()
        nC = m.label_noise(Dn)
        cap_R, cap_C = nR["ceiling"], nC["ceiling"]
        den = denoised_auc(c["auc_R"], kappa=nR["kappa"])
        rows.append((DSN[ds], q.replace("q_", "").replace("nasaAB", "NASA-TLX").upper(), len(d.subj), agree, nC["eps"], nR["e0"], nR["e1"], nR["eps"], cap_R, c["auc_R"], b["auc_R"] if b else np.nan, c["auc_C"]))
        summary[ds] = dict(q=q, n_sessions=int(len(d.subj)), n_subjects=int(len(np.unique(d.subj))), agreement=agree, noise_R=nR, noise_C=nC,
                           ceiling_R=cap_R, ceiling_C=cap_C, auc_R=c["auc_R"], auc_C=c["auc_C"], auc_agree=c["auc_agree"], denoised=den,
                           pi={str(k): float(v) for k, v in m.pi_.items()}, beta=float(m.beta_[0]), sigma=float(m.sigma_[0]),
                           alpha_sd=float(np.std(m.alpha_[:, 0])), cond_auc_R=b["auc_R"] if b else None, report_auc_C=r["auc_C"] if r else None,
                           report_auc_R=r["auc_R"] if r else None, cond_auc_C=b["auc_C"] if b else None, eps_C=nC["eps"], eps_R=nR["eps"])
    lines = ["\\begin{tabular}{llrrrrrrrrrr}", "\\toprule",
             "Dataset & Report & $n$ & Agree. & $\\hat\\varepsilon_C$ & $\\hat e_0^R$ & $\\hat e_1^R$ & $\\hat\\varepsilon_R$ & Ceiling$_R$ & AUC$_R$ TRIAD & AUC$_R$ B1 & AUC$_C$ TRIAD \\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]:.2f} & {r[4]:.2f} & {r[5]:.2f} & {r[6]:.2f} & {r[7]:.2f} & {r[8]:.2f} & {r[9]:.3f} & {r[10]:.3f} & {r[11]:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "byproducts.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return summary


def table_pi(summary):
    """Manipulation efficacy per condition (one small table per dataset)."""
    lines = ["\\begin{tabular}{llc}", "\\toprule", "Dataset & Condition & $\\hat\\pi_c$ \\\\", "\\midrule"]
    for ds in ["mphrc", "wesad", "sensecobot", "mphrc_all"]:
        if ds not in summary:
            continue
        pi = summary[ds]["pi"]
        for k in sorted(pi, key=lambda kk: -pi[kk]):
            if ds == "mphrc_all" and k not in ("manual-task", "cobot-task"):
                continue
            lines.append(f"{DSN[ds]} & {k} & {pi[k]:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "pi.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def table_case_day2(df, level="task"):
    q = "q_stai"
    lines = ["\\begin{tabular}{llcccc}", "\\toprule", "Classifier & Method & " + " & ".join(MNAMES[m] for m in METRICS) + " \\\\", "\\midrule"]
    for fam in ["lr", "hgb", "rf"]:
        for meth in ["cond", "report", "filter", "twohead", "triad_cq", "triad"]:
            c = agg(df, "mphrc_all", q, level, fam, meth)
            if c is None:
                continue
            lines.append(f"{fam.upper()} & {NAMES.get(meth, meth)} & " + " & ".join(fmt(c[m]) for m in METRICS) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "case_day2.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


SEMI_METHS = ["cond", "report", "twohead", "filter", "ds", "raykar", "triad_cq", "triad"]
SHORT = {"cond": "B1 Cond.", "report": "B2 Report", "twohead": "B4 Two-head", "filter": "B6 Filter", "ds": "B7 D--S", "raykar": "B8 Raykar",
         "triad_cq": "TRIAD $\\tau{=}0$", "triad": "\\textbf{TRIAD}"}


def _semi_rows(mode):
    files = glob.glob(os.path.join(ROOT, "results", f"semi_synthetic_{mode}_*_lr_task.json"))
    rows = []
    for f in files:
        rows += json.load(open(f))
    return pd.DataFrame(rows) if rows else None


def _semi_table(sd, keys, key_names, fname):
    meths = [m for m in SEMI_METHS if m in set(sd.method)]
    piv = sd.groupby(keys + ["method"])["auc_S"].agg(["mean", "std"]).reset_index()
    lines = ["\\begin{tabular}{" + "c" * len(keys) + "c" * len(meths) + "}", "\\toprule",
             " & ".join(key_names) + " & " + " & ".join(SHORT.get(m, m) for m in meths) + " \\\\", "\\midrule"]
    for kv, g in piv.groupby(keys):
        kv = kv if isinstance(kv, tuple) else (kv,)
        vals = {m: g[g.method == m]["mean"].values[0] for m in meths if (g.method == m).any()}
        best = max(vals, key=vals.get)
        cells = []
        for m in meths:
            if m in vals:
                s = f"{vals[m]:.3f}"
                cells.append(f"\\textbf{{{s}}}" if m == best else s)
            else:
                cells.append("--")
        lines.append(" & ".join(f"{v:.2f}" for v in kv) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, fname), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def table_semi():
    su = _semi_rows("uniform")
    ss = _semi_rows("structured")
    out = {}
    if su is not None:
        _semi_table(su, ["eps_C", "eps_R"], ["$\\varepsilon_C$", "$\\varepsilon_R$"], "semi_uniform.tex")
        out["uniform"] = su
        out["rec_uniform"] = su[su.method == "triad"].groupby(["eps_C", "eps_R"])[["eps_C_emp", "eps_R_emp", "eps_C_hat", "eps_R_hat"]].mean().reset_index()
    if ss is not None:
        _semi_table(ss, ["pi_fail", "eps_R"], ["$\\pi_{\\text{fail}}$", "$\\varepsilon_R$"], "semi_structured.tex")
        out["structured"] = ss
        out["rec_structured"] = ss[ss.method == "triad"].groupby(["pi_fail", "eps_R"])[["eps_C_emp", "eps_R_emp", "eps_C_hat", "eps_R_hat"]].mean().reset_index()
        # recovered vs realised manipulation efficacy per synthetic condition.  The realised P(S=1 | condition) differs
        # from the nominal pi_k of the generator (conditions are drawn given S, whose base rate the teacher sets), so
        # the synthetic data are regenerated from the stored seeds to obtain the quantity TRIAD actually estimates.
        sys.path.insert(0, os.path.dirname(__file__))
        from semi_synthetic import make_synthetic
        d_syn = load_dataset("mphrc")
        recs = []
        for (pf, eR), g in ss[ss.method == "triad"].groupby(["pi_fail", "eps_R"]):
            pis = pd.DataFrame(list(g.pi_hat)).mean()
            trues = []
            for rep in sorted(g.rep.unique()):
                seed = 1000 * int(rep) + int(10 * eR) + 7 + int(100 * pf)
                _, _, info = make_synthetic(d_syn, np.nan, eR, np.random.default_rng(seed), mode="structured", pi_fail=pf)
                trues.append(info["pi_true"])
            true = pd.DataFrame(trues).mean()
            recs.append((pf, eR, pis, true))
        conds = ["stressA", "stressB", "stressFail", "calmA", "calmB", "calmC"]
        lines = ["\\begin{tabular}{ccl" + "c" * len(conds) + "}", "\\toprule",
                 "$\\pi_{\\text{fail}}$ & $\\varepsilon_R$ & & " + " & ".join(c.replace("stress", "stressor ").replace("calm", "calm ").replace("Fail", "(failed)") for c in conds) + " \\\\",
                 "\\midrule", "\\multicolumn{3}{l}{nominal $\\pi_k$ of the generator} & " + " & ".join({"stressA": "0.90", "stressB": "0.85", "stressFail": "$\\pi_{\\text{fail}}$", "calmA": "0.10", "calmB": "0.12", "calmC": "0.15"}[c] for c in conds) + " \\\\", "\\midrule"]
        for pf, eR, pis, true in recs:
            head = "\\multirow{2}{*}{%.2f} & \\multirow{2}{*}{%.2f} & realised $P(S{=}1\\mid c)$ & " % (pf, eR)
            lines.append(head + " & ".join(f"{true.get(c, np.nan):.2f}" for c in conds) + " \\\\")
            lines.append(f" & & recovered $\\hat\\pi_c$ & " + " & ".join(f"{pis.get(c, np.nan):.2f}" for c in conds) + " \\\\")
            lines.append("\\addlinespace[2pt]")
        lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(TAB, "semi_pi.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        out["pi_structured"] = recs
        out["pi_abs_err"] = float(np.mean([np.mean([abs(pis.get(c, np.nan) - true.get(c, np.nan)) for c in conds]) for _, _, pis, true in recs]))
        out["pi_fail_err"] = float(np.mean([abs(pis.get("stressFail", np.nan) - true.get("stressFail", np.nan)) for _, _, pis, true in recs]))
    return out or None


def significance(df, level="task", fam="lr", B=2000):
    """Bootstrap TRIAD minus each baseline on AUC_agree / AUC_R / AUC_C (subject-level, paired)."""
    lines = ["\\begin{tabular}{llccc}", "\\toprule", "Dataset & Baseline & $\\Delta$AUC$_\\cap$ [95\\% CI] & $\\Delta$AUC$_R$ [95\\% CI] & $\\Delta$AUC$_C$ [95\\% CI] \\\\", "\\midrule"]
    out = {}
    for ds in ["mphrc", "wesad", "sensecobot", "mphrc_all"]:
        q = DATASETS[ds]["primary"]
        d = load_dataset(ds)
        def scores(meth):
            f = glob.glob(os.path.join(RUNS, ds, q, level, fam, f"{meth}_s0.npz"))
            if not f:
                return None
            z = np.load(f[0], allow_pickle=True)
            assert (z["session"] == d.session).all()
            return z["scores"]
        st = scores("triad")
        if st is None:
            continue
        for meth in ["cond", "report", "filter", "twohead", "raykar"]:
            sb = scores(meth)
            if sb is None:
                continue
            cells = []
            for metric in ["auc_agree", "auc_R", "auc_C"]:
                m, lo, hi, p = bootstrap_diff(st, sb, d, q, metric=metric, B=B, seed=0)
                star = "$^{*}$" if (np.isfinite(p) and p < 0.05) else ""
                cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
                out[(ds, meth, metric)] = dict(diff=m, lo=lo, hi=hi, p=p)
            lines.append(f"{DSN[ds]} & {NAMES.get(meth, meth)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "significance.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return out


HELDOUT = {"mphrc": ("q_stai", [("q_nasa", "NASA-TLX"), ("q_arousal", "SAM arousal")]),
           "wesad": ("q_stai6", [("q_panas_na", "PANAS neg.\\ affect"), ("q_arousal", "SAM arousal")]),
           "sensecobot": ("q_nasaA", [("q_nasaB", "NASA-TLX B")]),
           "mphrc_all": ("q_stai", [("q_nasa", "NASA-TLX"), ("q_arousal", "SAM arousal")])}
HO_METHS = ["cond", "report", "twohead", "filter", "ds", "raykar", "triad_cq", "triad"]
HO_SHORT = {"cond": "B1", "report": "B2", "twohead": "B4", "filter": "B6", "ds": "B7", "raykar": "B8", "triad_cq": "TRIAD $\\tau{=}0$", "triad": "\\textbf{TRIAD}"}


def _heldout_auc(ds, q_train, q_eval, fam, meth, level="task"):
    d = load_dataset(ds)
    files = sorted(glob.glob(os.path.join(RUNS, ds, q_train, level, fam, f"{meth}_s*.npz")))
    if not files:
        return np.nan, np.nan
    aucs, rhos = [], []
    for f in files:
        z = np.load(f[0] if isinstance(f, list) else f, allow_pickle=True)
        assert (z["session"] == d.session).all()
        s = z["scores"]
        R, Qc = d.R[q_eval], d.Qc[q_eval]
        ok = np.isfinite(R) & np.isfinite(s)
        aucs.append(roc_auc_score(R[ok].astype(int), s[ok]))
        ok = np.isfinite(Qc) & np.isfinite(s)
        rhos.append(spearmanr(Qc[ok], s[ok]).correlation)
    return float(np.mean(aucs)), float(np.mean(rhos))


def significance_heldout(fam="lr", B=2000, baselines=("cond", "report", "twohead", "raykar"), datasets=("mphrc", "wesad", "sensecobot", "mphrc_all")):
    """Participant-level paired bootstrap of TRIAD minus each baseline on the held-out-view AUC (seed-0 scores)."""
    lines = ["\\begin{tabular}{ll" + "c" * len(baselines) + "}", "\\toprule",
             "Dataset & Held-out view & " + " & ".join(f"$\\Delta$ vs {HO_SHORT[b]}" for b in baselines) + " \\\\", "\\midrule"]
    out = {}
    for ds in datasets:
        q_train, views = HELDOUT[ds]
        d = load_dataset(ds)

        def scores(meth):
            f = glob.glob(os.path.join(RUNS, ds, q_train, "task", fam, f"{meth}_s0.npz"))
            if not f:
                return None
            z = np.load(f[0], allow_pickle=True)
            assert (z["session"] == d.session).all()
            return z["scores"]

        st = scores("triad")
        if st is None:
            continue
        for q_eval, vname in views:
            cells = []
            for b in baselines:
                sb = scores(b)
                if sb is None:
                    cells.append("--")
                    continue
                m, lo, hi, p = bootstrap_diff(st, sb, d, q_eval, metric="auc_R", B=B, seed=0)
                star = "$^{*}$" if (np.isfinite(p) and p < 0.05) else ""
                z = lambda x: 0.0 if abs(x) < 5e-4 else x  # avoid printing -0.000
                cells.append(f"{z(m):+.3f} [{z(lo):+.3f}, {z(hi):+.3f}]{star}")
                out[(ds, q_eval, b)] = dict(diff=m, lo=lo, hi=hi, p=p)
            lines.append(f"{DSN[ds]} & {vname} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "heldout_sig.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return {"|".join(k): v for k, v in out.items()}


def table_heldout(fams=("lr", "hgb", "rf"), datasets=("mphrc", "wesad", "sensecobot", "mphrc_all")):
    """AUC against the participant-median split of a questionnaire that no method saw in training (held-out view).

    Every method is trained with the primary questionnaire (methods B1/B6-B8/TRIAD also with the design); the
    held-out questionnaire acts as a third, conditionally independent measurement of the state (Corollary of Prop. 2:
    the method ranking under any such view equals the ranking by AUC against S).
    """
    out = {}
    lines = ["\\begin{tabular}{lll" + "c" * len(HO_METHS) + "}", "\\toprule",
             "Dataset & Held-out view & Clf. & " + " & ".join(HO_SHORT[m] for m in HO_METHS) + " \\\\", "\\midrule"]
    for ds in datasets:
        q_train, views = HELDOUT[ds]
        for q_eval, vname in views:
            first = True
            for fam in fams:
                vals = {m: _heldout_auc(ds, q_train, q_eval, fam, m)[0] for m in HO_METHS}
                if all(not np.isfinite(v) for v in vals.values()):
                    continue
                out[(ds, q_eval, fam)] = vals
                best = max(vals, key=lambda k: vals[k] if np.isfinite(vals[k]) else -1)
                cells = []
                for m in HO_METHS:
                    v = vals[m]
                    if not np.isfinite(v):
                        cells.append("--")
                        continue
                    s = f"{v:.3f}"
                    if abs(v - vals[best]) < 0.0005:
                        s = f"\\textbf{{{s}}}"
                    elif v >= vals[best] - 0.01:
                        s = f"\\underline{{{s}}}"
                    cells.append(s)
                head = (f"{DSN[ds]} & {vname}" if first else " & ")
                lines.append(f"{head} & {fam.upper()} & " + " & ".join(cells) + " \\\\")
                first = False
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "heldout.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # summary counts per method: best / within 0.01 of best / mean AUC, plus the TRIAD family (TRIAD or its tau=0 variant)
    n_cells = len(out)

    def _cnt(pred):
        return sum(1 for v in out.values() if pred(v, max(x for x in v.values() if np.isfinite(x))))

    per_method = {m: dict(n_best=_cnt(lambda v, b, m=m: np.isfinite(v[m]) and v[m] >= b - 0.0005),
                          n_close=_cnt(lambda v, b, m=m: np.isfinite(v[m]) and v[m] >= b - 0.01),
                          mean=float(np.nanmean([v[m] for v in out.values()])),
                          mean_gap=float(np.nanmean([max(x for x in v.values() if np.isfinite(x)) - v[m] for v in out.values()])))
                  for m in HO_METHS}
    fam_close = _cnt(lambda v, b: max(v["triad"], v["triad_cq"]) >= b - 0.01)
    # failed-design regime (day 1+2): mean AUC of each method
    fail = {m: float(np.nanmean([v[m] for k, v in out.items() if k[0] == "mphrc_all"])) for m in HO_METHS}
    return dict(cells={"|".join(k): v for k, v in out.items()}, n_cells=n_cells, n_best=per_method["triad"]["n_best"],
                n_close=per_method["triad"]["n_close"], per_method=per_method, family_close=fam_close, mphrc_all_mean=fail)


# --------------------------------------------------------------------------- figures
def _save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)


def figures(df, summary, semi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

    # F1 agreement + cross-training matrix
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, ds in zip(axes, ["mphrc", "wesad", "sensecobot"]):
        q = DATASETS[ds]["primary"]
        c = agg(df, ds, q, "task", "lr", "cond")
        r = agg(df, ds, q, "task", "lr", "report")
        t = agg(df, ds, q, "task", "lr", "triad")
        M = np.array([[c["auc_C"], c["auc_R"]], [r["auc_C"], r["auc_R"]], [t["auc_C"], t["auc_R"]]])
        im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="Blues")
        for i in range(3):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="black" if M[i, j] < 0.85 else "white")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["eval:\ndesign", "eval:\nreport"])
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["train: design", "train: report", "TRIAD"])
        ax.set_title(f"{DSN[ds]} (agree {summary[ds]['agreement']:.0%})")
    fig.tight_layout(); _save(fig, "fig_cross"); plt.close(fig)

    # F2 manipulation efficacy
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), gridspec_kw=dict(width_ratios=[3, 1.6, 1.6, 0.9]))
    for ax, ds in zip(axes, ["mphrc", "wesad", "sensecobot", "mphrc_all"]):
        pi = summary[ds]["pi"]
        d = load_dataset(ds)
        if ds == "mphrc_all":
            pi = {k: v for k, v in pi.items() if k in ("manual-task", "cobot-task")}
        intent = {c: float(np.mean(d.design[d.cond == c] == 1)) if (d.design[d.cond == c] >= 0).all() else 0.5 for c in pi}
        keys = sorted(pi, key=lambda k: (-intent[k], -pi[k]))
        x = np.arange(len(keys))
        ax.bar(x, [pi[k] for k in keys], color=["#c0504d" if intent[k] > 0.5 else ("#9bbb59" if intent[k] < 0.5 else "#bfbfbf") for k in keys])
        ax.set_xticks(x); ax.set_xticklabels([k.replace("-task", "").replace("stroop", "stroop-") for k in keys], rotation=60, ha="right")
        ax.set_ylim(0, 1); ax.axhline(0.5, ls=":", c="grey", lw=0.8)
        ax.set_title(DSN[ds] if ds != "mphrc_all" else "MPHRC day 2", fontsize=8)
        if ds == "mphrc":
            ax.set_ylabel(r"$\hat\pi_c$")
    fig.tight_layout(); _save(fig, "fig_pi"); plt.close(fig)

    # F3 ceiling
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for ds, mk in zip(["mphrc", "wesad", "sensecobot"], ["o", "s", "^"]):
        s = summary[ds]
        ax.scatter([s["ceiling_R"]], [s["auc_R"]], marker=mk, s=40, label=f"{DSN[ds]}: TRIAD")
        ax.scatter([s["ceiling_R"]], [s["report_auc_R"]], marker=mk, s=40, facecolors="none", edgecolors="grey")
    xs = np.linspace(0.5, 1, 50)
    ax.plot(xs, xs, "k--", lw=0.8, label="ceiling $\\frac{1}{2}+\\frac{1}{2}\\hat\\kappa_R$")
    ax.set_xlabel(r"estimated ceiling for AUC$_R$"); ax.set_ylabel("observed AUC$_R$ (LOSO)")
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0.5, 1.0); ax.legend(fontsize=6, loc="upper left", frameon=False)
    fig.tight_layout(); _save(fig, "fig_ceiling"); plt.close(fig)

    # F4 semi-synthetic: uniform curves, structured bars, recovery of noise rates
    if semi is not None:
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.3, 1.2, 1]))
        styles = [("cond", "s", "C0"), ("filter", "v", "C2"), ("twohead", "^", "C1"), ("raykar", "x", "C4"), ("triad", "o", "C3")]
        lab = {"cond": "B1 Condition", "filter": "B6 Filter", "twohead": "B4 Two-head", "raykar": "B8 Raykar", "triad": "TRIAD"}
        if "uniform" in semi:
            sd = semi["uniform"]
            ax = axes[0]
            for m, mk, col in styles:
                for eR, ls in zip(sorted(sd.eps_R.unique()), ["-", "--"]):
                    g = sd[(sd.method == m) & (sd.eps_R == eR)].groupby("eps_C")["auc_S"].mean()
                    ax.plot(g.index, g.values, marker=mk, ls=ls, ms=4, lw=1, color=col, label=lab[m] if ls == "-" else None)
            ax.plot([], [], "k-", lw=1, label=r"$\varepsilon_R{=}0.20$"); ax.plot([], [], "k--", lw=1, label=r"$\varepsilon_R{=}0.35$")
            ax.set_xlabel(r"design-label noise $\varepsilon_C$ (uniform)"); ax.set_ylabel("AUC against latent state $S$")
            ax.margins(x=0.09)  # keep the markers at the smallest noise level off the y spine
            ax.set_title("(a) uniform design noise", fontsize=8)
        if "structured" in semi:
            ss = semi["structured"]
            ax = axes[1]
            cells = [(pf, eR) for pf in sorted(ss.pi_fail.unique(), reverse=True) for eR in sorted(ss.eps_R.unique())]
            x = np.arange(len(cells)); w = 0.16
            for k, (m, mk, col) in enumerate(styles):
                vals = [ss[(ss.method == m) & (ss.pi_fail == pf) & (ss.eps_R == eR)]["auc_S"].mean() for pf, eR in cells]
                ax.bar(x + (k - 2) * w, vals, w, color=col, label=lab[m])
            ax.set_xticks(x); ax.set_xticklabels([f"$\\pi_f{{=}}{pf}$\n$\\varepsilon_R{{=}}{eR}$" for pf, eR in cells], fontsize=6)
            ax.set_xlim(-0.5, len(cells) - 0.5)
            ax.set_ylim(0.7, 0.9); ax.set_ylabel("AUC against $S$")
            ax.set_title("(b) one designed stressor fails", fontsize=8)
        ax = axes[2]
        for key, mk in [("rec_uniform", "o"), ("rec_structured", "s")]:
            if key in semi:
                rec = semi[key]
                ax.scatter(rec["eps_C_emp"], rec["eps_C_hat"], marker=mk, color="C0", label=r"$\varepsilon_C$" + (" (uniform)" if "uniform" in key else " (structured)"), s=18)
                ax.scatter(rec["eps_R_emp"], rec["eps_R_hat"], marker=mk, color="C3", label=r"$\varepsilon_R$" + (" (uniform)" if "uniform" in key else " (structured)"), s=18)
        ax.plot([0, 0.45], [0, 0.45], "k--", lw=0.8)
        ax.set_xlabel("true flip rate"); ax.set_ylabel("TRIAD estimate")
        ax.set_xlim(-0.01, 0.47); ax.set_ylim(-0.01, 0.47)
        ax.legend(fontsize=5.5, frameon=False, loc="lower right", borderaxespad=0.2, handletextpad=0.3)
        ax.set_title("(c) recovered noise rates", fontsize=8)
        # one shared method legend under panels (a)-(b): keeps every axes free of text over data
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.40, 0.005), ncol=4, fontsize=6.5,
                   frameon=False, handlelength=1.8, columnspacing=1.3, handletextpad=0.4)
        fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.30, wspace=0.32)
        _save(fig, "fig_semi"); plt.close(fig)

    # F5 sensitivity: prior strength a, temperature tau, number of physiological sweeps
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75))
    q = DATASETS["mphrc"]["primary"]
    b = agg(df, "mphrc", q, "task", "lr", "cond")
    ax = axes[0]
    pts = []
    for meth, a in [("triad_noprior", 0), ("triad_a5", 5), ("triad_a10", 10), ("triad", 20), ("triad_a50", 50), ("triad_a100", 100)]:
        c = agg(df, "mphrc", q, "task", "lr", meth)
        if c:
            pts.append((a, c["auc_C"], c["auc_R"], c["auc_agree"]))
    if pts:
        pts = np.array(pts)
        # evenly spaced categorical positions: a symlog axis crowds 5/10/20 and 50/100 into each other
        xp = np.arange(len(pts))
        ax.plot(xp, pts[:, 1], "o-", ms=4, label="AUC$_C$"); ax.plot(xp, pts[:, 3], "s-", ms=4, label="AUC$_\\cap$"); ax.plot(xp, pts[:, 2], "^-", ms=4, label="AUC$_R$")
        ax.axhline(b["auc_C"], ls=":", c="C0", lw=0.8); ax.axhline(b["auc_agree"], ls=":", c="C1", lw=0.8); ax.axhline(b["auc_R"], ls=":", c="C2", lw=0.8)
        ax.set_xlabel("design-prior strength $a$"); ax.set_ylabel("AUC (LOSO, MPHRC)")
        ax.set_xticks(xp); ax.set_xticklabels([f"{int(a)}" for a in pts[:, 0]])
        ax.set_xlim(-0.35, len(pts) - 0.65)
        lo = min(pts[:, 1:].min(), b["auc_R"]); hi = max(pts[:, 1:].max(), b["auc_C"])
        ax.set_ylim(lo - 0.035, hi + 0.025)  # keep the baseline guides off the bottom spine
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3, fontsize=7,
                  frameon=False, handlelength=1.8, columnspacing=1.2, handletextpad=0.4)
        ax.set_title("(a) prior strength", fontsize=8)
    ax = axes[1]
    pts = []
    for meth, t in [("triad_cq", 0.0), ("triad_t0.5", 0.5), ("triad", 1.0), ("triad_t2.0", 2.0)]:
        c = agg(df, "mphrc", q, "task", "lr", meth)
        if c:
            pts.append((t, c["auc_C"], c["auc_R"], c["auc_agree"]))
    if pts:
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], "o-", ms=4); ax.plot(pts[:, 0], pts[:, 3], "s-", ms=4); ax.plot(pts[:, 0], pts[:, 2], "^-", ms=4)
        ax.axhline(b["auc_C"], ls=":", c="C0", lw=0.8); ax.axhline(b["auc_agree"], ls=":", c="C1", lw=0.8); ax.axhline(b["auc_R"], ls=":", c="C2", lw=0.8)
        ax.set_xticks(list(pts[:, 0])); ax.set_xticklabels([("%g" % t) for t in pts[:, 0]])
        ax.set_xlim(pts[:, 0].min() - 0.12, pts[:, 0].max() + 0.12)
        ax.set_ylim(axes[0].get_ylim())  # same scale as (a), so the two panels can be read together
        ax.set_xlabel(r"physiology weight $\tau$"); ax.set_title(r"(b) temperature $\tau$", fontsize=8)
    ax = axes[2]
    NX = [0, 1, 2, 8, 16]
    short = {"mphrc": "MPHRC", "mphrc_all": "MPHRC d1+2", "sensecobot": "SenseCobot"}
    for ds, col, mk in [("mphrc", "C0", "o"), ("mphrc_all", "C3", "s"), ("sensecobot", "C2", "^")]:
        qd = DATASETS[ds]["primary"]
        pts = []
        for meth, nx in [("triad_cq", 0), ("triad_x1", 1), ("triad_x2", 2), ("triad_x8", 8), ("triad_x16", 16)]:
            c = agg(df, ds, qd, "task", "lr", meth)
            if c:
                pts.append((nx, c["auc_agree"]))
        if pts:
            pts = np.array(pts)
            # same categorical trick as panel (a): the sweep budgets 0,1,2,8,16 are shown evenly spaced
            ax.plot(np.interp(pts[:, 0], NX, np.arange(len(NX))), pts[:, 1],
                    marker=mk, color=col, lw=1, ms=4, label=short[ds])
            c = agg(df, ds, qd, "task", "lr", "triad")
            nx_used = c.get("triad", {}).get("n_x") if isinstance(c.get("triad"), dict) else None
            if nx_used is not None and np.isfinite(nx_used):
                ax.scatter([np.interp(nx_used, NX, np.arange(len(NX)))], [c["auc_agree"]],
                           marker="*", s=90, color=col, zorder=5)
    ax.set_xticks(np.arange(len(NX))); ax.set_xticklabels([str(n) for n in NX])
    ax.set_xlim(-0.35, len(NX) - 0.65)
    ax.set_xlabel("physiological EM sweeps (fixed)"); ax.set_ylabel("AUC$_\\cap$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, fontsize=7,
              frameon=False, handlelength=1.8, handletextpad=0.4, columnspacing=1.0, labelspacing=0.35)
    ax.set_title("(c) sweeps; $\\star$ = anchored stop", fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.30, wspace=0.28)
    _save(fig, "fig_sensitivity"); plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sig", action="store_true", help="reuse the existing significance table (slow bootstrap)")
    args = ap.parse_args()
    df = load_runs()
    print("runs:", len(df))
    for fam in ["lr", "hgb", "rf"]:
        if (df.family == fam).any():
            table_main(df, fam, methods=ORDER if fam == "lr" else ["cond", "report", "twohead", "filter", "ds", "raykar", "triad_cq", "triad"])
    table_ablation(df)
    table_alternatives(df)
    regret = table_regret(df)
    summary = table_byproducts(df)
    table_pi(summary)
    table_case_day2(df)
    semi = table_semi()
    heldout = table_heldout()
    print("held-out views: TRIAD best in", heldout["n_best"], "/ within 0.01 in", heldout["n_close"], "of", heldout["n_cells"], "cells;",
          "TRIAD family within 0.01 in", heldout["family_close"])
    for m, v in heldout["per_method"].items():
        print(f"   {m:10s} best {v['n_best']:2d} close {v['n_close']:2d} mean {v['mean']:.3f} gap {v['mean_gap']:.3f} day1+2 {heldout['mphrc_all_mean'][m]:.3f}")
    prev = json.load(open(os.path.join(ROOT, "results", "summary.json"))) if os.path.exists(os.path.join(ROOT, "results", "summary.json")) else {}
    if args.skip_sig and "significance" in prev:
        sig = {tuple(k.split("|")): v for k, v in prev["significance"].items()}
    else:
        sig = significance(df)
    if args.skip_sig and "heldout_significance" in prev:
        ho_sig = prev["heldout_significance"]
    else:
        ho_sig = significance_heldout()
        for k, v in ho_sig.items():
            print(f"   held-out bootstrap {k:30s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] p={v['p']:.3f}")
    if (df.level == "window").any():
        # deployment (window-level) table for LR
        wd = ["mphrc", "wesad", "sensecobot"]
        wm = ["win_auc_C", "win_auc_R", "win_auc_agree"]
        lines = ["\\begin{tabular}{l" + "ccc" * len(wd) + "}", "\\toprule", "Method & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DSN[d]}}}" for d in wd) + " \\\\",
                 " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(wd))),
                 " & " + " & ".join(" & ".join(MNAMES[m] for m in wm) for _ in wd) + " \\\\", "\\midrule"]
        cells = {(d, meth): agg(df, d, DATASETS[d]["primary"], "window", "lr", meth) for d in wd for meth in ["cond", "report", "filter", "twohead", "ds", "raykar", "triad_cq", "triad"]}
        for meth in ["cond", "report", "filter", "twohead", "ds", "raykar", "triad_cq", "triad"]:
            row = [NAMES.get(meth, meth)]
            for d in wd:
                c = cells[(d, meth)]
                for m in wm:
                    if c is None or not np.isfinite(c.get(m, np.nan)):
                        row.append("--")
                    else:
                        best = max(cc.get(m, np.nan) for (dd, mm), cc in cells.items() if dd == d and cc is not None and np.isfinite(cc.get(m, np.nan)))
                        row.append(fmt(c[m], abs(c[m] - best) < 5e-4))
            if meth == "triad_cq":
                lines.append("\\midrule")
            lines.append(" & ".join(row) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(TAB, "window.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    figures(df, summary, semi)
    extra = {}
    if semi:
        extra["semi_pi_abs_err"] = semi.get("pi_abs_err")
        extra["semi_pi_fail_err"] = semi.get("pi_fail_err")
    json.dump(dict(summary=summary, regret=regret, heldout=heldout, heldout_significance=ho_sig, **extra,
                   significance={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sig.items()}),
              open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1, default=float)
    print("tables/figures written to", TAB, FIG)

