"""Journal experiment figures for Elsevier NC/ESWA and IEEE Sensors.

Numbers come from results/summary.json, the semi-synthetic JSON, cached runs,
and the same cells already printed in paper/tables/journal. Nothing is invented.
Output goes only to the two journal figure folders. CHI figures are not touched.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIRS = [
    os.path.join(ROOT, "paper", "elsevier", "figures"),
]
SUMMARY = os.path.join(ROOT, "results", "summary.json")
SEMI_U = os.path.join(ROOT, "results", "semi_synthetic_uniform_mphrc_lr_task.json")
SEMI_S = os.path.join(ROOT, "results", "semi_synthetic_structured_mphrc_lr_task.json")

# Okabe-Ito, colour-blind safe. TRIAD is the darkest bar.
C = {
    "cond": "#D55E00",
    "report": "#0072B2",
    "twohead": "#CC79A7",
    "filter": "#009E73",
    "raykar": "#E69F00",
    "triad": "#1B4F72",
    "calm": "#009E73",
    "stress": "#D55E00",
    "none": "#7F8C8D",
    "grid": "#E5E5E5",
}

LAB = {
    "cond": "Condition only",
    "report": "Self-report only",
    "twohead": "Two-head",
    "filter": "Agreement filter",
    "raykar": "Raykar",
    "triad": "TRIAD",
}

# Cells already in paper/tables/journal (logistic regression).
CLEAN = {
    "MultiPhysio-HRC": {
        "cond": (0.868, 0.610),
        "report": (0.700, 0.624),
        "twohead": (0.880, 0.636),
        "filter": (0.876, 0.632),
        "triad": (0.868, 0.629),
    },
    "WESAD": {
        "cond": (1.00, 0.824),
        "report": (0.998, 0.718),
        "twohead": (1.00, 0.781),
        "filter": (1.00, 0.814),
        "triad": (1.00, 0.825),
    },
    "SenseCobot": {
        "cond": (0.858, 0.733),
        "report": (0.780, 0.611),
        "twohead": (0.848, 0.687),
        "filter": (0.796, 0.694),
        "triad": (0.855, 0.715),
    },
}

FAILED = {
    "cond": (0.90, 0.50),
    "report": (0.56, 0.55),
    "twohead": (0.88, 0.52),
    "filter": (0.90, 0.50),
    "triad": (0.70, 0.61),
}

HELDOUT_CLEAN = [
    ("MPHRC / NASA-TLX", {"cond": 0.676, "report": 0.673, "twohead": 0.708, "raykar": 0.676, "triad": 0.701}),
    ("MPHRC / SAM arousal", {"cond": 0.750, "report": 0.675, "twohead": 0.757, "raykar": 0.749, "triad": 0.773}),
    ("WESAD / PANAS", {"cond": 0.868, "report": 0.797, "twohead": 0.846, "raykar": 0.868, "triad": 0.868}),
    ("WESAD / SAM arousal", {"cond": 0.818, "report": 0.811, "twohead": 0.813, "raykar": 0.818, "triad": 0.819}),
    ("SenseCobot / physical", {"cond": 0.725, "report": 0.667, "twohead": 0.765, "raykar": 0.724, "triad": 0.763}),
]
HELDOUT_FAIL = [
    ("MPHRC full / NASA-TLX", {"cond": 0.542, "report": 0.616, "twohead": 0.589, "raykar": 0.542, "triad": 0.661}),
    ("MPHRC full / SAM arousal", {"cond": 0.596, "report": 0.638, "twohead": 0.642, "raykar": 0.597, "triad": 0.709}),
]

ABLATION = [
    ("TRIAD", 0.63, 0.61, 0.70),
    ("No condition", 0.62, 0.56, 0.61),
    ("No report", 0.62, 0.46, 0.89),
    ("No physiology", 0.63, 0.61, 0.69),
    ("No validity gate", 0.64, 0.60, 0.63),
    ("No design prior", 0.63, 0.61, 0.60),
]

PI_LABEL = {
    "rest": "Rest",
    "meditation": "Meditation",
    "stroopeasy": "Stroop easy",
    "stroophard": "Stroop hard",
    "n-back": "N-back",
    "mat": "MAT",
    "hanoi": "Hanoi",
    "vr-plank": "VR plank",
    "vr-job-sim": "VR job",
    "base": "Baseline",
    "fun": "Amusement",
    "medi1": "Meditation 1",
    "medi2": "Meditation 2",
    "tsst": "TSST",
    "level1": "Level 1",
    "level2": "Level 2",
    "level3": "Level 3",
    "level4": "Level 4",
    "level5": "Level 5",
    "manual-task": "Manual work",
    "cobot-task": "Cobot work",
}

CALM = {
    "rest", "meditation", "base", "fun", "medi1", "medi2", "level1", "level2",
}
STRESS = {
    "stroopeasy", "stroophard", "n-back", "mat", "hanoi", "vr-plank", "vr-job-sim",
    "tsst", "level4", "level5", "manual-task", "cobot-task",
}


def _style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _save(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight", pad_inches=0.04)
        fig.savefig(os.path.join(d, name + ".png"), bbox_inches="tight", pad_inches=0.04, dpi=300)
    print("wrote", name)


def _load_summary():
    with open(SUMMARY, encoding="utf-8") as f:
        return json.load(f)["summary"]


def fig_cross(plt):
    """Two-column: train-by-eval matrix on the three clean protocols."""
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.15))
    rows = ["Train: design", "Train: report", "TRIAD"]
    cols = ["Design", "Report"]
    keys = ["cond", "report", "triad"]
    titles = ["MultiPhysio-HRC", "WESAD", "SenseCobot"]
    disagree = ["36% disagree", "25% disagree", "13% disagree"]
    for ax, title, note in zip(axes, titles, disagree):
        block = CLEAN[title]
        M = np.array([block[k] for k in keys], dtype=float)
        im = ax.imshow(M, vmin=0.50, vmax=1.00, cmap="YlGnBu")
        for i in range(3):
            for j in range(2):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="#111111" if v < 0.86 else "white", fontsize=8, fontweight="medium")
        ax.set_xticks([0, 1], cols)
        ax.set_yticks([0, 1, 2], rows if ax is axes[0] else ["", "", ""])
        ax.set_title(f"{title}  ({note})", pad=4, fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([0.93, 0.20, 0.012, 0.58])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("AUC")
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.91, top=0.86, bottom=0.14, wspace=0.14)
    _save(fig, "fig_cross")
    plt.close(fig)


def _heatmap(ax, matrix, row_labels, col_labels, vmin=0.50, vmax=1.00):
    """Annotated heatmap. Dark cells get white type; nothing sits on a border."""
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="YlGnBu", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="#111111" if v < 0.86 else "white", fontsize=8, fontweight="medium")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(matrix.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def fig_failed(plt):
    """Single-column: failed industrial protocol as horizontal bars."""
    fig, ax = plt.subplots(figsize=(3.40, 3.15))
    methods = ["cond", "report", "twohead", "filter", "triad"]
    short = ["Condition", "Self-report", "Two-head", "Filter", "TRIAD"]
    y = np.arange(len(methods))
    h = 0.36
    d = np.array([FAILED[m][0] for m in methods])
    r = np.array([FAILED[m][1] for m in methods])
    ax.barh(y + h / 2, d, h, color=C["cond"], label="Design AUC", zorder=3)
    ax.barh(y - h / 2, r, h, color=C["report"], label="Report AUC", zorder=3)
    ax.axvline(0.50, color="#888888", ls="--", lw=0.9, zorder=2)
    ax.set_xlim(0.0, 1.18)
    ax.set_xlabel("Leave-one-subject-out AUC")
    ax.set_yticks(y, short)
    ax.xaxis.grid(True, color=C["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for yi, vd, vr in zip(y, d, r):
        ax.text(vd + 0.03, yi + h / 2, f"{vd:.2f}", va="center", ha="left",
                fontsize=7.2, color=C["cond"])
        ax.text(vr + 0.03, yi - h / 2, f"{vr:.2f}", va="center", ha="left",
                fontsize=7.2, color=C["report"])
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=C["cond"], label="Design AUC"),
            plt.Rectangle((0, 0), 1, 1, color=C["report"], label="Report AUC"),
            Line2D([0], [0], color="#888888", ls="--", lw=0.9, label="Chance"),
        ],
        loc="upper center", bbox_to_anchor=(0.42, 1.18), ncol=3, frameon=False, fontsize=7.0,
    )
    fig.subplots_adjust(left=0.28, right=0.97, top=0.84, bottom=0.14)
    _save(fig, "fig_failed")
    plt.close(fig)


def fig_heldout(plt):
    """Two-column: questionnaires no method trained on, as annotated heatmaps."""
    methods = ["cond", "report", "twohead", "raykar", "triad"]
    rows = [LAB[m] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25),
                             gridspec_kw=dict(width_ratios=[1.55, 0.78]))
    Mc = np.array([[row[1][m] for row in HELDOUT_CLEAN] for m in methods])
    Mf = np.array([[row[1][m] for row in HELDOUT_FAIL] for m in methods])
    im = _heatmap(axes[0], Mc, rows, ["MPHRC\nNASA-TLX", "MPHRC\nSAM", "WESAD\nPANAS",
                                      "WESAD\nSAM", "SenseCobot\nphysical"])
    axes[0].set_title("(a) Clean protocols", loc="left", pad=6)
    _heatmap(axes[1], Mf, [""] * 5, ["MPHRC full\nNASA-TLX", "MPHRC full\nSAM"])
    axes[1].set_title("(b) Failed industrial protocol", loc="left", pad=6)
    cax = fig.add_axes([0.92, 0.22, 0.014, 0.58])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Held-out AUC")
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.90, top=0.84, bottom=0.18, wspace=0.14)
    _save(fig, "fig_heldout")
    plt.close(fig)


def fig_pi(plt, summary):
    """Two-column: recovered efficacy. Horizontal bars so names never collide."""
    panels = [
        ("(a) MultiPhysio-HRC, day 1", summary["mphrc"]["pi"],
         ["rest", "meditation", "stroopeasy", "stroophard", "n-back", "mat", "hanoi", "vr-plank", "vr-job-sim"]),
        ("(b) WESAD", summary["wesad"]["pi"],
         ["base", "fun", "medi1", "medi2", "tsst"]),
        ("(c) SenseCobot", summary["sensecobot"]["pi"],
         ["level1", "level2", "level3", "level4", "level5"]),
        ("(d) Industrial tasks on the full protocol", summary["mphrc_all"]["pi"],
         ["rest", "meditation", "manual-task", "cobot-task"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35))
    for ax, (title, pi, keys) in zip(axes.ravel(), panels):
        keys = [k for k in keys if k in pi]
        y = np.arange(len(keys))
        vals = [pi[k] for k in keys]
        colors = [C["stress"] if k in STRESS else (C["calm"] if k in CALM else C["none"]) for k in keys]
        ax.barh(y, vals, color=colors, height=0.68, zorder=3)
        ax.set_yticks(y, [PI_LABEL.get(k, k) for k in keys])
        ax.set_xlim(0, 1.18)
        ax.axvline(0.50, color="#888888", ls="--", lw=0.8, zorder=2)
        ax.set_title(title, loc="left", pad=4)
        ax.set_xlabel(r"Recovered efficacy  $\hat\pi_c$")
        ax.xaxis.grid(True, color=C["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for yi, v in zip(y, vals):
            ax.text(v + 0.03, yi, f"{v:.2f}", va="center", ha="left", fontsize=7.5)
        ax.invert_yaxis()
    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor=C["stress"], label="Designed stressor"),
            Patch(facecolor=C["calm"], label="Designed calm"),
            Patch(facecolor=C["none"], label="No declared intent"),
        ],
        loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03),
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.07, wspace=0.36, hspace=0.32)
    _save(fig, "fig_pi")
    plt.close(fig)


def fig_ceiling(plt, summary):
    """Two-column: observed report AUC versus the identified ceiling."""
    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    items = [
        ("MPHRC", summary["mphrc"]),
        ("WESAD", summary["wesad"]),
        ("SenseCobot", summary["sensecobot"]),
    ]
    y = np.arange(len(items))
    for i, (name, s) in enumerate(items):
        ceil = s["ceiling_R"]
        tri = s["auc_R"]
        rep = s["report_auc_R"]
        ax.plot([0.50, ceil], [i, i], color="#C8C8C8", lw=5, solid_capstyle="butt", zorder=1)
        ax.plot(ceil, i, marker="|", color="#555555", ms=12, mew=1.6, zorder=3)
        ax.plot(tri, i + 0.16, marker="o", color=C["triad"], ms=8, zorder=4)
        ax.plot(rep, i - 0.16, marker="o", markerfacecolor="white", markeredgecolor=C["report"],
                markeredgewidth=1.5, ms=7.5, zorder=5)
        ax.text(tri, i + 0.38, f"{tri:.2f}", ha="center", va="bottom", fontsize=7.2, color=C["triad"])
        ax.text(rep, i - 0.38, f"{rep:.2f}", ha="center", va="top", fontsize=7.2, color=C["report"])
        ax.text(min(ceil + 0.025, 0.97), i, f"{ceil:.2f}", ha="left", va="center",
                fontsize=7.0, color="#555555")
    ax.set_yticks(y, [it[0] for it in items])
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(-0.70, 2.70)
    ax.set_xlabel("Report AUC")
    ax.xaxis.grid(True, color=C["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=C["triad"], lw=0, label="TRIAD"),
            Line2D([0], [0], marker="o", markerfacecolor="white",
                   markeredgecolor=C["report"], lw=0, label="Report-only"),
            Line2D([0], [0], color="#555555", lw=1.4, label="Ceiling"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=7.2,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.18)
    _save(fig, "fig_ceiling")
    plt.close(fig)


def fig_ablation(plt):
    """Two-column: what each piece is for, as an annotated heatmap."""
    names = [r[0] for r in ABLATION]
    M = np.array([[r[1], r[2], r[3]] for r in ABLATION], dtype=float)
    fig, ax = plt.subplots(figsize=(7.15, 2.05))
    im = _heatmap(ax, M, names, ["Clean, report", "Failed, report", "Failed, design"],
                  vmin=0.45, vmax=0.90)
    cax = fig.add_axes([0.91, 0.16, 0.012, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("AUC")
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.89, top=0.96, bottom=0.14)
    _save(fig, "fig_ablation")
    plt.close(fig)


def fig_semi(plt):
    """Two-column: known latent state. Heatmap sits full-width so labels never collide."""
    uni = pd.DataFrame(json.load(open(SEMI_U, encoding="utf-8")))
    st = pd.DataFrame(json.load(open(SEMI_S, encoding="utf-8")))
    styles = [
        ("cond", "s", C["cond"], "Condition only"),
        ("twohead", "^", C["twohead"], "Two-head"),
        ("filter", "v", C["filter"], "Agreement filter"),
        ("raykar", "D", C["raykar"], "Raykar"),
        ("triad", "o", C["triad"], "TRIAD"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.40))

    ax = axes[0]
    for m, mk, col, lab in styles:
        g = uni[(uni.method == m) & (np.isclose(uni.eps_R, 0.20))].groupby("eps_C")["auc_S"].mean()
        ax.plot(g.index, g.values, marker=mk, color=col, lw=1.3, ms=4.5, label=lab)
    ax.set_xlabel(r"Design flip rate  $\varepsilon_C$")
    ax.set_ylabel("AUC against true $S$")
    ax.set_title(r"(a) Uniform noise, $\varepsilon_R{=}0.20$", loc="left", pad=3, fontsize=8.5)
    ax.set_ylim(0.68, 0.92)
    ax.yaxis.grid(True, color=C["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    rec_ut = uni[uni.method == "triad"].groupby(["eps_C", "eps_R"], as_index=False).mean(numeric_only=True)
    rec_st = st[st.method == "triad"].groupby(["pi_fail", "eps_R"], as_index=False).mean(numeric_only=True)
    ax.scatter(rec_ut["eps_C_emp"], rec_ut["eps_C_hat"], marker="o", color=C["cond"], s=22,
               label=r"$\varepsilon_C$", zorder=3)
    ax.scatter(rec_ut["eps_R_emp"], rec_ut["eps_R_hat"], marker="o", color=C["triad"], s=22,
               label=r"$\varepsilon_R$", zorder=3)
    ax.scatter(rec_st["eps_C_emp"], rec_st["eps_C_hat"], marker="s", color=C["cond"], s=22,
               facecolors="white", edgecolors=C["cond"], linewidths=1.2, zorder=3)
    ax.scatter(rec_st["eps_R_emp"], rec_st["eps_R_hat"], marker="s", color=C["triad"], s=22,
               facecolors="white", edgecolors=C["triad"], linewidths=1.2, zorder=3)
    ax.plot([0, 0.45], [0, 0.45], color="#444444", ls="--", lw=0.8, zorder=2)
    ax.set_xlim(-0.02, 0.47)
    ax.set_ylim(-0.02, 0.47)
    ax.set_xlabel("True flip rate")
    ax.set_ylabel("TRIAD estimate")
    ax.set_title("(b) Recovered flip rates", loc="left", pad=3, fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=6.2)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[2]
    cells = [(0.50, 0.20), (0.50, 0.35), (0.30, 0.20), (0.30, 0.35)]
    M = []
    for m, mk, col, lab in styles:
        row = []
        for pf, eR in cells:
            sub = st[(st.method == m) & np.isclose(st.pi_fail, pf) & np.isclose(st.eps_R, eR)]
            row.append(sub["auc_S"].mean() if len(sub) else np.nan)
        M.append(row)
    _heatmap(ax, np.array(M), [s[3] for s in styles],
             [r"$\pi{=}0.5$", r"$\pi{=}0.5$", r"$\pi{=}0.3$", r"$\pi{=}0.3$"],
             vmin=0.70, vmax=0.88)
    ax.set_xlabel(r"Failed stressor; $\varepsilon_R=0.20,0.35,0.20,0.35$")
    ax.set_title("(c) One designed stressor fails", loc="left", pad=3, fontsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.22, wspace=0.38)
    _save(fig, "fig_semi")
    plt.close(fig)


def _agg_run(ds, method, metric="auc_R"):
    pattern = os.path.join(ROOT, "results", "runs", ds, "*", "task", "lr", f"{method}_s*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    vals = []
    extra = {}
    for f in files:
        r = json.load(open(f, encoding="utf-8"))
        if metric in r:
            vals.append(r[metric])
        if "triad" in r and isinstance(r["triad"], dict):
            extra = r["triad"]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "extra": extra}


def fig_sensitivity(plt):
    """Two-column: same hyperparameters on every corpus."""
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.85))
    prior = [("triad_noprior", 0), ("triad_a5", 5), ("triad_a10", 10),
             ("triad", 20), ("triad_a50", 50), ("triad_a100", 100)]
    tau = [("triad_cq", 0.0), ("triad_t0.5", 0.5), ("triad", 1.0), ("triad_t2.0", 2.0)]
    sweeps = [("triad_cq", 0), ("triad_x1", 1), ("triad_x2", 2), ("triad_x8", 8), ("triad_x16", 16)]

    ax = axes[0]
    xs, yc, yr = [], [], []
    for meth, a in prior:
        c = _agg_run("mphrc", meth, "auc_C")
        r = _agg_run("mphrc", meth, "auc_R")
        if c and r:
            xs.append(str(a))
            yc.append(c["mean"])
            yr.append(r["mean"])
    if xs:
        xp = np.arange(len(xs))
        ax.plot(xp, yc, "o-", color=C["cond"], ms=5, label="Design AUC")
        ax.plot(xp, yr, "s-", color=C["report"], ms=5, label="Report AUC")
        ax.set_xticks(xp, xs)
        ax.set_xlabel("Design-prior strength $a$")
        ax.set_ylabel("AUC on MultiPhysio-HRC")
        ax.set_title("(a) Prior strength", loc="left", pad=4)
        ax.yaxis.grid(True, color=C["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(min(yr) - 0.04, max(yc) + 0.03)

    ax = axes[1]
    xs, yc, yr = [], [], []
    for meth, t in tau:
        c = _agg_run("mphrc", meth, "auc_C")
        r = _agg_run("mphrc", meth, "auc_R")
        if c and r:
            xs.append(f"{t:g}")
            yc.append(c["mean"])
            yr.append(r["mean"])
    if xs:
        ax.plot(np.arange(len(xs)), yc, "o-", color=C["cond"], ms=5)
        ax.plot(np.arange(len(xs)), yr, "s-", color=C["report"], ms=5)
        ax.set_xticks(np.arange(len(xs)), xs)
        ax.set_xlabel(r"Physiology weight $\tau$")
        ax.set_title(r"(b) Physiology weight", loc="left", pad=4)
        ax.yaxis.grid(True, color=C["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        if axes[0].has_data():
            ax.set_ylim(axes[0].get_ylim())

    ax = axes[2]
    for ds, col, mk, lab in [
        ("mphrc", C["triad"], "o", "MPHRC day 1"),
        ("mphrc_all", C["cond"], "s", "MPHRC full"),
        ("sensecobot", C["filter"], "^", "SenseCobot"),
    ]:
        pts = []
        for meth, nx in sweeps:
            r = _agg_run(ds, meth, "auc_R")
            if r:
                pts.append((nx, r["mean"]))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(np.arange(len(xs)), ys, marker=mk, color=col, ms=5, label=lab)
    ax.set_xticks(np.arange(5), ["0", "1", "2", "8", "16"])
    ax.set_xlabel("Physiological sweeps")
    ax.set_ylabel("Report AUC")
    ax.set_title("(c) Sweep budget", loc="left", pad=4)
    ax.yaxis.grid(True, color=C["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=6.8)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.32, -0.02))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.22, wspace=0.36)
    _save(fig, "fig_sensitivity")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    summary = _load_summary()
    fig_cross(plt)
    fig_failed(plt)
    fig_pi(plt, summary)
    fig_ceiling(plt, summary)
    fig_ablation(plt)
    fig_semi(plt)
    print("done")


if __name__ == "__main__":
    main()
