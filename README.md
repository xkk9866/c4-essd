# TRIAD: three-view latent-state inference for wearable stress sensing

Public code: [https://github.com/xkk9866/c4-essd](https://github.com/xkk9866/c4-essd).

TRIAD estimates a latent activation state from three views a wearable study already collects: the designed condition, one primary questionnaire, and physiology. The fit returns a posterior, a deployed detector, a manipulation check per condition, a report-noise rate, and an AUC ceiling. This repository ships the estimator, processed windows, experiment scripts and cached results. Manuscript sources are kept locally and are not part of the public repo.

## What is in here

```
src/triad/
  model.py         TRIAD: three-view latent-state EM (design instrument, person-anchored reports,
                   cross-fitted physiology under the report-validity constraint); label_noise / attenuation /
                   ceiling utilities
  graded.py        GradedTRIAD: continuous-activation variant (appendix D, "design alternatives"; not the final method)
  baselines.py     B1-B8: condition-only, report-only, report regression, two-head, noise-robust MLPs
                   (GCE, SCE, forward, co-teaching), agreement filter, Dawid-Skene, Raykar
  evaluation.py    dataset loading, per-participant normalisation, LOSO driver, metrics, paired bootstrap,
                   make_method() name -> estimator (all variants used in the paper)
  features.py      HRV / EDA / respiration / accelerometer window features (WESAD, SenseCobot)
  data/            builders that turn the raw datasets into data/processed/*_windows.csv
scripts/
  run_experiments.py   LOSO benchmark grid (datasets x questionnaires x classifier families x methods x seeds)
  semi_synthetic.py    semi-synthetic study with a known latent state (uniform / structured design noise)
  make_tables.py       aggregates results/runs and results/semi_synthetic_*.json into paper/tables and paper/figures
                       (incl. the held-out-view table, computed from cached per-session scores)
  make_journal_figures.py  compact Elsevier experiment figures (cross, failed, pi, ceiling, ablation, semi)
  dev_heldout.py       held-out-view AUCs on the console (same computation as the paper table)
  dev_diag.py          one-off LOSO diagnostics for any method name / keyword overrides
  compose_figures.py   upscales the illustrated artwork in paper/figures/art/ to 4K
                       (3840x2160); the generator's own labels are kept, nothing is typeset on top
  run_final.ps1, run_refresh.ps1, run_extra.ps1   the exact driver scripts used for the reported runs
  build_submission.py  packs dist/chi2027_{source,supplementary}.zip; refuses to build if any shipped file
                       carries an author name or a local absolute path, and writes '/' entry names so the
                       archives unpack correctly on macOS and Linux
  make_video.py        the 4:58 anonymised video figure: renders nine 1080p slides from the paper's own
                       figures and results/summary.json, narrates them with a male neural voice (edge-tts),
                       muxes per segment with ffmpeg, and emits a matching .srt
data/processed/        60-s window features + questionnaires for the three datasets (derived, redistributable)
results/runs/          one .json (metrics, TRIAD by-products) and one .npz (per-session scores) per run
results/semi_synthetic_*.json
paper/                 CHI 2027 manuscript (acmart, manuscript+review+anonymous)
paper/elsevier/        two Elsevier submissions that reuse the same numbers, not the same story
  els-cas-templates/   official CAS class; do not edit
  refs.bib             shared bibliography
  figures/             method/pipeline art plus journal experiment figures (fig_cross, fig_failed,
                       fig_heldout, fig_pi, fig_ceiling, fig_ablation, fig_semi, fig_sensitivity)
  nc/                  Neurocomputing paper (method + identifiability)
  eswa/                Expert Systems with Applications paper (expert system + failed design)
paper/ieeeSensorJounal/ IEEE Sensors Journal submission (jsen.sty as shipped)
  IEEE-JSEN-LaTeX-template-202405/  official class; do not edit
  accepted/            5 open-access JSEN exemplars (PDF)
  figures/             method/sensing art plus the same eight journal experiment figures
  main.pdf             compiled manuscript
papers/elsevier/       5 Neurocomputing + 5 ESWA open-access exemplars (PDF)
docs/elsevier必中指南.md  craft playbook distilled from those ten papers
docs/IEEE-Sensors必中指南.md  craft playbook distilled from the five JSEN papers
papers/chi/            five accepted CHI full papers (PDF + extracted text) used as writing exemplars
docs/research_proposal.md
docs/chi_exemplar_analysis.md
docs/CHI必中指南.md
docs/PCS投稿填写指南.md
SUPPLEMENTARY.md
dist/                  CHI upload pack (source ZIP, supplementary ZIP, video figure)
```

## Datasets

| Dataset | Source | Licence | What we use |
|---|---|---|---|
| MultiPhysio-HRC | Bussolan et al., Robotics 14(12):184, 2025 (Zenodo) | research use | released 60-s features, STAI-Y1 / NASA-TLX / SAM; day 1 (`mphrc`) and day 1+2 (`mphrc_all`) |
| WESAD | Schmidt et al., ICMI 2018 | research use | RespiBAN chest signals -> 37 features, STAI-6 / PANAS / SAM |
| SenseCobot | Borghi et al., HRI 2024 (Zenodo 10.5281/zenodo.8363762) | CC-BY 4.0 | Shimmer ECG/GSR, E4, AFFDEX -> 65 features, NASA-TLX items |

Raw recordings are not redistributed. `data/processed/*_windows.csv` are derived window-level features and are
sufficient to reproduce every number in the paper. To rebuild them from the raw archives, point the builders at
your own copies and run `python -m triad.data.<dataset>` (see the module docstrings):

```powershell
$env:TRIAD_MPHRC_DIR      = "<path>\MultiPhysio-HRC\features"
$env:TRIAD_WESAD_DIR      = "<path>\WESAD"
$env:TRIAD_SENSECOBOT_DIR = "<path>\SenseCobot"
```

Unset, they fall back to `data/raw/<corpus>/`. No dataset location is hard-coded, so nothing in the source
identifies the machine it was run on.

## Reproducing the paper

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "src"

# main benchmark (logistic regression, session level; add --families hgb rf --seeds 0 1 2 for the tree ensembles)
python scripts/run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --level task --qs all
# window-level deployment
python scripts/run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --level window --methods cond report filter twohead ds raykar triad triad_cq
# sensitivity rows
python scripts/run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --level task --methods triad_a0 triad_a5 triad_a10 triad_a50 triad_a100 triad_t0.5 triad_t2.0 triad_x1 triad_x2 triad_x8 triad_x16 triad_noanchor triad_student triad_eb
# design alternatives (appendix D)
python scripts/run_experiments.py --datasets mphrc sensecobot mphrc_all --families lr --level task --methods triad_rcal triad_med triad_hard gtriad gtriad_c
# noise-robust MLP baselines
python scripts/run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --level task --methods mlp_ce mlp_gce mlp_sce mlp_forward mlp_coteach mlp_twohead --seeds 0 1 2
# held-out views: SenseCobot is additionally trained with the mental-effort item alone (q_nasaA)
python scripts/run_experiments.py --datasets sensecobot --families lr hgb rf --level task --qs q_nasaA --methods cond report twohead filter ds raykar triad_cq triad --seeds 0 1 2
# semi-synthetic study
python scripts/semi_synthetic.py --mode uniform --reps 3
python scripts/semi_synthetic.py --mode structured --reps 5
# tables and figures (incl. participant-level bootstraps: significance.tex, heldout_sig.tex; --skip-sig reuses cached ones)
python scripts/make_tables.py
# the two composed 4K figures (only needed if the artwork in paper/figures/art/ or the numbers change)
python scripts/compose_figures.py
cd paper; latexmk -pdf main.tex
```

## Elsevier manuscripts (Neurocomputing and ESWA)

Two independent papers share the TRIAD numbers and the CAS template. They do not share a title, an introduction, a related-work cut, or a discussion. The class files in `paper/elsevier/els-cas-templates` are used as shipped; do not edit them.

```powershell
python scripts/compose_elsevier_figures.py
python scripts/make_journal_figures.py
$root = (Get-Location).Path -replace '\\','/'
$env:TEXINPUTS = "$root/paper/elsevier/els-cas-templates//;"
Set-Location paper\elsevier\nc;   latexmk -pdf -interaction=nonstopmode main.tex
Set-Location ..\eswa;             latexmk -pdf -interaction=nonstopmode main.tex
```

PDFs (one latest file each): `paper/elsevier/nc/main.pdf` (10 numbered pages) and `paper/elsevier/eswa/main.pdf` (11 numbered pages), plus the CAS graphical-abstract and highlights pages. Authors: Kang Yao, Yang Zhang, Weiwei Fu, Jinjiang Cui. Code URL in both manuscripts: `https://github.com/xkk9866/c4-essd`.
Journal tables in `paper/tables/journal/`: datasets, main, failed, heldout; ESWA also uses implications (five tables). WESAD's design-axis 1.00 column is omitted from the main table and from Figure 3 because every design-aware method saturates on the linearly separable TSST. Figure 1 is a two-column overview; Figure 2 is a single-column portrait framework. Compact experiment figures: cross, failed, pi, ceiling, semi, ablation. CHI figures in `paper/figures/` are not touched. The remaining `Overfull \hbox` at `\maketitle` is reproduced by the untouched CAS class.

## IEEE Sensors Journal manuscript

One paper, written for the sensor journal rather than copied from the Elsevier pair: sensing chain first, then the estimator. The class files in `paper/ieeeSensorJounal/IEEE-JSEN-LaTeX-template-202405` are used as shipped; do not edit them. Copy `LOGO-jsen-web.eps` into the compile directory so `epstopdf` can convert the journal mark.

```powershell
python scripts/compose_jsen_figures.py
python scripts/make_journal_figures.py
$root = (Get-Location).Path -replace '\\','/'
$env:TEXINPUTS = "$root/paper/ieeeSensorJounal/IEEE-JSEN-LaTeX-template-202405//;"
Copy-Item paper\ieeeSensorJounal\IEEE-JSEN-LaTeX-template-202405\LOGO-jsen-web.eps paper\ieeeSensorJounal\
Set-Location paper\ieeeSensorJounal; latexmk -pdf -interaction=nonstopmode main.tex
```

PDF: `paper/ieeeSensorJounal/main.pdf` (14 pages; authors Kang Yao, Yang Zhang, Weiwei Fu, Jinjiang Cui).
Writing playbook: `docs/IEEE-Sensors必中指南.md`.
The sensing-chain table (channels, window, feature count, primary report) sits in Section III. Shared result tables are datasets, main and failed; held-out, ablation and semi-synthetic results are the same journal figures as Elsevier. Single-column floats use `[!t]`; two-column floats use `figure*` / `table*` with `[!t]`. The two remaining `Overfull \hbox` lines occur `while \output is active` and come from the journal header in `jsen.sty`.


Method names accepted by `run_experiments.py --methods` (see `make_method` in `evaluation.py`): `cond report regress
twohead filter ds raykar mlp_*`, `triad`, `triad_cq` (tau = 0), `triad_no{c,q,cf,alpha,prior,anchor}`, `triad_a<A>`,
`triad_t<TAU>`, `triad_x<N>`, `triad_student`, `triad_eb`, `triad_mq`, `triad_rcal`, `triad_med`, `triad_hard`,
`gtriad[_c]`.

Runs are cached under `results/runs/<dataset>/<questionnaire>/<level>/<family>/<method>_s<seed>.{json,npz}`;
`--force` recomputes. The full grid takes a few hours on a 16-core machine; TRIAD itself fits in seconds.

## Using TRIAD on your own data

```python
from triad.model import TRIAD
from sklearn.linear_model import LogisticRegression

m = TRIAD(clf=LogisticRegression(C=0.1, max_iter=5000))
m.fit(X, cond, design, Q, None, subj)      # X: n x d features, cond: condition names, design: 0/1/-1 intent,
                                           # Q: n x m questionnaire scores, subj: participant ids
scores = m.predict_logit(X_new)            # deployed detector (log-odds of the activated state)
m.pi_                                      # manipulation efficacy per condition
m.label_noise(R)                           # flip rates, kappa and AUC ceiling for any binary label R
```

Key options: `pi_prior` (pseudo-sessions behind the design intent, 20; `"median"` / `"eb"` for the data-driven
centres of appendix D), `tau` (weight of the physiological view, 1), `n_iter_x` (maximum physiological sweeps, 16),
`anchor` / `anchor_tol` (report-validity constraint, on / delta = 0.005), `cross_fit` / `n_folds` (out-of-fold nuisance
predictions, on / 5), `calibrate` (`False`, `"platt"`, `"report"`, `"cq"`: physiological link calibration; the paper
uses `False`), `deploy` (`"soft"` posterior labels or `"hard"` MAP labels for the deployed classifier),
`q_dist="t"` (Student-t reports). `m.n_x_used_` reports how many physiological sweeps the constraint accepted.

## Submission compliance (CHI 2027)

| Requirement | Limit | Ours |
|---|---|---|
| Abstract | <= 150 words | 147 |
| Body length (excl. references, captions, appendices) | ~7,000-8,000 average; > 12,000 desk-rejected | 8,372 `texcount` words-in-text, of which 475 are inline-formula tokens (prose ~7,900) |
| References | 70-100 is the range in accepted exemplars | 81 cited |
| Figures / tables | full-width teaser on page 1, panels cited verbatim in the intro | `fig_teaser.png`, panels (1)-(4) |
| Build | zero overfull boxes, zero undefined references | clean `latexmk` run, 32 pages |
| Video figure | optional; H.264 MP4, 1080p, 16:9, <= 300 MB, <= 5 min recommended, must be anonymous | 4:58, 12.5 MB, 1920x1080 @ 30 fps, yuv420p, AAC 48 kHz; no identity in frame or in the container metadata |
| Video subtitles | strongly encouraged at submission, required at camera-ready | `dist/chi2027_video_figure.srt`, 76 cues |

`docs/CHI必中指南.md` holds the full pre-submission checklist, including the two items still requiring a human
pass (a manual anonymity sweep of the prose, and the camera-ready author block).

## Anonymity

This package is anonymised for review. Author information, acknowledgements and institution-specific paths are
withheld until camera-ready.
