# TRIAD

Triangulating a latent psycho-physiological state from the design condition, the self-report and wearable physiology.

This repository contains the estimator, baselines, processed 60-second windows for three public corpora, and the scripts that reproduce the leave-one-subject-out numbers.

## Install

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Data

| Corpus | Source | What is shipped |
|---|---|---|
| MultiPhysio-HRC | Bussolan et al., *Robotics* 14(12):184, 2025 | released 60 s features, STAI-Y1 / NASA-TLX / SAM |
| WESAD | Schmidt et al., ICMI 2018 | 37 chest-band window features, STAI-6 / PANAS / SAM |
| SenseCobot | Borghi et al., HRI 2024 | 65 Shimmer / E4 / AFFDEX features, NASA-TLX items |

Raw recordings are not redistributed. `data/processed/*_windows.csv` are enough to rerun every comparison. To rebuild them from your own copies of the archives:

```powershell
$env:TRIAD_MPHRC_DIR      = "<path>\MultiPhysio-HRC\features"
$env:TRIAD_WESAD_DIR      = "<path>\WESAD"
$env:TRIAD_SENSECOBOT_DIR = "<path>\SenseCobot"
python -m triad.data.mphrc
python -m triad.data.wesad
python -m triad.data.sensecobot
```

## Fit TRIAD

```python
from triad.model import TRIAD
from sklearn.linear_model import LogisticRegression

m = TRIAD(clf=LogisticRegression(C=0.1, max_iter=5000))
m.fit(X, cond, design, Q, None, subj)
scores = m.predict_logit(X_new)
m.pi_                 # manipulation efficacy per condition
m.label_noise(R)      # flip rates, kappa and AUC ceiling for a binary label R
```

Default hyperparameters used in the papers: `pi_prior=20`, `tau=1`, at most 16 physiological sweeps, report-validity tolerance `0.005`, five-fold grouped cross-fitting.

## Reproduce the benchmarks

```powershell
python scripts/run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --level task --qs all
python scripts/semi_synthetic.py --mode uniform --reps 3
python scripts/semi_synthetic.py --mode structured --reps 5
```

Cached metrics live under `results/runs/` and `results/semi_synthetic_*.json`. `results/summary.json` stores the numbers cited in the papers.

Method tags accepted by `--methods`: `cond`, `report`, `regress`, `twohead`, `filter`, `ds`, `raykar`, `mlp_*`, `triad`, and the TRIAD ablations `triad_cq`, `triad_no{c,q,cf,alpha,prior,anchor}`, `triad_a<A>`, `triad_t<TAU>`, `triad_x<N>`.

## Layout

```
src/triad/          estimator, baselines, LOSO driver, feature builders
scripts/            experiment grid and semi-synthetic study
data/processed/     60 s window tables
results/            cached metrics
```
