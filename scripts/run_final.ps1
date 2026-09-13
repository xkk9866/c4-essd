# Final experiment driver (sequential, modest parallelism: the machine is shared).
$env:PYTHONWARNINGS = "ignore"
Set-Location $PSScriptRoot\..
# 1. semi-synthetic with known latent state: uniform label noise and a failed-manipulation (structured) scenario
python scripts\semi_synthetic.py --mode uniform --dataset mphrc --reps 3 --n_jobs 6 2>&1 | Tee-Object -FilePath logs\semi_uniform.log
python scripts\semi_synthetic.py --mode structured --dataset mphrc --reps 3 --n_jobs 6 2>&1 | Tee-Object -FilePath logs\semi_structured.log
# 2. tree-ensemble deployment classifiers: TRIAD rows recomputed with the logistic nuisance E-step (fast, calibrated)
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families hgb rf --methods triad triad_cq --seeds 0 1 2 --level task --n_jobs 6 --force 2>&1 | Tee-Object -FilePath logs\grid_trees_triad.log
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families hgb rf --methods cond report filter twohead ds raykar --seeds 0 1 2 --level task --n_jobs 6 2>&1 | Tee-Object -FilePath logs\grid_trees_rest.log
# 3. robustness rows: report likelihood / baseline shrinkage / temperature / prior strength
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --methods triad_student triad_eb triad_a0 triad_a5 triad_a10 triad_a50 triad_a100 triad_t0.5 triad_t2.0 --level task --n_jobs 6 2>&1 | Tee-Object -FilePath logs\grid_sens_all.log
# 4. window-level (real-time) deployment for mphrc_all as well
python scripts\run_experiments.py --datasets mphrc_all --families lr --methods cond report filter twohead ds raykar triad triad_cq --level window --n_jobs 6 2>&1 | Tee-Object -FilePath logs\grid_lr_window_all.log
# 5. noise-robust MLPs on the remaining datasets (if missing)
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --methods mlp_ce mlp_gce mlp_sce mlp_forward mlp_coteach mlp_twohead --seeds 0 1 2 --level task --n_jobs 6 2>&1 | Tee-Object -FilePath logs\grid_mlp_all.log
"FINAL DONE" | Tee-Object -FilePath logs\final_done.txt
