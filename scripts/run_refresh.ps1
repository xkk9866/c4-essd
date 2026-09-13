# Recompute every TRIAD row with the final model (report-anchored physiological sweeps); baselines are unchanged.
$env:PYTHONWARNINGS = "ignore"
Set-Location $PSScriptRoot\..
$T = "triad triad_cq triad_nocf triad_noalpha triad_noprior triad_noc triad_noq triad_mq triad_noanchor"
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --methods triad triad_cq triad_nocf triad_noalpha triad_noprior triad_noc triad_noq triad_mq triad_noanchor --level task --qs all --n_jobs 6 --force 2>&1 | Tee-Object -FilePath logs\refresh_lr_task.log
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --methods triad triad_cq --level window --n_jobs 6 --force 2>&1 | Tee-Object -FilePath logs\refresh_lr_window.log
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --methods triad_student triad_eb triad_a0 triad_a5 triad_a10 triad_a50 triad_a100 triad_t0.5 triad_t2.0 --level task --n_jobs 6 --force 2>&1 | Tee-Object -FilePath logs\refresh_lr_sens.log
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families hgb rf --methods triad triad_cq --seeds 0 1 2 --level task --n_jobs 6 --force 2>&1 | Tee-Object -FilePath logs\refresh_trees.log
python scripts\semi_synthetic.py --mode uniform --dataset mphrc --reps 3 --methods triad triad_cq --n_jobs 6 --force 2>&1 | Tee-Object -FilePath logs\refresh_semi_uniform.log
python scripts\semi_synthetic.py --mode structured --dataset mphrc --reps 3 --methods triad triad_cq --n_jobs 6 --force 2>&1 | Tee-Object -FilePath logs\refresh_semi_structured.log
"REFRESH DONE" | Tee-Object -FilePath logs\refresh_done.txt
