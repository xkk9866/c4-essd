# Sequential benchmark driver (avoids memory pressure from many concurrent worker pools).
$env:PYTHONWARNINGS = "ignore"
Set-Location $PSScriptRoot\..
$DS = "mphrc wesad sensecobot mphrc_all"
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --level task --qs all --n_jobs 12 2>&1 | Tee-Object -FilePath logs\grid_lr_task.log
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families hgb rf --methods cond report filter twohead ds raykar triad triad_cq --seeds 0 1 2 --level task --n_jobs 12 2>&1 | Tee-Object -FilePath logs\grid_trees_task.log
python scripts\run_experiments.py --datasets mphrc wesad sensecobot mphrc_all --families lr --methods mlp_ce mlp_gce mlp_sce mlp_forward mlp_coteach mlp_twohead --seeds 0 1 2 --level task --n_jobs 8 2>&1 | Tee-Object -FilePath logs\grid_mlp_task.log
python scripts\semi_synthetic.py --dataset mphrc --reps 5 --n_jobs 12 2>&1 | Tee-Object -FilePath logs\semi_synthetic.log
python scripts\run_experiments.py --datasets mphrc wesad sensecobot --families lr --methods cond report filter twohead ds raykar triad triad_cq --level window --n_jobs 12 2>&1 | Tee-Object -FilePath logs\grid_lr_window.log
python scripts\run_experiments.py --datasets mphrc --families lr --methods triad_a0 triad_a5 triad_a10 triad_a50 triad_a100 triad_t0.5 triad_t2.0 --level task --n_jobs 12 2>&1 | Tee-Object -FilePath logs\grid_sens.log
"ALL DONE" | Tee-Object -FilePath logs\grid_done.txt
