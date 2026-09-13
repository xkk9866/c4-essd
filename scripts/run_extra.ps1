# Extra sensitivity rows: number of EM sweeps with the physiological view (anchoring analysis).
$env:PYTHONWARNINGS = "ignore"
Set-Location $PSScriptRoot\..
python scripts\run_experiments.py --datasets mphrc mphrc_all sensecobot --families lr --methods triad_x1 triad_x2 triad_x8 triad_x16 --level task --n_jobs 6 2>&1 | Tee-Object -FilePath logs\grid_xsweeps.log
# semi-synthetic: the same sweep on the hardest uniform cell and one structured cell
python scripts\semi_synthetic.py --mode uniform --dataset mphrc --eps_C 0.30 --eps_R 0.20 0.35 --reps 3 --methods triad_x1 triad_x8 triad_x16 --n_jobs 6 2>&1 | Tee-Object -FilePath logs\semi_uniform_x.log
python scripts\semi_synthetic.py --mode structured --dataset mphrc --pi_fail 0.3 --eps_R 0.20 0.35 --reps 3 --methods triad_x1 triad_x8 triad_x16 --n_jobs 6 2>&1 | Tee-Object -FilePath logs\semi_structured_x.log
"EXTRA DONE" | Tee-Object -FilePath logs\extra_done.txt
