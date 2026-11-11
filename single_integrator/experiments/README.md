# Section VII experiments

The four simulation experiments of Section VII of the paper, run with the hyperparameters of the paper over 10
seeds. One run = one cell (policy, environment, state dimension, demonstrations, shortcut or scramble) x one algorithm x
one seed; every run saves its model and appends one result line. `cells.py` enumerates the cells once and is the only
place the experiments are defined.

## The experiments (41 cells x 5 algorithms x 10 seeds = 2050 runs)

Algorithms: `bc`, `inner` (InneR), `frobenius`, `nuclear`, `inner_c` (InneR-C, k = B). Policies: `mlp` (CPU) and
`diffusion` (GPU). Environments: `static` and `dynamic`. Shared settings: n = 16 unless varied, m = 4, starts uniform in
[-10, 10]^n, 40 steps, success = final distance below 1, batch 200, MLP 200 epochs at lr 1e-3, diffusion 1000 epochs at
lr 3e-3, k = 8, gamma 0.2 (MLP) / 0.3 (diffusion), Fisher rank 8, N_a = 8, N_eps = 4; norm weights 0.01 / 0.01 (MLP) and
1e-4 / 1e-3 (diffusion). Seeds 0 to 9: the seed sets the dataset, the initial weights, the batch order, the scramble draw
and the 1000 test starts, so the five algorithms of a seed are paired.

| Id | Question | Cells | Extra flags |
|---|---|---|---|
| E1 | dataset size | n = 16; static 25, 50, 100, 200, 400 demos; dynamic 50, 100, 200, 400, 800 (both policies) | none |
| E2 | irrelevant state dimension | static n = 8 (MLP only), 12, 16, 24, 32 at 100 (MLP) / 50 (diffusion) demos; dynamic n = 13, 16, 24, 32 at 400 / 200 | `--state_dim` |
| E3 | shifted train/test distribution | n = 16, rho = 0.9; MLP 200 static / 400 dynamic demos; diffusion 50 / 300 | `--noise_correlation 0.9` |
| E4 | dispersed information | n = 16, scramble condition 3, bend 0.5; MLP 400 / 800; diffusion 100 / 600 | `--scramble_condition 3 --scramble_bend 0.5` |

The n = 16 cells of E2 are E1 cells (same models, run once, listed under both experiments in the csv). Every run records
success and goal error; E1 also plots the squared Frobenius norm of the Jacobian (trace of F for diffusion) and the
InneR null-space ratio, which every run records for its final model on 200 training states (`jacobian_norm`,
`null_ratio`, plus `nuclear_norm` and the InneR-C ratio `null_ratio_batch`). `python cells.py` prints the cell list.

## Files

- `cells.py`: the experiments (cells, algorithms, seeds), the tag of a run, and which experiments a cell belongs to.
- `run_experiment.py`: trains and evaluates one run; writes `models/<tag>.pt` and a line of `results.jsonl`.
- `make_jobs.py`: writes `jobs_cpu.txt` (MLP, 1050 runs) and `jobs_gpu.txt` (diffusion, 1000 runs), seed-major.
- `run_queue.py`: runs a job list with N workers, skipping runs already in `results.jsonl`; `STOP_<name>` stops it.
- `collect_results.py`: `results.jsonl` -> `results.csv` (one row per run and experiment) and `summary.csv` (mean, s.d.
  and count over seeds per cell and algorithm).
- `plot_paper.py`: `results.csv` -> the paper's figures as SVG in `figures-svg/`; `plot_results.py`: the same with success
  rates, as PNG in `figures/`.
- `results.jsonl`, `results.csv`, `summary.csv`: tracked by git; `models/` and `logs/` are not.

Tag: `<policy>_<algo>_<env>_d<n>_n<demos>[_rho0.9|_cond3]_s<seed>`, e.g. `diffusion_inner_dynamic_d16_n300_rho0.9_s4`.

## Running

```bash
cd single_integrator
toy_environment/venv/bin/python experiments/make_jobs.py
nohup toy_environment/venv/bin/python experiments/run_queue.py --name cpu --jobs experiments/jobs_cpu.txt --workers 10 > experiments/queue_cpu.log 2>&1 &
nohup toy_environment/venv/bin/python experiments/run_queue.py --name gpu --jobs experiments/jobs_gpu.txt --workers 3 > experiments/queue_gpu.log 2>&1 &
toy_environment/venv/bin/python experiments/collect_results.py
```

The `results.jsonl` in this repo holds all 2050 runs, so the queue skips everything; delete it to rerun.

Scheduling (measured on this machine, 28 threads and an RTX 4060): MLP runs use 2 threads each, 10 to 14 at a time;
diffusion runs 3 to 4 at a time on the GPU. Cost per run in seconds per demonstration: MLP 0.12 (BC) and 0.8
(regularized); diffusion 0.8 and about 4, times 1.4 at n = 32. The whole campaign is about 56 CPU worker-hours (6 hours
of wall time) and 200 GPU worker-hours (65 hours with 3 workers); the diffusion runs at dynamic 600 and 800 demonstrations
are the longest (40 to 55 minutes each).

## Imported runs

930 of the result lines come from earlier campaigns with the same cell, algorithm, seed, hyperparameters and code path
(their `source` field names the original file and tag). Before the import the driver reproduced an archived MLP run, an
archived scramble run and an archived diffusion run bit for bit. The imported lines lack `null_ratio_batch` (the InneR-C
ratio was not recorded then).
