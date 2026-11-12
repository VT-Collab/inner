# Drone landing simulations

MLP and diffusion policies for the drone landing task (Section VIII of the paper), simulated with [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones).

- `policy/`: diffusion policy, its regularizers, training and rollouts
- `policy/mlp/`: MLP policy, its regularizers, training and rollouts
- `experiments/`: conditions of each experiment, the runner, result collection and plotting
- `mlp_runs/`: results of the MLP runs
- `get_dataset.py`, `replay_demo.py`, `visualize_demo.py`: collect and view expert demonstrations (see `datasets/README.md`)

## Install

Linux, Python 3.12. Diffusion training needs a CUDA GPU. From the repository root:

```bash
git submodule update --init
cd drone
conda create -n drone-policy python=3.12 pip -y
conda activate drone-policy
pip install -r requirements.txt
```

## Data

The runner reads `datasets/data/{e1,e2,e3}/`, each with `train.pkl`, `train_lengths.pkl`, `test.pkl`, `test_lengths.pkl`, `test_extras.pkl` and `norm_stats.pkl`. These datasets are not in the repository.

## Run

| Experiment | Question |
| --- | --- |
| `e1` | irrelevant state dimensions |
| `e2` | dataset size |
| `e3` | shifted train and test distributions |
| `e4` | information dispersed through the state (uses the `e2` data) |

One command trains and tests one policy, method and seed:

```bash
python experiments/run_experiment.py --policy mlp --experiment e1 \
  --method inner --seed 2 --demos 500 --state-dim 19

python experiments/run_experiment.py --policy diffusion --experiment e2 \
  --method inner --seed 2 --demos 100 --state-dim 22
```

- `--method`: `bc`, `inner`, `frobenius`, `nuclear`, `inner_c`
- `--seed`: 2, 12, ..., 92
- `--demos`, `--state-dim`: a pair from the table below
- `--root`: output folder (default `mlp_runs/` or `manuscript_runs/`)

A run that is already recorded is skipped; pass a new `--root` to train it again.

| | MLP | Diffusion |
| --- | --- | --- |
| `e1` demos; dimensions | 500; 19, 26, 33, 40 | 100; 19, 26, 33, 40 |
| `e2` demos (22D) | 100, 300, 500, 700 | 25, 50, 75, 100 |
| `e3`, `e4` demos (22D) | 500 | 100 |
| InneR neighbors; rank | 8; 4 | 32; 8 |
| InneR, InneR-C weight | 0.01 | 0.01 |
| Frobenius; Nuclear weight | 0.01; 0.01 | 0.0001; 0.001 |

## Results and figures

```bash
python experiments/collect_results.py
python experiments/plot_results.py
```

`collect_results.py` writes the result sheets and `plot_results.py` draws the figures into `manuscript_runs/csv_figures/results/`. Plotting needs the runs of both policies, plus Times New Roman and Liberation Serif (set `DRONE_FONT_PATH` to a Times New Roman `.ttf` if it is not installed).
