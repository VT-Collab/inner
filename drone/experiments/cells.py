import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import DP_RESULTS
from naming import checkpoint_relative
ROOT = DP_RESULTS
SEEDS = [2,12,22,32,42,52,62,72,82,92]
EXPERIMENTS = ['e1','e2','e3','e4']
METHODS = ['dp','dp_inner','dp_frobenius','dp_nuclear','dp_inner_c']
WEIGHTS = dict(dp=0., dp_inner=.01, dp_frobenius=.0001, dp_nuclear=.001, dp_inner_c=.01)
def conditions(experiment):
    return [(n, 22) for n in [25, 50, 75, 100]] if experiment == 'e2' else (
        [(100, d) for d in [19, 26, 33, 40]] if experiment == 'e1' else [(100, 22)])


def config(experiment, n, dimension, seed, method, smoke=False):
    cfg = dict(experiment=experiment, demos=n, state_dim=dimension, seed=seed, method=method,
                epochs=1 if smoke else 100, batch_size=200, hidden_dim=128, lr=.001,
                weight=WEIGHTS[method], rank=8, k='full_batch' if method=='dp_inner_c' else 32, action_samples=8, score_samples=4,
                reg_batch_size=200, diffusion_steps=100, evaluation_steps=48 if smoke else 960,
                evaluation_cases=2 if smoke else 250,
                state_normalization='selected_training_prefix_zscore_constant_guard_1e-6_E2_extras_identity',
                action_normalization='selected_training_minmax_minus1_plus1',
                episode_selection='first_N_complete_episodes', smoke=smoke)
    if experiment == 'e4':
        cfg.update(scramble_seed=seed, scramble_condition=3., scramble_shift=1.,
                   scramble_bend=.5, scramble_tau=5.,
                   state_normalization='E2 selected training prefix then fixed nonlinear scramble')
    cfg['checkpoint']=checkpoint_relative(cfg)
    return cfg


def run_folder(root, cfg):
    return root/cfg['experiment']/f"seed{cfg['seed']}"/f"n{cfg['demos']}_d{cfg['state_dim']}"/cfg['method']
