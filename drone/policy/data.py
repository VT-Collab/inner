import pickle
from pathlib import Path
import numpy as np
import torch
from paths import DATA
def load_pickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def read_data(experiment, split, dtype=np.float32):
    rows = np.asarray(load_pickle(DATA / experiment / f'{split}.pkl'), dtype=dtype)
    lengths = np.asarray(load_pickle(DATA / experiment / f'{split}_lengths.pkl'), dtype=np.int64)
    expected = dict(e2=26, e1=44, e3=26)[experiment]
    if rows.ndim != 2 or rows.shape[1] != expected or not np.isfinite(rows).all():
        raise ValueError(f'Invalid {experiment}/{split} rows')
    if (lengths <= 0).any() or lengths.sum() != len(rows):
        raise ValueError('Invalid episode boundaries')
    return rows, lengths


def select_inputs(rows, dimension):
    return torch.from_numpy(rows[:, :dimension].copy())


def normalization(states, actions, experiment=None):
    # Physical/raw features are standardized using ONLY the selected training prefix.
    # Actions use training min/max, since DDIM clips to [-1,1].
    mean = states.mean(0)
    std = states.std(0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    if experiment == 'e2':
        # Supplied E2 norm_stats deliberately keep uniform nuisance values in [-1,1].
        mean[19:] = 0
        std[19:] = 1
    lo, hi = actions.amin(0), actions.amax(0)
    return dict(state_mean=mean, state_std=std, action_mean=(lo+hi)/2,
                action_std=((hi-lo)/2).clamp_min(1e-6))

