"""Shared experimental settings for state scaling and synthetic nuisance inputs.

The state indices are fixed by get_dataset.py: [20D observation, 3D target].
Do not scale the appended clutter a second time.
"""

import numpy as np
import torch

ORIGINAL_STATE_DIM = 23
ACTION_DIM = 4

# Deliberately fixed engineering bounds covering the observed flight envelope.
# Panda-style: map each bound to about -10/+10 without clipping. Unlike
# train-set zscore these stay IDENTICAL for every dataset size and seed.
# The 23rd target-height dimension is constant, so its two bounds coincide.
STATE_MIN = [
    -0.20, -0.40, 0.00,                   # drone position
    -0.15, -0.15, -0.70, 0.75,           # quaternion xyzw
    -0.20, -0.20, -1.50,                 # roll, pitch, yaw
    -0.40, -0.40, -0.40,                 # linear velocity
    -4.00, -4.00, -1.00,                 # angular velocity
    0.00, 0.00, 0.00, 0.00,             # previous RPM
    0.20, -0.30, 0.0125,                 # target XYZ
]
STATE_MAX = [
    0.60, 0.40, 0.80,
    0.15, 0.15, 0.70, 1.05,
    0.20, 0.20, 1.50,
    0.40, 0.40, 0.40,
    4.00, 4.00, 1.00,
    22000.0, 22000.0, 22000.0, 22000.0,
    0.50, 0.30, 0.0125,
]
assert len(STATE_MIN) == len(STATE_MAX) == ORIGINAL_STATE_DIM


def panda_scaling(dtype=torch.float32):
    """Return center and divisor such that nominal bounds map to [-10,10]."""
    lo = torch.tensor(STATE_MIN, dtype=dtype)
    hi = torch.tensor(STATE_MAX, dtype=dtype)
    center = (hi + lo) / 2
    divisor = (hi - lo) / 20
    divisor = torch.where(divisor.abs() < 1e-6, torch.ones_like(divisor), divisor)
    return center, divisor


def sample_train_clutter(n_rows, dim, half_range, seed):
    """Independent noise for each row, fixed throughout training (like Panda)."""
    if dim == 0:
        return None
    if dim < 0 or half_range < 0:
        raise ValueError("Invalid clutter dimension or range")
    gen = torch.Generator().manual_seed(int(seed))
    return (2 * torch.rand((n_rows, dim), generator=gen) - 1) * float(half_range)


def episode_clutter_base(case_id, dim, seed):
    """One U(-1,1) vector per episode, reproducible across BC/InneR and scales."""
    if dim <= 0:
        return None
    # Only caller-supplied seed, case id, and dimensionality affect the draw.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(case_id)]))
    return rng.uniform(-1, 1, size=dim).astype(np.float32)


def is_descent_phase(obs20, target):
    """Exactly the expert's approach/descent switch from get_dataset.expert_rpm."""
    error_xy = np.linalg.norm(np.asarray(obs20[:2]) - np.asarray(target[:2]))
    speed_xy = np.linalg.norm(obs20[10:12])
    return bool(error_xy < 0.04 and speed_xy < 0.06)
