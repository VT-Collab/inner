"""Shared constants for the one-drone landing task."""

import pickle

import numpy as np


CONTROL_HZ = 48
SIM_HZ = 240
ALTITUDE = 0.7
LANDING_HEIGHT = 0.0125  # Half the CF2X collision cylinder's height.
RENDER_EVERY = 5


def lengths_path(dataset_path):
    """Companion file marking trajectory boundaries in the flat dataset."""
    return dataset_path.with_name(f"{dataset_path.stem}_lengths.pkl")


def load_demo(dataset_path, demo_number):
    """Select one trajectory from the flat state-action rows."""
    with dataset_path.open("rb") as file:
        rows = pickle.load(file)
    with lengths_path(dataset_path).open("rb") as file:
        lengths = pickle.load(file)
    if not 1 <= demo_number <= len(lengths):
        raise ValueError(f"--demo must be between 1 and {len(lengths)}")
    if any(length < 1 for length in lengths) or sum(lengths) != len(rows):
        raise ValueError("trajectory lengths do not match the dataset")
    start = sum(lengths[:demo_number - 1])
    data = np.asarray(rows[start:start + lengths[demo_number - 1]], dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 27 or not np.isfinite(data).all():
        raise ValueError("expected finite one-drone demonstration rows with 27 columns")
    return data
