"""All drone inputs, dependencies and outputs live under this directory."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT/'datasets/data'
DP_RESULTS = ROOT/'manuscript_runs'
MLP_RESULTS = ROOT/'mlp_runs'
SIMULATOR = ROOT/'gym-pybullet-drones'
