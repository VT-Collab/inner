#!/bin/bash

# stop at the first error, and run from this folder
set -e
cd "$(dirname "$0")"

# seed of this run (default 0), passed to every script: ./bash.sh 3 collects, trains, and tests with seed 3
SEED=${1:-0}

# collect one dataset, shared by both policies
mkdir -p datasets models
python3 get_dataset.py $SEED

# MLP policy: baseline (standard BC, 0), ours (InneR, 1), Jacobian regularization baselines (Frobenius, 2; nuclear, 3), and InneR without clustering (InneR-C, 4)
python3 train_mlp_policy.py $SEED
python3 test_mlp_policy.py $SEED

# diffusion policy: baseline (standard diffusion policy, 0), ours (InneR with the Fisher information, 1), Fisher regularization baselines (Frobenius, 2; nuclear, 3), and InneR without clustering (InneR-C, 4)
python3 train_diffusion_policy.py $SEED
python3 test_diffusion_policy.py $SEED
