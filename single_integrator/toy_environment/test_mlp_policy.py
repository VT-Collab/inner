import sys
import torch
import numpy as np
from models import MLPPolicy
from config import state_dim, action_dim, hidden_dim, dynamic_goal, n_steps


# names of the trained models, by save number (see train_mlp_policy.py)
# we typically do not test (2: "Frobenius", 3: "nuclear", 4: "InneR-C") for spot checks (the norms were worse than the baseline in
# preliminary experiments); to roll them out, add them here after enabling their train_model calls in train_mlp_policy.py
MODEL_NAMES = {0: "BC", 1: "InneR"}


# check the performance of the learned policy: roll it out from the initial states for n_steps, like the expert demonstrations
def test_policy(savenumber, init_states, state_dim, hidden_dim, action_dim, n_steps):

    # load the trained model
    model = MLPPolicy(state_dim, hidden_dim, action_dim)
    model.load_state_dict(torch.load('models/mlp_weights' + str(savenumber)))
    model.eval()

    # copy the shared initial states (the rollout updates states in place)
    states = init_states.copy()
    goals = states[:, action_dim:2*action_dim].copy()
    # if using the dynamic goal environment, rows whose last state entry is negative use the next goal indices
    # (the rollout only updates the robot entries, so the goal of each row is fixed)
    if dynamic_goal:
        toggle = states[:, -1:] < 0.0
        goals = np.where(toggle, states[:, 2*action_dim:3*action_dim], goals)

    # rollout the learned policy
    with torch.no_grad():
        for _ in range(n_steps):
            # get action from model
            state_tensor = torch.FloatTensor(states)
            actions = model(state_tensor).numpy()
            # compute next states
            states[:, 0:action_dim] += actions

    # calculate average error across the batch, and the success rate of the paper (final distance below one expert action)
    errors = np.linalg.norm(goals - states[:, 0:action_dim], axis=1)
    average_goal_error = np.mean(errors)
    success_rate = np.mean(errors < 1.0)
    print(f"MLP model {savenumber} ({MODEL_NAMES[savenumber]}) performance: {np.round(average_goal_error, 2)} | success: {success_rate:.3f}")
    return average_goal_error


# test models
if __name__ == "__main__":

    # seed of this run, from the command line (default 0); bash.sh passes the same seed to every script
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # sample one seeded batch of initial states shared by every model
    n_tests = 1000
    rng = np.random.default_rng(seed)
    init_states = rng.uniform(-10, 10, (n_tests, state_dim))

    # get goal error for baseline (standard BC, 0), ours (InneR, 1), the Jacobian regularization baselines (Frobenius, 2; nuclear, 3), and InneR without clustering (InneR-C, 4)
    goal_errors = {}
    for savenumber in MODEL_NAMES:
        goal_errors[savenumber] = test_policy(savenumber, init_states, state_dim, hidden_dim, action_dim, n_steps)
