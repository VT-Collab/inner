import sys
import pickle
import numpy as np
from config import state_dim, action_dim, dynamic_goal, n_steps, correlated_noise


# get an action that goes directly to the goal
def get_action(state, goal_toggle):
	# get the goal (the goal indices change if the toggle is on)
	goal = state[action_dim:2*action_dim]
	if goal_toggle:
		goal = state[2*action_dim:3*action_dim]
	# move towards goal
	action_vec = goal - state[0:action_dim]
	# limit action magnitude
	action_norm = np.linalg.norm(action_vec)
	if action_norm > 1.0:
		action_vec /= action_norm
	return action_vec

# make the noise entries of an initial state correlated with the active goal (in place)
# noise entry k becomes alpha * goal[k % action_dim] + (1 - alpha) * itself, which stays in [-10, 10] and draws no random numbers
# alpha = rho / (rho + sqrt(1 - rho^2)) makes the correlation exactly rho for independent uniform draws
# (with the dynamic goal the switch is not changed, so with state_dim = 3 * action_dim + 1 there is nothing to correlate)
def correlate_noise(state, dynamic_goal, rho):
	# the noise entries come after the goal, or after the second goal and before the switch with a dynamic goal
	goal = state[action_dim:2*action_dim]
	noise = state[2*action_dim:]
	if dynamic_goal:
		noise = state[3*action_dim:-1]
		if state[-1] < 0.0:
			goal = state[2*action_dim:3*action_dim]
	# mix each noise entry with its goal coordinate (the goal is repeated cyclically over the noise entries)
	alpha = rho / (rho + np.sqrt(1.0 - rho**2))
	noise *= 1.0 - alpha
	noise += alpha * np.resize(goal, len(noise))

# add a (state, action) pair to the dataset
def add_to_dataset(dataset, state, action):
	dataset.append(state.tolist() + action.tolist())

# collect a dataset of expert demonstrations: from a random initial state, the expert moves towards the goal for n_steps
# steps and every (state, action) pair along the way is a datapoint (the same rollout the test scripts use for the policies)
# with noise_correlation > 0, the noise entries of each initial state are correlated with the goal (correlate_noise)
def get_dataset(state_dim, n_demos, dynamic_goal, n_steps, noise_correlation=0.0):
	dataset = []
	for _ in range(n_demos):
		# pick an initial state uniformly at random
		state = np.random.uniform(-10, 10, state_dim)
		# if correlated, tie the noise entries to the goal (training data only; the test states stay independent)
		if noise_correlation > 0.0:
			correlate_noise(state, dynamic_goal, noise_correlation)
		# if using the dynamic goal environment, switch the goal based on the sign of the last state entry
		goal_toggle = dynamic_goal and state[-1] < 0.0
		# roll out the expert for n_steps (goal and noise dims stay fixed)
		for _ in range(n_steps):
			# calculate action towards the goal
			action = get_action(state, goal_toggle)
			# add the state-action pair to the dataset (the state before the action is applied, as in the test rollouts)
			add_to_dataset(dataset, state, action)
			# update the state
			state[0:action_dim] += action
	pickle.dump(dataset, open("datasets/dataset.pkl", "wb"))
	print("dataset has this many state-action pairs:", len(dataset))


# get dataset
if __name__ == "__main__":

	# seed of this run, from the command line (default 0); bash.sh passes the same seed to every script
	seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
	np.random.seed(seed)

	# hyperparameters
	# state (dimensions set in config.py):
		# first action_dim are xyz-rot of robot (for example)
		# next action_dim are xyz-rot of goal cube (for example)
		# alternatively, if using a dynamic goal environment, the indices for the goal change
		# based on whether the final state element is positive or negative
		# additional dimensions are noise
	# number of expert demonstrations; each is n_steps long (set in config.py), so the dataset has n_demos * n_steps datapoints
	n_demos = 100

	# correlation between each noise entry and the goal (only if correlated_noise is set in config.py, otherwise independent)
	noise_correlation = 0.9 if correlated_noise else 0.0

	# collect dataset
	get_dataset(state_dim, n_demos, dynamic_goal, n_steps, noise_correlation)