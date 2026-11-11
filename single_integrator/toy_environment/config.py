# dimensions of the point-reach environment
# state: the first action_dim entries are the robot, the next action_dim entries are the goal, and the rest are irrelevant noise
# (with a dynamic goal, the action_dim entries after the goal are a second goal, and the last entry is the switch between them)
state_dim = 16
action_dim = 4
assert state_dim >= 2 * action_dim, "the state must hold the robot and the goal"

# determine whether the goal has static indices in the state, or whether the goal changes based on a switch
# static goal: the goal is always the next action_dim entries after the robot
# dynamic goal: if the last state entry (the switch) is negative, the goal is instead the action_dim entries after that
# note that if dynamic_goal is True, the state_dim must be > 3 * action_dim so that the switch is a noise entry
dynamic_goal = False
assert not dynamic_goal or state_dim > 3 * action_dim, "the state must hold the robot, both goals, and the switch"

# length of every rollout when collecting data and testing policies
n_steps = 40

# width of the hidden layers of every policy network (MLP and diffusion)
hidden_dim = 256

# random scrambling of the state seen by the policies (StateScramble in models.py): z = A2 phi(A1 s + b1) + b2 with a random
# invertible matrix A1, random offsets b1 and b2, a random rotation A2, and a bend phi of each entry, so the task-relevant
# information is spread across every entry of the policy input but none of it is lost. the dataset and the tests keep the
# axis-aligned state above; the transformation is drawn from the seed of a run (the same for every model of that run) and
# saved with the weights, so every model is tested with the transformation it was trained with
scramble_state = False
# upper bound on the ratio of the largest to the smallest singular value of A1 (1 gives a rotation, larger values also stretch)
scramble_condition = 3.0
# the entries of the offsets b1 and b2 are uniform in [-scramble_shift, scramble_shift]
scramble_shift = 5.0
# strength of the bend, phi_i(x) = x + c_i tau tanh(x / tau) with c_i uniform in [-scramble_bend, scramble_bend] and tau = 5
# (below 1, so phi is strictly increasing and the map stays invertible); 0 gives the affine scramble z = A1 s + b1
scramble_bend = 0.5
assert 0.0 <= scramble_bend < 1.0, "the bend must be below 1 for the scramble to be invertible"

# correlate the irrelevant entries of the training data with the goal (a shortcut that does not hold at test time)
# when on, get_dataset.py makes every noise entry of a demonstration's initial state correlated with one coordinate of the
# (active) goal, with Pearson correlation 0.9 (correlate_noise in get_dataset.py); when off they are independent as before
# the test scripts keep drawing independent uniform states, so the correlation exists only in the training data
correlated_noise = False
