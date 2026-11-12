# settings of the 19D drone-landing experiments, shared by every script
import numpy as np


## policy input
# the 19D state of the drone (CtrlAviary's 20D observation without its last 4 entries, the previous motor RPM) and the target:
#   0:3 XYZ position (m), 3:7 quaternion (x, y, z, w), 7:10 roll, pitch, yaw (rad),
#   10:13 XYZ linear velocity (m/s), 13:16 XYZ angular velocity (rad/s), 16:19 landing target (x, y, z)
# the action is the four motor RPM commands
state_dim = 19
action_dim = 4

# each experiment appends irrelevant entries to the 19D state:
#   clean  nothing (19D)
#   e2     3 entries of uniform noise in [-1, 1], drawn once per demonstration / test episode (22D)
#   e1_k   the position (3) and quaternion (4) of k other drones flying freely around the workspace (26D, 33D, 40D)
#   e3     a fake landing goal, correlated with the real goal in the training data only (22D)
#   e4     the e2 input passed through a fixed random invertible nonlinear map (StateScramble in models.py), so the
#          irrelevant noise is spread over every input entry (22D)
extra_dims = {"clean": 0, "e2": 3, "e1_1": 7, "e1_2": 14, "e1_3": 21, "e3": 3, "e4": 3}
experiments = tuple(extra_dims)

# defaults; run_experiment.py requires explicit condition arguments
default_seed = 2
default_experiment = "e2"
default_n_demos = 500


## how the datasets in datasets/data were collected
# ONE dataset for every training seed: the training seed sets only the initial weights, the batch order and the e4 map
# training demos: 1000 start states and targets are drawn with seed data_seed, and the dataset keeps 750 of them, in the
# order of default_rng(data_seed).permutation(1000); the extra entries of e2, e1, e3 also come from data_seed
# a training set of N demos is the first N demos of the dataset, so smaller training sets are nested in larger ones
data_seed = 0
n_candidate_demos = 1000
n_train_demos = 750
# test cases: 250 start states and targets drawn with a separate seed
test_seed = 3003
n_test_cases = 250

# timing: control at 48 Hz, physics at 240 Hz
ctrl_hz = 48
pyb_hz = 240
# a demonstration lasts at most 12 s, including up to 1 s after touchdown; a test rollout lasts at most 20 s
demo_steps = 12 * ctrl_hz
test_steps = 20 * ctrl_hz

# start: XY uniform in [-0.1, 0.1] m, altitude 0.7 m, level and at rest
start_xy_range = 0.10
altitude = 0.7
# landing target: XY uniform in this box, at the drone's center height when resting on the ground
target_box_lo = np.array([0.25, -0.25])
target_box_hi = np.array([0.45, 0.25])
landing_height = 0.0125
# the expert flies at most this fast (m/s)
expert_speed = 0.25

# e2: half range of the uniform noise (appended after normalizing the 19D state, and never normalized itself)
e2_half_range = 1.0
# e1: workspace of the other drones (same as the flight envelope of our drone)
e1_workspace_lo = np.array([-0.20, -0.40, 0.10])
e1_workspace_hi = np.array([0.60, 0.40, 0.80])
e1_max_drones = 3
# e3: Pearson correlation of each XY coordinate of the fake goal with the real one in the training data
e3_rho = 0.9
# e4: the scramble z = A2 phi(A1 x + b1) + b2 of the normalized e2 input x, drawn from the training seed (the same for
# every model of a seed): condition number of A1, bend of phi, offsets b1 and b2 uniform in [-shift, shift]
e4_condition = 3.0
e4_bend = 0.5
e4_shift = 1.0


## success of a test rollout (policy/mlp/evaluation.py): ground contact, XY error, Z error, speed and tilt within these
## bounds for stable_steps consecutive control steps
success_xy = 0.05
success_z = 0.06
success_speed = 0.15
success_tilt_deg = 20.0
success_stable_steps = 10


## training (policy/mlp/training.py)
hidden_dim = 128
batch_size = 200
learning_rate = 1e-3
epochs = 100
# InneR: number of nearest neighbors of each state (Algorithm 1) and the weight of the loss (Equation 27);
# InneR-C uses the same weight
k_neighbors = 8
gamma = 0.01
# weights of the Jacobian-norm baselines (squared Frobenius norm and nuclear norm of J)
gamma_frobenius = 0.01
gamma_nuclear = 0.01
# lower bound of the detached trace in the InneR ratio
eps = 1e-6


def input_dim(experiment):
    return state_dim + extra_dims[experiment]


# the extra entries of e4 are the e2 noise (scrambled together with the state)
def data_key(experiment):
    if experiment == "e4":
        return "e2"
    return experiment[:2] if experiment.startswith("e1") else experiment


def run_name(experiment, n_demos, seed):
    return f"{experiment}_n{n_demos}_s{seed}"
