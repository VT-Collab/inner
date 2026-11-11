# state: 7 joint positions, gripper (1 open, -1 closed), n_objects x (x, y, z, yaw)
# action: change in the 7 joint positions per step, gripper command (1 open, -1 closed)
n_joints = 7
proprio_dim = n_joints + 1
n_objects = 4
object_dim = 4
state_dim = proprio_dim + n_objects * object_dim
action_dim = n_joints + 1


def dims(stats):
    """state and action dimensions of a dataset, from its stats"""
    from kinematics import STATE_SPACES, ACTION_SPACES
    arm_state, arm_action = STATE_SPACES[stats.get("state_space", "joint")], ACTION_SPACES[stats.get("action_space", "joint")]
    labels = stats.get("labels", range(n_objects))
    no_yaw = [k for k in stats.get("no_yaw", []) if k in labels]
    one = arm_state + 1 + len(labels) * object_dim - len(no_yaw) + 3 * len(stats.get("relative", []))
    return one * stats.get("obs_window", 1), (arm_action + 1) * stats.get("chunk", 1)
