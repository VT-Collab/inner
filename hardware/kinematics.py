import numpy as np
from config import n_joints

# franka panda, modified DH (a_{i-1}, d_i, alpha_{i-1}); then flange (d = 0.107) and hand (-45 deg, tcp 0.1034)
DH = [(0, 0.333, 0), (0, 0, -np.pi / 2), (0, 0.316, np.pi / 2), (0.0825, 0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0, 0, np.pi / 2), (0.088, 0, np.pi / 2)]


def _link(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    T = np.zeros(theta.shape + (4, 4))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 3] = ct, -st, a
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2], T[..., 1, 3] = st * ca, ct * ca, -sa, -sa * d
    T[..., 2, 0], T[..., 2, 1], T[..., 2, 2], T[..., 2, 3] = st * sa, ct * sa, ca, ca * d
    T[..., 3, 3] = 1
    return T


def fk(q):
    """tcp pose (..., 4, 4) for joints q (..., 7)"""
    q = np.asarray(q, dtype=np.float64)
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for i, (a, d, alpha) in enumerate(DH):
        T = T @ _link(a, d, alpha, q[..., i])
    zero = np.zeros(q.shape[:-1])
    T = T @ _link(0, 0.107, 0, zero) @ _link(0, 0.1034, 0, zero - np.pi / 4)
    return T


def so3_log(R):
    """rotation vector of R, for angles below pi"""
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    angle = np.arccos(cos)
    vee = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    scale = np.where(angle < 1e-6, 0.5, angle / (2 * np.sin(np.maximum(angle, 1e-6))))
    return scale[..., None] * vee


def so3_exp(v):
    angle = np.linalg.norm(v, axis=-1)[..., None, None]
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    small = angle < 1e-6
    a = np.where(small, 1.0, np.sin(angle) / np.where(small, 1, angle))
    b = np.where(small, 0.5, (1 - np.cos(angle)) / np.where(small, 1, angle) ** 2)
    return np.eye(3) + a * K + b * K @ K


def pose_error(T, T_goal):
    """error from T to T_goal in the base frame: position, rotation vector"""
    return np.concatenate([T_goal[..., :3, 3] - T[..., :3, 3],
                           so3_log(T_goal[..., :3, :3] @ np.swapaxes(T[..., :3, :3], -1, -2))], axis=-1)


def jacobian(q, eps=1e-6):
    """geometric jacobian (..., 6, 7) of the tcp by finite differences"""
    T = fk(q)
    Tq = fk(q[..., None, :] + eps * np.eye(n_joints))
    return np.swapaxes(pose_error(T[..., None, :, :], Tq), -1, -2) / eps


def ik(q, T_goal, iters=10, damping=1e-2):
    """joints that reach T_goal from q, by damped least squares"""
    q = np.array(q, dtype=np.float64)
    for _ in range(iters):
        J = jacobian(q)
        e = pose_error(fk(q), T_goal)
        JJt = J @ np.swapaxes(J, -1, -2) + damping ** 2 * np.eye(6)
        q = q + (np.swapaxes(J, -1, -2) @ np.linalg.solve(JJt, e[..., None]))[..., 0]
    return q


STATE_SPACES = {"joint": n_joints, "cartesian": 9, "position": 3}
# _abs actions are targets, rotation vector from R_REF
ACTION_SPACES = {"joint": n_joints, "cartesian": 6, "joint_abs": n_joints, "cartesian_abs": 6}
# gripper pointing down (hand x = base x)
R_REF = np.diag([1.0, -1.0, -1.0])


def arm_state(q, space):
    if space == "joint":
        return np.asarray(q)
    T = fk(q)
    if space == "position":
        return T[..., :3, 3]
    return np.concatenate([T[..., :3, 3], T[..., :3, 0], T[..., :3, 1]], axis=-1)


def arm_action(q, q_next, space):
    """action that moves the arm from q to q_next"""
    if space == "joint":
        return np.asarray(q_next) - np.asarray(q)
    if space == "joint_abs":
        return np.asarray(q_next)
    if space == "cartesian_abs":
        T = fk(q_next)
        return np.concatenate([T[..., :3, 3], so3_log(T[..., :3, :3] @ R_REF.T)], axis=-1)
    return pose_error(fk(q), fk(q_next))


def joint_target(q, action, space):
    """joint target for an action at q"""
    if space == "joint":
        return np.asarray(q) + action
    if space == "joint_abs":
        return np.asarray(action)
    T = fk(q)
    goal = T.copy()
    if space == "cartesian_abs":
        goal[..., :3, 3] = action[..., :3]
        goal[..., :3, :3] = so3_exp(action[..., 3:6]) @ R_REF
    else:
        goal[..., :3, 3] += action[..., :3]
        goal[..., :3, :3] = so3_exp(action[..., 3:6]) @ T[..., :3, :3]
    return ik(q, goal)


def tcp_yaw(R):
    """yaw of the hand x axis in the base xy plane"""
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def wrap90(a):
    """wrap an angle to [-pi/4, pi/4), a square cube's period"""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4
