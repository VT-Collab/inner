import os
import glob
import pickle
import argparse
import h5py
import numpy as np
from objects import mask_points, points_yaw, mean_yaw
from config import n_joints, n_objects, object_dim, dims
from kinematics import fk, arm_state, arm_action, so3_exp, so3_log, R_REF, tcp_yaw, wrap90, STATE_SPACES, ACTION_SPACES


def fill_gaps(x):
    """fill steps where an object was not seen (NaN) with its last seen value; x is (T, n_objects, d)"""
    steps = np.arange(len(x))
    for k in range(x.shape[1]):
        seen = ~np.isnan(x[:, k]).any(-1)
        if not seen.any():
            x[:, k] = 0.0
            continue
        last = np.maximum.accumulate(np.where(seen, steps, -1))
        x[:, k] = x[np.where(last < 0, steps[seen][0], last), k]
    return x


def object_yaw(priv, raw, calib):
    """yaw of every object per step, (T, n_objects), averaged over the cameras that see it"""
    yaws = []
    for cam in priv.attrs["cameras"]:
        c = calib[cam]
        masks, depth = priv[cam]["masks"], raw["obs"][f"{cam}_img_depth"]
        yaw = np.full(masks.shape[:2], np.nan)
        for t in range(len(yaw)):
            m, d = masks[t], depth[t]
            for k in np.flatnonzero(m.any(axis=(1, 2))):
                points = mask_points(m[k], d, c["K"], c["depth_scale"], c["T_base_cam"])
                if points is not None:
                    yaw[t, k] = points_yaw(points)
        yaws.append(yaw)
    return mean_yaw(np.stack(yaws))


def object_features(priv, raw, calib):
    """object state per step, (T, n_objects, object_dim) = (x, y, z, yaw)"""
    if "object_state" in priv:
        objects = np.asarray(priv["object_state"], dtype=np.float32)
        assert objects.shape[1:] == (len(priv.attrs.get("labels", range(n_objects))), object_dim), \
            f"object_state has shape {objects.shape}"
        return fill_gaps(objects)
    assert object_dim == 4, "without object_state, each object is (x, y, z, yaw)"
    assert "masks" in priv[priv.attrs["cameras"][0]], "yaw needs the masks (postprocess with save_masks: True)"
    pos = fill_gaps(np.asarray(priv["object_pos"], dtype=np.float32))
    yaw = fill_gaps(object_yaw(priv, raw, calib)[..., None])[..., 0]
    yaw = np.unwrap(yaw, period=np.pi, axis=0)
    return np.concatenate([pos, yaw[..., None]], axis=-1).astype(np.float32)


def raw_path(priv_path, priv, raw_dir):
    """path of the raw demo (joint states, gripper) a privileged demo was made from"""
    if "obs" in priv:
        return priv_path
    source = priv.attrs.get("source", "")
    if raw_dir is not None:
        return os.path.join(raw_dir, os.path.basename(source or priv_path))
    if os.path.exists(source):
        return source
    raise FileNotFoundError(f"raw demo {source} not found, pass --raw_dir")


def step_times(raw):
    """step times of a demo in seconds"""
    if "time" in raw["obs"]:
        return np.asarray(raw["obs"]["time"], dtype=np.float64)
    fps = float(raw.attrs.get("record_freq", raw.attrs.get("fps", 20)))
    return np.arange(int(raw.attrs["num_samples"])) / fps


def resample(t, x, grid, hold=False):
    """x at the grid times, linear or held with hold=True"""
    if hold:
        return x[np.searchsorted(t, grid, side="right") - 1]
    return np.stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])], axis=-1)


def process_demo(priv_path, raw_dir, hz, n_phases, calib, state_space="joint", action_space="joint", max_gap=0.5, keep=None,
                 target_steps=1, gripper_ahead=False, no_yaw=None, gripper_lead=0, relative=None, obs_window=1, chunk=1,
                 chunk_relative=False, label_from_plan=False, rot_relative=False, yaw_relative=None):
    """(state, action) arrays of one demo on a 1/hz grid, split at recording gaps"""
    with h5py.File(priv_path, "r") as priv, h5py.File(raw_path(priv_path, priv, raw_dir), "r") as raw:
        objects = object_features(priv, raw, calib)
        objects = objects if keep is None else objects[:, keep]
        if yaw_relative is not None and yaw_relative.any():
            qy = np.asarray(raw["obs"]["joint_state"], dtype=np.float64)[:, :n_joints]
            # unwrapped here, wrapped after resampling (no interpolation across the jump)
            d = objects[:, yaw_relative, 3] - tcp_yaw(fk(qy)[:, :3, :3])[:, None]
            objects[:, yaw_relative, 3] = np.unwrap(d, period=np.pi / 2, axis=0)
        cols = [j * objects.shape[2] + d for j in range(objects.shape[1]) for d in range(objects.shape[2])
                if no_yaw is None or not (no_yaw[j] and d == 3)]
        objects = objects.reshape(len(objects), -1)[:, cols]
        yaw_cols = [cols.index(j * 4 + 3) for j in np.flatnonzero(yaw_relative)] if yaw_relative is not None else []
        q = np.asarray(raw["obs"]["joint_state"], dtype=np.float32)[:, :n_joints]
        gripper = np.asarray(raw["obs"]["gripper_state"], dtype=np.float32).reshape(-1, 1)
        gripper_action = np.asarray(raw["actions"]["gripper_action"], dtype=np.float32).reshape(-1, 1)
        q_plan = np.asarray(raw["plan"]["joint_state"], dtype=np.float32)[:, :n_joints] if label_from_plan and "plan" in raw else None
        t = step_times(raw)
    assert len(objects) == len(q) == len(t), f"{priv_path}: {len(objects)} tracked steps, {len(q)} robot steps"

    states, actions = [], []
    cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
    for seg in np.split(np.arange(len(t)), cuts):
        ts = t[seg]
        for phase in range(n_phases):
            grid = np.arange(ts[0] + phase / (n_phases * hz), ts[-1] + 1e-9, 1.0 / hz)
            if len(grid) < 2:
                continue
            qg = resample(ts, q[seg], grid)
            obj = resample(ts, objects[seg], grid[:-1])
            obj[:, yaw_cols] = wrap90(obj[:, yaw_cols])
            parts = [arm_state(qg[:-1], state_space), resample(ts, gripper[seg], grid[:-1], hold=True), obj]
            if relative is not None:
                tcp = fk(qg[:-1])[:, :3, 3]
                parts += [obj[:, c:c + 3] - tcp for c in relative]
            S = np.concatenate(parts, axis=-1)
            idx = np.clip(np.arange(len(S))[:, None] - np.arange(obs_window - 1, -1, -1)[None], 0, len(S) - 1)
            states.append(S[idx].reshape(len(S), -1))
            ahead = np.minimum(np.arange(1, len(qg)) + target_steps - 1, len(qg) - 1)
            grip = resample(ts, gripper_action[seg], grid, hold=True)
            lead = target_steps if gripper_ahead else gripper_lead
            g_idx = np.minimum(np.arange(len(qg) - 1) + lead, len(qg) - 1)
            qt = qg if q_plan is None else resample(ts, q_plan[seg], grid)
            A = np.concatenate([arm_action(qg[:-1], qt[ahead], action_space), grip[g_idx]], axis=-1)
            idx = np.clip(np.arange(len(A))[:, None] + np.arange(chunk)[None], 0, len(A) - 1)
            C = A[idx]
            if chunk_relative:
                # target positions from the tcp at this step, finer than absolute after normalizing
                C[:, :, :3] -= fk(qg[:-1])[:, None, :3, 3]
            if rot_relative:
                Rc = fk(qg[:-1])[:, None, :3, :3]
                C[:, :, 3:6] = so3_log(so3_exp(C[:, :, 3:6]) @ R_REF @ np.swapaxes(Rc, -1, -2))
            actions.append(C.reshape(len(A), -1))
    return np.concatenate(states), np.concatenate(actions), q[0], np.diff(t)


def min_max(x):
    """per-dimension min and max, a constant dimension gets a range of 1"""
    lo, hi = x.min(axis=0), x.max(axis=0)
    hi = np.where(hi - lo < 1e-6, lo + 1.0, hi)
    return lo, hi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--priv_dir", required=True)
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--calibration", default="/workspace/src/config/calibration/cameras.yaml")
    parser.add_argument("--hz", type=float, default=10.0)
    # grids per demo, offset by 1 / (n_phases * hz)
    parser.add_argument("--n_phases", type=int, default=2)
    parser.add_argument("--n_demos", type=int, default=0)
    parser.add_argument("--state_space", choices=list(STATE_SPACES), default="joint")
    parser.add_argument("--action_space", choices=list(ACTION_SPACES), default="joint")
    # _abs action spaces: the target is this many grid steps ahead
    parser.add_argument("--target_steps", type=int, default=1)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--gripper_ahead", action="store_true")
    parser.add_argument("--no_yaw", nargs="+", default=[])
    parser.add_argument("--gripper_lead", type=int, default=0)
    parser.add_argument("--relative", nargs="+", default=[])
    # states stacked over this many steps (this one last), actions over this many (this one first)
    parser.add_argument("--obs_window", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--chunk_relative", action="store_true")
    parser.add_argument("--rot_relative", action="store_true")
    parser.add_argument("--yaw_relative", nargs="+", default=[])
    parser.add_argument("--label_from_plan", action="store_true")
    parser.add_argument("--out", default="datasets/dataset.pkl")
    args = parser.parse_args()

    assert args.target_steps == 1 or args.action_space.endswith("_abs"), "--target_steps needs an _abs action space"
    paths = sorted(glob.glob(os.path.join(args.priv_dir, "demo*.hdf5")),
                   key=lambda p: int(os.path.basename(p)[5:-5]))
    if args.n_demos > 0:
        paths = paths[:args.n_demos]
    assert len(paths) > 0, f"no demos in {args.priv_dir}"

    with h5py.File(paths[0], "r") as priv:
        calib = None
        if "object_state" not in priv:
            from perception import load_calibration
            calib = load_calibration(args.calibration, list(priv.attrs["cameras"]))
        labels = [str(label) for label in priv.attrs["labels"]]
    keep = None
    if args.objects is not None:
        assert set(args.objects) <= set(labels), f"--objects {args.objects}, the demos have {labels}"
        keep, labels = [labels.index(o) for o in args.objects], list(args.objects)

    relative_cols = None
    if args.relative:
        widths = [3 if k in args.no_yaw else 4 for k in labels]
        relative_cols = [int(sum(widths[:labels.index(k)])) for k in args.relative]
    states, actions, starts, dts = [], [], [], []
    for path in paths:
        s, a, q0, dt = process_demo(path, args.raw_dir, args.hz, args.n_phases, calib, args.state_space, args.action_space, keep=keep,
                                    target_steps=args.target_steps, gripper_ahead=args.gripper_ahead,
                                    no_yaw=np.array([k in args.no_yaw for k in labels]), gripper_lead=args.gripper_lead,
                                    relative=relative_cols, obs_window=args.obs_window, chunk=args.chunk,
                                    chunk_relative=args.chunk_relative, label_from_plan=args.label_from_plan,
                                    rot_relative=args.rot_relative,
                                    yaw_relative=np.array([k in args.yaw_relative for k in labels]))
        states.append(s)
        actions.append(a)
        starts.append(q0)
        dts.append(dt)
    states, actions, dts = np.concatenate(states), np.concatenate(actions), np.concatenate(dts)
    dts = dts[dts < 0.5]
    print(f"recorded at {1 / np.median(dts):.1f} Hz (median), step time {1000 * dts.min():.0f}-{1000 * dts.max():.0f} ms")

    state_min, state_max = min_max(states)
    action_min, action_max = min_max(actions)
    stats = {
        "state_min": state_min, "state_max": state_max,
        "action_min": action_min, "action_max": action_max,
        # the policy starts from the average first joint position of the demos
        "home": np.mean(starts, axis=0),
        "control_hz": args.hz,
        "labels": labels,
        "state_space": args.state_space,
        "action_space": args.action_space,
        "target_steps": args.target_steps,
        "gripper_ahead": args.gripper_ahead,
        "no_yaw": [k for k in labels if k in args.no_yaw],
        "gripper_lead": args.target_steps if args.gripper_ahead else args.gripper_lead,
        "relative": list(args.relative),
        "obs_window": args.obs_window,
        "chunk": args.chunk,
        "chunk_relative": args.chunk_relative,
        "rot_relative": args.rot_relative,
        "yaw_relative": [k for k in labels if k in args.yaw_relative],
        "label_from_plan": args.label_from_plan,
    }
    assert (states.shape[1], actions.shape[1]) == dims(stats)
    data = np.concatenate([(states - state_min) / (state_max - state_min) * 2 - 1,
                           (actions - action_min) / (action_max - action_min) * 2 - 1], axis=-1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pickle.dump({"data": data.astype(np.float32), "stats": stats}, open(args.out, "wb"))
    print(f"{len(paths)} demos, {len(data)} datapoints at {args.hz:g} Hz ({args.n_phases} phases) -> {args.out}")
