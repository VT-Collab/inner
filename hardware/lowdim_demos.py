import os
import glob
import argparse
import h5py
import numpy as np
from track_pot import synthetic_objects

# synthetic_objects(rng) rows
FRUIT = ["apple", "orange", "banana"]

# panda joint 1 limits (rad)
Q1_LIMITS = (-2.8973, 2.8973)


def rotate_z(points, theta):
    """rotate (..., 3) points about the robot base z axis"""
    c, s = np.cos(theta), np.sin(theta)
    return points @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def write_demo(path, demo, attrs):
    with h5py.File(path, "w") as f:
        for key, value in demo.items():
            f.create_dataset(key, data=value)
        for key, value in attrs.items():
            f.attrs[key] = value


def extract(priv_path, raw_dir=None):
    """robot data and object states of a demo, without the images"""
    with h5py.File(priv_path, "r") as priv:
        source = priv.attrs["source"]
    if raw_dir is not None:
        source = os.path.join(raw_dir, os.path.basename(source))
    with h5py.File(priv_path, "r") as priv, h5py.File(source, "r") as raw:
        demo = {key: raw[key][:] for key in ("obs/joint_state", "obs/gripper_state", "obs/ee_state",
                                             "actions/gripper_action", "obs/time") if key in raw}
        demo["object_state"] = priv["object_state"][:]
        demo["object_visible"] = priv["object_visible"][:]
        attrs = {key: raw.attrs[key] for key in ("fps", "record_freq", "num_samples", "task_description") if key in raw.attrs}
        attrs.update(labels=priv.attrs["labels"], synthetic_objects=priv.attrs["synthetic"],
                     source=priv.attrs["source"], synthetic=False)
        if "missing" in priv.attrs:
            attrs["missing"] = [str(k) for k in priv.attrs["missing"]]
    return demo, attrs


def extract_lowdim(path):
    """load a real demo written by this script"""
    with h5py.File(path, "r") as f:
        demo = {}
        f.visititems(lambda name, obj: demo.__setitem__(name, obj[:]) if isinstance(obj, h5py.Dataset) else None)
        attrs = {key: f.attrs[key] for key in ("fps", "record_freq", "num_samples", "task_description", "labels",
                                               "synthetic_objects", "source", "synthetic") if key in f.attrs}
    return demo, attrs


def borrow_missing(real):
    """give objects missing from a real demo a pose borrowed from another demo"""
    labels = [str(k) for k in real[0][1]["labels"]]

    def first(state, k):
        seen = np.flatnonzero(np.isfinite(state[:, k, 0]))
        return state[seen[0], k] if len(seen) else None

    for i, (demo, attrs) in enumerate(real):
        for name in attrs.get("missing", []):
            k = labels.index(name)
            others = [first(demo["object_state"], j) for j in range(len(labels)) if j != k]
            others = [o[:2] for o in others if o is not None]
            best = None
            for j, (donor, donor_attrs) in enumerate(real):
                p = None if j == i or name in donor_attrs.get("missing", []) else first(donor["object_state"], k)
                if p is None:
                    continue
                gap = min(np.linalg.norm(p[:2] - o) for o in others)
                if best is None or gap > best[0]:
                    best = (gap, j, p)
            if best is None:
                continue
            demo["object_state"][:, k] = best[2]
            demo["object_visible"][:, k] = True
            attrs["borrowed"] = list(attrs.get("borrowed", [])) + [name]
            print(f"{attrs['base_demo']}: {name} from {real[best[1]][1]['base_demo']}, {100 * best[0]:.0f} cm from the nearest object")


def rotate(demo, attrs, theta, rng):
    """copy of the demo with the scene turned by theta about the base z axis"""
    demo = {key: value.copy() for key, value in demo.items()}
    demo["obs/joint_state"][:, 0] += theta
    demo["obs/ee_state"][:, :3, :] = rotate_z(demo["obs/ee_state"][:, :3, :].transpose(0, 2, 1), theta).transpose(0, 2, 1)
    objects = demo["object_state"]
    objects[..., :3] = rotate_z(objects[..., :3], theta)
    objects[..., 3] += theta
    labels = [str(k) for k in attrs["labels"]]
    synthetic = [labels.index(str(k)) for k in attrs["synthetic_objects"]]
    # drawn even when unused, so the rotations stay the same with or without synthetic objects
    fake = synthetic_objects(rng)
    if synthetic:
        objects[:, synthetic] = fake[[FRUIT.index(labels[k]) for k in synthetic]]
    return demo, dict(attrs, synthetic=True, rotation=theta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--priv_dir", default="/media/benjamin/T7/inner-product/inner_product/push/train_privileged")
    parser.add_argument("--out_dir", default="demos/push")
    parser.add_argument("--from_lowdim", default=None)
    parser.add_argument("--raw_dir", default=None, help="raw demos, if not at the path saved in the privileged demos")
    parser.add_argument("--n_total", type=int, default=50)
    parser.add_argument("--min_deg", type=float, default=5.0)
    parser.add_argument("--max_deg", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src = args.from_lowdim or args.priv_dir
    paths = sorted(glob.glob(os.path.join(src, "demo_*.hdf5")), key=lambda p: int(os.path.basename(p)[5:-5]))
    if args.from_lowdim:
        paths = [p for p in paths if not h5py.File(p, "r").attrs.get("synthetic", False)]
    assert len(paths) > 0, f"no tracked demos in {src}"
    assert os.path.abspath(args.out_dir) != os.path.abspath(src), "--out_dir would overwrite the source demos"
    os.makedirs(args.out_dir, exist_ok=True)

    real = []
    for i, path in enumerate(paths):
        demo, attrs = extract_lowdim(path) if args.from_lowdim else extract(path, args.raw_dir)
        attrs["base_demo"] = os.path.basename(path)
        real.append((demo, attrs))
    borrow_missing(real)
    for i, (demo, attrs) in enumerate(real):
        write_demo(os.path.join(args.out_dir, f"demo_{i}.hdf5"), demo, attrs)

    rng = np.random.default_rng(args.seed)
    first = str(real[0][1]["labels"][0])
    pot_xy = []
    for i in range(len(real), args.n_total):
        demo, attrs = real[(i - len(real)) % len(real)]
        theta = np.deg2rad(rng.uniform(args.min_deg, args.max_deg)) * rng.choice([-1, 1])
        demo, attrs = rotate(demo, attrs, theta, rng)
        q1 = demo["obs/joint_state"][:, 0]
        assert Q1_LIMITS[0] < q1.min() and q1.max() < Q1_LIMITS[1], f"joint 1 out of limits for demo {i}"
        write_demo(os.path.join(args.out_dir, f"demo_{i}.hdf5"), demo, attrs)
        pot_xy.append(demo["object_state"][:, 0, :2])
        print(f"demo_{i}: {attrs['base_demo']} turned {np.rad2deg(theta):+.1f} deg")

    real_xy = np.concatenate([d["object_state"][:, 0, :2] for d, _ in real])
    print(f"{len(real)} real + {max(args.n_total - len(real), 0)} rotated demos -> {args.out_dir}")
    print(f"{first} x, y range: real", real_xy.min(0).round(2), real_xy.max(0).round(2), end="")
    if pot_xy:
        pot_xy = np.concatenate(pot_xy)
        print(", rotated", pot_xy.min(0).round(2), pot_xy.max(0).round(2))
