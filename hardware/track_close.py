# object tracking for the close demos
# usage: python track_close.py --demo_dir <T7>/close/train   (writes <demo_dir>_privileged; skips tracked demos)
import os
import glob
import argparse
import h5py
import numpy as np
import cv2
from objects import points_yaw
from track_pot import find_fruit, box_tracker
from track_stack import load_camera
from robot_mask import RobotMask

LABELS = ["handle", "kcup", "apple", "orange", "banana"]
HIGH = ["handle", "kcup"]
LIVE = ["handle"]
FRUIT = ["apple", "orange", "banana"]
PROMPTS = {"handle": "silver handle", "kcup": "purple cup", "apple": "red apple", "orange": "orange", "banana": "banana"}
COLORS = {"handle": (0, 255, 0), "kcup": (255, 0, 255), "apple": (255, 0, 0), "orange": (255, 140, 0), "banana": (255, 255, 0)}


def first_prompts(seg, rgb, depth, c, labels, threshold=0.3):
    """first-frame prompts {label: ("mask", (H, W) bool) or ("box", (4,))}, None if not found"""
    out = {k: None for k in labels}
    boxes = find_fruit(rgb, depth, c)
    for k in labels:
        if k in FRUIT and boxes.get(k) is not None:
            out[k] = ("box", boxes[k])
    todo = [k for k in labels if out[k] is None]
    if todo:
        masks, scores = seg.detect(rgb, [PROMPTS[k] for k in todo], threshold)
        for i, k in enumerate(todo):
            if scores[i] > 0:
                if k in HIGH:
                    out[k] = ("mask", masks[i])
                else:
                    v, u = np.nonzero(masks[i])
                    out[k] = ("box", np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float32))
    return out


def close_sessions(seg, labels, prompts):
    """EdgeTAM sessions for the found labels, as [(tracker, label indices)]"""
    from perception import CameraTracker
    masked = [k for k in labels if prompts[k] is not None and prompts[k][0] == "mask"]
    boxed = [k for k in labels if prompts[k] is not None and prompts[k][0] == "box"]

    class MaskTracker(CameraTracker):
        def _start(self, image):
            proc = self.seg.edgetam_processor
            self.session = proc.init_video_session(inference_device=self.seg.device, dtype=self.seg.dtype)
            self.session.video_height, self.session.video_width = image.shape[:2]
            self.obj_ids = list(range(len(self.labels)))
            proc.add_inputs_to_inference_session(self.session, frame_idx=self.t, obj_ids=self.obj_ids,
                                                 input_masks=[prompts[k][1].astype(np.float32) for k in self.labels])
            return True

    out = []
    if masked:
        out.append((MaskTracker(seg, masked), [labels.index(k) for k in masked]))
    if boxed:
        out.append((box_tracker(seg, boxed, {k: prompts[k][1] for k in boxed}), [labels.index(k) for k in boxed]))
    return out


class CloseState:
    """object positions fixed from the first measurements; live objects keep updating"""

    def __init__(self, labels, c, correction=None, draw=None, shape=(1080, 1920), static=15, near=0.05, erode=3,
                 live=LIVE, window=3):
        self.labels, self.c, self.correction, self.static, self.near = list(labels), c, correction, static, near
        self.live, self.window = [k for k, label in enumerate(self.labels) if label in live], window
        self.robot = RobotMask(draw, c["T_base_cam"], shape=shape, margin=6) if draw is not None else None
        self.kernel = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
        self.reset()

    def reset(self):
        self.acc = {k: [] for k in range(len(self.labels))}
        self.fixed, self.recent = {}, {k: [] for k in range(len(self.labels))}

    def robot_mask(self, q, shape):
        if self.robot is None or q is None:
            return np.zeros(shape, bool)
        return self.robot(q)

    def measure(self, k, mask, depth):
        c = self.c
        mask = cv2.erode(mask.astype(np.uint8), self.kernel).astype(bool)
        v, u = np.nonzero(mask)
        z = depth[v, u].astype(np.float32) * c["depth_scale"]
        ok = (z > 0.1) & (z < 3.0)
        if ok.sum() < 30:
            return None
        u, v, z = u[ok], v[ok], z[ok]
        if self.labels[k] in HIGH:
            front = z < np.percentile(z, 10) + self.near
            u, v, z = u[front], v[front], z[front]
        cam = np.stack([(u - c["K"][0, 2]) * z / c["K"][0, 0], (v - c["K"][1, 2]) * z / c["K"][1, 1], z], axis=1)
        points = cam @ c["T_base_cam"][:3, :3].T + c["T_base_cam"][:3, 3]
        if self.correction is not None:
            points[:, :2] = points[:, :2] @ self.correction[:, :2].T + self.correction[:, 2]
        p = np.median(points, axis=0)
        if self.labels[k] in FRUIT and p[2] >= 0.15:
            return None
        return np.array([*p, points_yaw(points) if self.labels[k] in FRUIT else 0.0], dtype=np.float32)

    def __call__(self, masks, depth, q=None):
        """masks (n, H, W) bool, depth (H, W), q (7,) or None; returns state (n, 4), NaN until fixed, and fixed (n,)"""
        robot = self.robot_mask(q, masks.shape[1:])
        for k in range(len(self.labels)):
            if k in self.fixed and k not in self.live:
                continue
            m = self.measure(k, masks[k] & ~robot, depth) if masks[k].any() else None
            if k in self.fixed:
                if m is not None:
                    self.recent[k] = (self.recent[k] + [m])[-self.window:]
                    self.fixed[k] = np.array([*np.median(np.array(self.recent[k])[:, :3], axis=0), 0.0], dtype=np.float32)
                continue
            if m is not None:
                self.acc[k].append(m)
            if len(self.acc[k]) >= self.static:
                self.finish(k)
        return self.state()

    def finish(self, k, min_n=1):
        if k not in self.fixed and len(self.acc[k]) >= min_n:
            r = np.array(self.acc[k])
            yaw = r[0, 3] + np.median((r[:, 3] - r[0, 3] + np.pi / 2) % np.pi - np.pi / 2)
            self.fixed[k] = np.array([*np.median(r[:, :3], axis=0), yaw], dtype=np.float32)

    def state(self):
        s = np.full((len(self.labels), 4), np.nan, dtype=np.float32)
        for k, f in self.fixed.items():
            s[k] = f
        return s, np.array([k in self.fixed for k in range(len(self.labels))])


def overlay(frame, masks, labels):
    for i, k in enumerate(labels):
        frame[masks[i]] = (0.45 * frame[masks[i]] + 0.55 * np.array(COLORS[k])).astype(np.uint8)
    return frame


def track(raw_path, out_path, seg, args, cam):
    """track one demo and write its privileged state file"""
    c, corr, draw = cam
    with h5py.File(raw_path, "r") as raw:
        rgb, depth = raw["obs"]["orbbec_img_rgb"], raw["obs"]["orbbec_img_depth"]
        q = raw["obs"]["joint_state"][:, :7]
        n, h, w = rgb.shape[:3]
        prompts = first_prompts(seg, rgb[0], depth[0], c, LABELS)
        trackers = close_sessions(seg, LABELS, prompts)
        est = CloseState(LABELS, c, corr, draw, shape=(h, w), static=args.static, live=args.live)
        per_frame = np.full((n, len(LABELS), 4), np.nan, dtype=np.float32)
        for t in range(n if args.live else min(args.max_frames, n)):
            m = np.zeros((len(LABELS), h, w), bool)
            s = np.zeros(len(LABELS), np.float32)
            for tracker, idx in trackers:
                m[idx], s[idx] = tracker.step(rgb[t])
            m[s < args.vis_threshold] = False
            per_frame[t], fixed = est(m, depth[t], q[t])
            if t == 0:
                view = overlay(rgb[0].copy(), m, LABELS)
            if not args.live and fixed[[k for k in range(len(LABELS)) if prompts[LABELS[k]] is not None]].all():
                break
        for k in range(len(LABELS)):
            est.finish(k)
        state, fixed = est.state()
        object_state = np.repeat(state[None], n, axis=0)
        for k in [LABELS.index(label) for label in args.live]:
            seen = np.flatnonzero(np.isfinite(per_frame[:, k, 0]))
            if len(seen):
                object_state[:, k] = per_frame[:, k]
                object_state[:seen[0], k] = per_frame[seen[0], k]
        counts = [len(est.acc[k]) if k not in est.fixed else args.static for k in range(len(LABELS))]
    missing = [k for i, k in enumerate(LABELS) if not fixed[i]]
    cv2.imwrite(out_path.replace(".hdf5", "_first.jpg"), cv2.cvtColor(cv2.resize(view, (w // 2, h // 2)), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    tmp_path = out_path + ".tmp"
    with h5py.File(tmp_path, "w") as out:
        out["object_state"] = object_state
        out["object_visible"] = np.repeat(fixed[None], n, axis=0)
        out["object_held"] = np.zeros((n, len(LABELS)), bool)
        out.attrs["source"] = os.path.realpath(raw_path)
        out.attrs["cameras"] = ["orbbec"]
        out.attrs["tracked"] = [k for k in LABELS if prompts[k] is not None]
        out.attrs["labels"] = LABELS
        out.attrs["synthetic"] = []
        # not found on the first frame
        out.attrs["missing"] = missing
        out.attrs["static"] = args.static
        out.attrs["live"] = list(args.live)
    os.replace(tmp_path, out_path)
    return state, missing, prompts, counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", default="/media/collab/T7/CoData/inner-product/inner_product/close/train")
    parser.add_argument("--out_dir", default=None, help="default: <demo_dir>_privileged")
    parser.add_argument("--calibration", default="calibration/orbbec_stack.yaml")
    parser.add_argument("--static", type=int, default=15, help="each object fixed at the median of its first N measurements")
    parser.add_argument("--max_frames", type=int, default=60, help="without --live objects")
    parser.add_argument("--live", nargs="*", default=LIVE, help="objects updated every frame once fixed ('--live' alone: none)")
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from perception import Segmenter
    out_dir = args.out_dir or args.demo_dir.rstrip("/") + "_privileged"
    os.makedirs(out_dir, exist_ok=True)
    cam = load_camera(args.calibration)
    seg = Segmenter(device=args.device)
    paths = sorted(glob.glob(f"{args.demo_dir}/demo_*.hdf5"), key=lambda p: int(os.path.basename(p)[5:-5]))
    for p in paths:
        out_path = os.path.join(out_dir, os.path.basename(p))
        if os.path.exists(out_path):
            continue
        state, missing, prompts, counts = track(p, out_path, seg, args, cam)
        how = ", ".join(f"{k} {prompts[k][0] if prompts[k] else '-'} {counts[i]}" for i, k in enumerate(LABELS))
        print(f"{os.path.basename(p)}: {how}" + (f"; missing {missing}" if missing else ""))
        print("   " + ", ".join(f"{k} {np.round(100 * state[i, :3], 1)}" for i, k in enumerate(LABELS)), flush=True)
    print("TRACK DONE")
