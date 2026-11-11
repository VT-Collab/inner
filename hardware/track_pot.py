import os
import glob
import argparse
import yaml
import h5py
import numpy as np
import cv2
from tqdm import tqdm
from objects import mask_points, points_yaw, wrap
from process_demos import fill_gaps


# opencv hue ranges (0-180)
FRUIT_HUES = {"apple": [(0, 8), (170, 180)], "orange": [(10, 20)], "banana": [(22, 35)]}


def above_table(depth, c, z_lo, z_hi):
    """(H, W) bool mask of pixels between z_lo and z_hi above the table, away from the robot base"""
    h, w = depth.shape
    v, u = np.mgrid[:h, :w]
    z = depth.astype(np.float32) * c["depth_scale"]
    cam = np.stack([(u - c["K"][0, 2]) * z / c["K"][0, 0], (v - c["K"][1, 2]) * z / c["K"][1, 1], z], axis=-1)
    base = cam @ c["T_base_cam"][:3, :3].T + c["T_base_cam"][:3, 3]
    return (z > 0.1) & (base[..., 2] > z_lo) & (base[..., 2] < z_hi) & (base[..., 0] > 0.25)


def largest_blob(mask, min_pixels):
    """xywh and area of the largest blob of a mask, or None"""
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return stats[k] if stats[k, cv2.CC_STAT_AREA] >= min_pixels else None


def find_pot(rgb, depth, c, min_pixels=500):
    """xyxy pixel box of the pot on the first frame, or None"""
    h, w = depth.shape
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white = (hsv[..., 1] < 40) & (hsv[..., 2] > 190)
    blob = largest_blob(white & above_table(depth, c, 0.02, 0.25), min_pixels)
    if blob is None:
        return None
    x, y, bw, bh, _ = blob
    return np.array([max(x - 0.15 * bw, 0), max(y - 0.1 * bh, 0),
                     min(x + 1.15 * bw, w - 1), min(y + 1.5 * bh, h - 1)], dtype=np.float32)


def find_fruit(rgb, depth, c, min_pixels=300, exclude=None):
    """{label: xyxy box or None} for apple, orange, banana on the first frame; exclude is an (H, W) bool mask to skip"""
    h, w = depth.shape
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colored = (hsv[..., 1] > 120) & (hsv[..., 2] > 80) & above_table(depth, c, 0.005, 0.15)
    if exclude is not None:
        colored &= ~exclude
    boxes = {}
    for label, ranges in FRUIT_HUES.items():
        hue = np.zeros_like(colored)
        for lo, hi in ranges:
            hue |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
        blob = largest_blob(colored & hue, min_pixels)
        if blob is None:
            boxes[label] = None
            continue
        x, y, bw, bh, _ = blob
        boxes[label] = np.array([max(x - 0.1 * bw, 0), max(y - 0.1 * bh, 0),
                                 min(x + 1.1 * bw, w - 1), min(y + 1.1 * bh, h - 1)], dtype=np.float32)
    return boxes


LABELS = ["pot", "apple", "orange", "banana"]
COLORS = {"pot": (255, 153, 0), "apple": (0, 200, 255), "orange": (180, 0, 255), "banana": (0, 90, 255)}


def box_tracker(seg, labels, boxes):
    """EdgeTAM tracker prompted with first-frame boxes; reprompt(k, mask) re-prompts object k with a mask"""
    from perception import CameraTracker

    class BoxTracker(CameraTracker):
        def _start(self, image):
            proc = self.seg.edgetam_processor
            self.session = proc.init_video_session(inference_device=self.seg.device, dtype=self.seg.dtype)
            self.session.video_height, self.session.video_width = image.shape[:2]
            self.obj_ids = list(range(len(self.labels)))
            proc.add_inputs_to_inference_session(self.session, frame_idx=self.t, obj_ids=self.obj_ids,
                                                 input_boxes=[[boxes[k].tolist() for k in self.labels]],
                                                 original_size=image.shape[:2])
            return True

        def reprompt(self, k, mask):
            self.seg.edgetam_processor.add_inputs_to_inference_session(
                self.session, frame_idx=self.t, obj_ids=[k], input_masks=[mask.astype(np.float32)], original_size=mask.shape)

    return BoxTracker(seg, labels)


class SpillGuard:
    """re-prompts with the last clean mask when more than max_high of the mask's points are above z (m)"""

    def __init__(self, k, max_high=0.05, z=0.08):
        self.k, self.max_high, self.z, self.last_clean = k, max_high, z, None

    def __call__(self, tracker, mask, depth, c):
        p = mask_points(mask, depth, c["K"], c["depth_scale"], c["T_base_cam"], erode_px=0) if mask.any() else None
        high = 0.0 if p is None else float(np.mean(p[:, 2] > self.z))
        if high > self.max_high and self.last_clean is not None:
            tracker.reprompt(self.k, self.last_clean)
        if high <= self.max_high and mask.any():
            self.last_clean = mask.copy()
        return high


def retrack_pot(raw_path, old_path, out_path, edgetam, args, c):
    """rewrite a privileged demo with the pot tracked again using SpillGuard"""
    with h5py.File(raw_path, "r") as raw, h5py.File(old_path, "r") as old:
        rgb, depth = raw["obs"]["orbbec_img_rgb"], raw["obs"]["orbbec_img_depth"]
        boxes = {"pot": find_pot(rgb[0], depth[0], c)}
        tracker, guard = box_tracker(edgetam, ["pot"], boxes), SpillGuard(0)
        tmp_path = out_path + ".tmp"
        with h5py.File(tmp_path, "w") as out:
            for key, value in old.attrs.items():
                out.attrs[key] = value
            out.attrs["spill_guard"] = True
            src = old["orbbec/masks"]
            pot = LABELS.index("pot") if src.shape[1] == len(LABELS) else list(old.attrs["tracked"]).index("pot")
            masks = out.create_dataset("orbbec/masks", shape=src.shape, dtype=bool, chunks=src.chunks, compression="gzip")
            scores = old["orbbec/scores"][:]
            for t in tqdm(range(len(rgb)), desc=os.path.basename(raw_path), leave=False):
                m, s = tracker.step(rgb[t])
                m[s < args.vis_threshold] = False
                guard(tracker, m[0], depth[t], c)
                frame = src[t]
                frame[pot], scores[t, pot] = m[0], s[0]
                masks[t] = frame
            out.create_dataset("orbbec/scores", data=scores)
    os.replace(tmp_path, out_path)


def track(raw_path, out_path, edgetam, args, c):
    """track the not yet tracked objects in one demo; writes orbbec/masks, orbbec/scores and an overlay video"""
    from perception import CameraTracker

    class BoxTracker(CameraTracker):
        def _start(self, image):
            proc = self.seg.edgetam_processor
            self.session = proc.init_video_session(inference_device=self.seg.device, dtype=self.seg.dtype)
            self.session.video_height, self.session.video_width = image.shape[:2]
            self.obj_ids = list(range(len(self.labels)))
            proc.add_inputs_to_inference_session(self.session, frame_idx=self.t, obj_ids=self.obj_ids,
                                                 input_boxes=[[boxes[k].tolist() for k in self.labels]],
                                                 original_size=image.shape[:2])
            return True

    old = h5py.File(out_path, "r") if os.path.exists(out_path) else None
    done = [str(k) for k in old.attrs.get("tracked", ["pot"])] if old is not None else []
    with h5py.File(raw_path, "r") as raw:
        rgb, depth = raw["obs"]["orbbec_img_rgb"], raw["obs"]["orbbec_img_depth"]
        n, h, w = rgb.shape[:3]
        boxes = find_fruit(rgb[0], depth[0], c)
        boxes["pot"] = find_pot(rgb[0], depth[0], c)
        todo = [k for k in LABELS if k not in done and boxes[k] is not None]
        for k in LABELS:
            if k not in done and boxes[k] is None:
                print(f"[!] no {k} found on the first frame of {raw_path}")
        tracker = BoxTracker(edgetam, todo) if todo else None
        video = cv2.VideoWriter(out_path.replace(".hdf5", "_orbbec.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                float(raw.attrs.get("fps", 20)), (w // 2, h // 2))
        tmp_path = out_path + ".tmp"
        with h5py.File(tmp_path, "w") as out:
            out.attrs["source"] = os.path.realpath(raw_path)
            out.attrs["cameras"] = ["orbbec"]
            out.attrs["tracked"] = done + todo
            first = np.full((len(LABELS), 4), np.nan, dtype=np.float32)
            if old is not None:
                first[[LABELS.index(k) for k in done]] = old.attrs.get("first_boxes", old.attrs.get("first_box"))
            for k in todo:
                first[LABELS.index(k)] = boxes[k]
            out.attrs["first_boxes"] = first[[LABELS.index(k) for k in done + todo]]
            masks = out.create_dataset("orbbec/masks", shape=(n, len(LABELS), h, w), dtype=bool,
                                       chunks=(1, 1, h, w), compression="gzip")
            scores = np.zeros((n, len(LABELS)), dtype=np.float32)
            for t in tqdm(range(n), desc=os.path.basename(raw_path), leave=False):
                image = rgb[t]
                m = np.zeros((len(LABELS), h, w), dtype=bool)
                for i, k in enumerate(done):
                    m[LABELS.index(k)] = old["orbbec/masks"][t, i]
                    scores[t, LABELS.index(k)] = old["orbbec/scores"][t, i]
                if tracker is not None:
                    mt, st = tracker.step(image)
                    mt[st < args.vis_threshold] = False
                    for i, k in enumerate(todo):
                        m[LABELS.index(k)], scores[t, LABELS.index(k)] = mt[i], st[i]
                masks[t] = m
                frame = image.copy()
                for i, k in enumerate(LABELS):
                    frame[m[i]] = (0.5 * frame[m[i]] + 0.5 * np.array(COLORS[k])).astype(np.uint8)
                video.write(cv2.cvtColor(cv2.resize(frame, (w // 2, h // 2)), cv2.COLOR_RGB2BGR))
            out.create_dataset("orbbec/scores", data=scores)
        video.release()
    if old is not None:
        old.close()
    os.replace(tmp_path, out_path)


def object_state(priv, raw, c, rng, max_step=0.03):
    """(x, y, z, yaw) per object in the robot base frame from saved masks; max_step (m) is the largest fruit jump per step"""
    masks, depth = priv["orbbec"]["masks"], raw["obs"]["orbbec_img_depth"]
    tracked = [str(k) for k in priv.attrs.get("tracked", ["pot"])]
    n = len(masks)
    state = np.full((n, len(LABELS), 4), np.nan, dtype=np.float32)
    channels = {k: (LABELS.index(k) if masks.shape[1] == len(LABELS) else i) for i, k in enumerate(tracked)}
    for k, ch in channels.items():
        obj = state[:, LABELS.index(k)]
        area = np.array([masks[t, ch].sum() for t in range(n)])
        if k == "pot":
            ok = (area > 0) & (area <= 2 * np.median(area[area > 0])) if (area > 0).any() else area > 0
        else:
            ok = (area > 0.5 * area[0]) & (area < 1.5 * area[0])
        last = None
        for t in np.flatnonzero(ok):
            points = mask_points(masks[t, ch], depth[t], c["K"], c["depth_scale"], c["T_base_cam"])
            if points is not None and "xy_correction" in c:
                A = c["xy_correction"]
                points[:, :2] = points[:, :2] @ A[:, :2].T + A[:, 2]
            # a good mask sits near the table
            if points is None or np.median(points[:, 2]) >= 0.08:
                continue
            if k == "pot" and np.mean(points[:, 2] >= 0.08) > 0.05:
                continue
            p = np.median(points, axis=0)
            if k != "pot" and last is not None and np.linalg.norm(p[:2] - last[:2]) > max_step:
                continue
            obj[t, :3], obj[t, 3], last = p, points_yaw(points), p
    visible = ~np.isnan(state[..., 0])
    for t in range(1, n):
        state[t, ~visible[t]] = state[t - 1, ~visible[t]]
    # yaw is mod pi
    yaw = fill_gaps(state[..., 3:].copy())[..., 0]
    state[..., 3] = np.where(np.isnan(state[..., 3]), np.nan, np.unwrap(yaw, period=np.pi, axis=0))

    synthetic = [k for k in LABELS if k not in tracked]
    if synthetic:
        idx = [LABELS.index(k) for k in synthetic]
        state[:, idx] = synthetic_objects(rng)[[i - 1 for i in idx]]
        visible[:, idx] = True
    return state, visible, synthetic


def synthetic_objects(rng):
    """fixed random poses on the table for apple, orange, banana"""
    return np.stack([rng.uniform(0.35, 0.8, 3), rng.uniform(-0.3, 0.3, 3),
                     np.full(3, 0.04), wrap(rng.uniform(-np.pi, np.pi, 3))], axis=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", default="/media/benjamin/T7/inner-product/inner_product/push/train")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--calibration", default="calibration/orbbec.yaml")
    parser.add_argument("--edgetam_model", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0, help="for the synthetic objects")
    parser.add_argument("--device", default="cuda")
    # recompute object_state from saved masks only
    parser.add_argument("--states_only", action="store_true")
    # re-track the pot with SpillGuard from --priv_in into --out_dir
    parser.add_argument("--retrack_pot", action="store_true")
    parser.add_argument("--priv_in", default=None, help="default: <demo_dir>_privileged")
    args = parser.parse_args()

    from perception import load_calibration
    c = load_calibration(args.calibration, ["orbbec"])["orbbec"]
    correction = yaml.safe_load(open(args.calibration))["orbbec"].get("xy_correction")
    if correction is not None:
        c["xy_correction"] = np.array(correction, dtype=np.float32)
    out_dir = args.out_dir or args.demo_dir.rstrip("/") + "_privileged"
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.demo_dir, "demo_*.hdf5")), key=lambda p: int(os.path.basename(p)[5:-5]))

    def untracked(p):
        q = os.path.join(out_dir, os.path.basename(p))
        if not os.path.exists(q):
            return True
        with h5py.File(q, "r") as f:
            return set(LABELS) - set(str(k) for k in f.attrs.get("tracked", ["pot"]))
    todo = [p for p in paths if untracked(p)]
    if args.retrack_pot:
        priv_in = args.priv_in or args.demo_dir.rstrip("/") + "_privileged"
        assert os.path.abspath(priv_in) != os.path.abspath(out_dir), "--out_dir would overwrite --priv_in"
        todo = []
        import torch
        from types import SimpleNamespace
        from transformers import EdgeTamVideoModel, Sam2VideoProcessor
        dtype = torch.bfloat16
        edgetam = SimpleNamespace(
            device=args.device, dtype=dtype,
            edgetam=EdgeTamVideoModel.from_pretrained(args.edgetam_model, dtype=dtype).to(args.device).eval(),
            edgetam_processor=Sam2VideoProcessor.from_pretrained(args.edgetam_model))
        for p in tqdm(paths, desc="re-tracking the pot"):
            out_path = os.path.join(out_dir, os.path.basename(p))
            if not os.path.exists(out_path):
                retrack_pot(p, os.path.join(priv_in, os.path.basename(p)), out_path, edgetam, args, c)
    if todo and not args.states_only:
        import torch
        from types import SimpleNamespace
        from transformers import EdgeTamVideoModel, Sam2VideoProcessor
        dtype = torch.bfloat16
        edgetam = SimpleNamespace(
            device=args.device, dtype=dtype,
            edgetam=EdgeTamVideoModel.from_pretrained(args.edgetam_model, dtype=dtype).to(args.device).eval(),
            edgetam_processor=Sam2VideoProcessor.from_pretrained(args.edgetam_model))
        for p in tqdm(todo, desc="tracking"):
            track(p, os.path.join(out_dir, os.path.basename(p)), edgetam, args, c)

    for p in tqdm(paths, desc="object states"):
        priv_path = os.path.join(out_dir, os.path.basename(p))
        if not os.path.exists(priv_path):
            continue
        rng = np.random.default_rng([args.seed, int(os.path.basename(p)[5:-5])])
        with h5py.File(priv_path, "a") as priv, h5py.File(p, "r") as raw:
            state, visible, synthetic = object_state(priv, raw, c, rng)
            for key in ("object_state", "object_visible"):
                if key in priv:
                    del priv[key]
            priv["object_state"] = state
            priv["object_visible"] = visible
            priv.attrs["labels"] = LABELS
            priv.attrs["synthetic"] = synthetic
        moved = [np.linalg.norm(state[-1, k, :2] - state[np.argmax(visible[:, k]), k, :2]) for k in range(len(LABELS))]
        print(f"{os.path.basename(p)}: seen " + ", ".join(f"{k} {v:.0%}" for k, v in zip(LABELS, visible.mean(0)))
              + "; moved " + ", ".join(f"{k} {m:.2f} m" for k, m in zip(LABELS, moved)))
