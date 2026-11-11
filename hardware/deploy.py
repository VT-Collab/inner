import os
import sys
import time
import queue
import argparse
import threading
import cv2
import numpy as np
from termcolor import colored

from robot import FrankaResearch3
from cameras import CamerasSubscriber
from perception import Segmenter, CameraTracker, load_calibration, mask_centroid_3d, transform_points
from policy import Policy
from objects import mask_points, points_yaw, mean_yaw, unwrap_to
import json
from kinematics import fk, ik, arm_state, joint_target, so3_exp, so3_log, R_REF, tcp_yaw, wrap90
from config import n_joints, object_dim
from viewer import Viewer, LOG, draw_masks, model_info
from collections import deque

KEYS = "g go, s/space stop, 1/0 success/failure, r reset (next trial), h home, p print state, q quit"


def say(msg, color="blue"):
    print(colored("[Deploy] ", color) + msg, flush=True)
    LOG.append(time.strftime("%H:%M:%S ") + msg)


class Mirrored:
    """camera subscriber whose frames are flipped left-right"""

    def __init__(self, sub):
        self.sub = sub

    def get_last_frames(self):
        frames = self.sub.get_last_frames()
        return None if frames is None else {k: np.ascontiguousarray(v[:, ::-1]) for k, v in frames.items()}

    def close_subscriber(self):
        self.sub.close_subscriber()


class ObjectTracker:
    """live object state (x, y, z, yaw) in the robot base frame, computed in its own thread"""

    def __init__(self, args, labels):
        self.labels = labels
        self.cameras = args.cameras
        self.calib = load_calibration(args.calibration, self.cameras)
        self.vis_threshold = args.vis_threshold
        self.load_models(args)
        topics = [f"{cam}_img_{kind}" for cam in self.cameras for kind in ("rgb", "depth")]
        self.sub = CamerasSubscriber(topics=topics, server_addr=args.camera_addr, port=args.camera_port)
        self.sub.start_thread()
        if getattr(args, "mirror_live", False):
            self.sub = Mirrored(self.sub)

        self.lock = threading.Lock()
        self.clear()
        self.reset_requested = threading.Event()
        self.done = False
        self.update_times = deque(maxlen=30)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def load_models(self, args):
        self.segmenter = Segmenter(args.sam3_model, args.edgetam_model, args.device)
        self.trackers = {cam: CameraTracker(self.segmenter, self.labels, args.det_threshold) for cam in self.cameras}

    def clear(self):
        with self.lock:
            self.pos = np.full((len(self.labels), 3), np.nan, dtype=np.float32)
            self.yaw = np.full(len(self.labels), np.nan, dtype=np.float32)
            self.masks = None

    def reset(self, timeout=10.0, idle=None):
        """start a new episode: forget the objects and detect them again"""
        self.reset_requested.set()
        start = time.time()
        while self.reset_requested.is_set() and time.time() - start < timeout:
            if idle is not None:
                idle()
            time.sleep(0.01)

    def restart_tracking(self):
        for tracker in self.trackers.values():
            tracker.reset()

    def _run(self):
        while not self.done:
            if self.reset_requested.is_set():
                self.restart_tracking()
                self.clear()
                self.reset_requested.clear()
            frames = self.sub.get_last_frames()
            if frames is None:
                time.sleep(0.01)
                continue
            self.update(frames)
            if self.masks is not None:
                self.update_times.append(time.time())

    def update(self, frames):
        pos, yaw = [], []
        for cam in self.cameras:
            c = self.calib[cam]
            depth = frames[f"{cam}_img_depth"]
            masks, scores = self.trackers[cam].step(frames[f"{cam}_img_rgb"])
            masks[scores < self.vis_threshold] = False
            if cam == self.cameras[0]:
                self.masks = masks
            pos_cam = np.full((len(self.labels), 3), np.nan, dtype=np.float32)
            yaw_cam = np.full(len(self.labels), np.nan)
            for k in np.flatnonzero(masks.any(axis=(1, 2))):
                pos_cam[k] = mask_centroid_3d(masks[k], depth, c["K"], c["depth_scale"])
                points = mask_points(masks[k], depth, c["K"], c["depth_scale"], c["T_base_cam"])
                if points is not None:
                    yaw_cam[k] = points_yaw(points)
            pos.append(transform_points(c["T_base_cam"], pos_cam))
            yaw.append(yaw_cam)
        pos, yaw = np.stack(pos), mean_yaw(np.stack(yaw))
        seen = ~np.isnan(pos).any(-1)
        with self.lock:
            for k in np.flatnonzero(seen.any(0)):
                self.pos[k] = np.nanmean(pos[:, k], axis=0)
            for k in np.flatnonzero(~np.isnan(yaw)):
                self.yaw[k] = yaw[k] if np.isnan(self.yaw[k]) else unwrap_to(self.yaw[k], yaw[k])

    def all_seen(self):
        with self.lock:
            return not (np.isnan(self.pos).any() or np.isnan(self.yaw).any())

    def missing(self):
        with self.lock:
            return [self.labels[k] for k in np.flatnonzero(np.isnan(self.pos).any(-1) | np.isnan(self.yaw))]

    def state(self):
        with self.lock:
            return np.concatenate([self.pos, self.yaw[:, None]], axis=-1).reshape(-1)

    def close(self):
        self.done = True
        self.sub.close_subscriber()


class PushTracker(ObjectTracker):
    """push-task perception on the live orbbec"""

    def __init__(self, args, labels):
        import yaml
        from track_pot import LABELS
        assert set(labels) <= set(LABELS), f"push perception tracks {LABELS}, the policy has {list(labels)}"
        args.cameras = ["orbbec"]
        calib = yaml.safe_load(open(args.calibration))["orbbec"]
        correction = calib.get("xy_correction")
        self.draw = calib.get("draw")
        self.correction = None if correction is None else np.array(correction, dtype=np.float32)
        self.max_step = args.max_step
        self.spill_guard = not getattr(args, "no_spill_guard", False)
        super().__init__(args, labels)

    def load_models(self, args):
        import torch
        from types import SimpleNamespace
        from transformers import EdgeTamVideoModel, Sam2VideoProcessor
        dtype = torch.bfloat16
        self.edgetam = SimpleNamespace(
            device=args.device, dtype=dtype,
            edgetam=EdgeTamVideoModel.from_pretrained(args.edgetam_model, dtype=dtype).to(args.device).eval(),
            edgetam_processor=Sam2VideoProcessor.from_pretrained(args.edgetam_model))
        self.restart_tracking()

    def restart_tracking(self):
        self.tracker, self.first_area, self.last, self.guard = None, None, None, None

    def start(self, rgb, depth, c):
        from track_pot import find_pot, find_fruit, box_tracker, SpillGuard
        boxes = find_fruit(rgb, depth, c)
        boxes["pot"] = find_pot(rgb, depth, c)
        if any(boxes[k] is None for k in self.labels):
            return False
        # a mask prompt for one object of a multi-object session fails in transformers
        groups = [[k for k in self.labels if k == "pot"], [k for k in self.labels if k != "pot"]]
        self.tracker = [(box_tracker(self.edgetam, g, boxes), [self.labels.index(k) for k in g]) for g in groups if g]
        if self.spill_guard and "pot" in self.labels:
            self.guard = SpillGuard(0)
        print(colored("[PushTracker] ", "green") + "Found " + ", ".join(self.labels))
        return True

    def update(self, frames):
        c = self.calib["orbbec"]
        rgb, depth = frames["orbbec_img_rgb"], frames["orbbec_img_depth"]
        if self.tracker is None and not self.start(rgb, depth, c):
            return
        masks = np.zeros((len(self.labels),) + rgb.shape[:2], dtype=bool)
        scores = np.zeros(len(self.labels), dtype=np.float32)
        for tracker, idx in self.tracker:
            masks[idx], scores[idx] = tracker.step(rgb)
        masks[scores < self.vis_threshold] = False
        if self.guard is not None:
            pot_tracker, idx = self.tracker[0]
            self.guard(pot_tracker, masks[idx[0]], depth, c)
        self.masks = masks
        area = masks.sum(axis=(1, 2))
        if self.first_area is None:
            self.first_area = area.astype(np.float32)
            self.last = [None] * len(self.labels)
        for k in range(len(self.labels)):
            if self.labels[k] == "pot":
                ok = 0 < area[k] <= 2 * self.first_area[k]
            else:
                ok = 0.5 * self.first_area[k] < area[k] < 1.5 * self.first_area[k]
            points = mask_points(masks[k], depth, c["K"], c["depth_scale"], c["T_base_cam"]) if ok else None
            if points is None:
                continue
            if self.correction is not None:
                points[:, :2] = points[:, :2] @ self.correction[:, :2].T + self.correction[:, 2]
            if np.median(points[:, 2]) >= 0.08:
                continue
            if self.labels[k] == "pot" and np.mean(points[:, 2] >= 0.08) > 0.05:
                continue
            p = np.median(points, axis=0)
            if self.labels[k] != "pot" and self.last[k] is not None and np.linalg.norm(p[:2] - self.last[k][:2]) > self.max_step:
                continue
            self.last[k] = p
            yaw = points_yaw(points)
            with self.lock:
                self.pos[k] = p
                self.yaw[k] = yaw if np.isnan(self.yaw[k]) else unwrap_to(self.yaw[k], yaw)


class StackTracker(PushTracker):
    """stack-task perception on the live orbbec"""

    def __init__(self, args, labels):
        from track_stack import LABELS, load_camera
        assert set(labels) <= set(LABELS), f"stack perception tracks {LABELS}, the policy has {list(labels)}"
        args.cameras = ["orbbec"]
        _, self.correction, self.draw = load_camera(args.calibration)
        self.q, self.gripper, self.cube_size = None, 1.0, args.cube_size
        self.static = getattr(args, "static_cubes", 0)
        ObjectTracker.__init__(self, args, labels)

    def set_robot(self, q, gripper):
        with self.lock:
            self.q, self.gripper = np.array(q, dtype=np.float64), float(gripper)

    def restart_tracking(self):
        self.tracker, self.guards, self.est = None, {}, None

    def start(self, rgb, depth, c):
        from track_stack import find_objects, cube_prompts, sessions, CubeGuard, StackState, CUBES
        boxes = find_objects(rgb, depth, c)
        if any(boxes[k] is None for k in self.labels):
            return False
        boxes, points = cube_prompts(rgb, depth, c, boxes, self.cube_size)
        self.tracker = sessions(self.edgetam, self.labels, boxes, points)
        self.guards = {idx[0]: CubeGuard() for _, idx in self.tracker if self.labels[idx[0]] in CUBES}
        self.est = StackState(self.labels, c, self.correction, self.draw, self.cube_size, shape=rgb.shape[:2],
                              static=self.static)
        print(colored("[StackTracker] ", "green") + "Found " + ", ".join(self.labels))
        return True

    def update(self, frames):
        c = self.calib["orbbec"]
        rgb, depth = frames["orbbec_img_rgb"], frames["orbbec_img_depth"]
        if self.tracker is None and not self.start(rgb, depth, c):
            return
        with self.lock:
            q, gripper = self.q, self.gripper
        masks = np.zeros((len(self.labels),) + rgb.shape[:2], dtype=bool)
        scores = np.zeros(len(self.labels), dtype=np.float32)
        for tracker, idx in self.tracker:
            masks[idx], scores[idx] = tracker.step(rgb)
        masks[scores < self.vis_threshold] = False
        robot = self.est.robot_mask(q, rgb.shape[:2])
        for tracker, idx in self.tracker:
            if idx[0] in self.guards:
                self.guards[idx[0]](tracker, masks[idx[0]], robot)
        self.masks = masks
        state, seen = self.est(masks, depth, q, gripper)
        with self.lock:
            for k in np.flatnonzero(seen):
                self.pos[k], self.yaw[k] = state[k, :3], state[k, 3]


class CloseTracker(StackTracker):
    """close-task perception: SAM 3 for the handle and k-cup, fruit by color"""

    def __init__(self, args, labels):
        from track_close import LABELS
        from track_stack import load_camera
        assert set(labels) <= set(LABELS), f"close perception tracks {LABELS}, the policy has {list(labels)}"
        args.cameras = ["orbbec"]
        _, self.correction, self.draw = load_camera(args.calibration)
        self.q, self.gripper = None, 1.0
        self.static = getattr(args, "static_cubes", 0) or 15
        self.live = [] if getattr(args, "close_static", False) else None
        self.pending = False
        ObjectTracker.__init__(self, args, labels)

    def retrack(self):
        """detect the fruit again mid-trial"""
        from track_close import FRUIT
        if any(k in FRUIT for k in self.labels):
            self.pending = True

    def restart_tracking(self):
        super().restart_tracking()
        self.pending = False

    def all_seen(self):
        return not self.pending and super().all_seen()

    def missing(self):
        return ["fruit (re-detecting)"] if self.pending else super().missing()

    def _retrack(self, rgb, depth, c):
        from track_close import first_prompts, close_sessions, FRUIT
        fruit = [k for k in self.labels if k in FRUIT]
        prompts = first_prompts(self.edgetam, rgb, depth, c, fruit)
        if any(prompts[k] is None for k in fruit):
            return
        idx = [self.labels.index(k) for k in fruit]
        (tracker, sub), = close_sessions(self.edgetam, fruit, prompts)
        self.tracker = [(t, i) for t, i in self.tracker if not set(i) & set(idx)] + [(tracker, [idx[j] for j in sub])]
        for k in idx:
            self.est.fixed.pop(k, None)
            self.est.acc[k], self.est.recent[k] = [], []
        with self.lock:
            self.pos[idx], self.yaw[idx] = np.nan, np.nan
        self.pending = False
        print(colored("[CloseTracker] ", "green") + "Re-detected " + ", ".join(fruit))

    def load_models(self, args):
        self.edgetam = Segmenter(args.sam3_model, args.edgetam_model, args.device)
        self.restart_tracking()

    def start(self, rgb, depth, c):
        from track_close import first_prompts, close_sessions, CloseState
        prompts = first_prompts(self.edgetam, rgb, depth, c, self.labels)
        if any(prompts[k] is None for k in self.labels):
            return False
        self.tracker = close_sessions(self.edgetam, self.labels, prompts)
        kw = {} if self.live is None else {"live": self.live}
        self.est = CloseState(self.labels, c, self.correction, self.draw, shape=rgb.shape[:2], static=self.static, **kw)
        print(colored("[CloseTracker] ", "green") + "Found " + ", ".join(self.labels))
        return True

    def update(self, frames):
        c = self.calib["orbbec"]
        rgb, depth = frames["orbbec_img_rgb"], frames["orbbec_img_depth"]
        if self.tracker is None and not self.start(rgb, depth, c):
            return
        if self.pending:
            self._retrack(rgb, depth, c)
        with self.lock:
            q = self.q
        masks = np.zeros((len(self.labels),) + rgb.shape[:2], dtype=bool)
        scores = np.zeros(len(self.labels), dtype=np.float32)
        for tracker, idx in self.tracker:
            masks[idx], scores[idx] = tracker.step(rgb)
        masks[scores < self.vis_threshold] = False
        self.masks = masks
        state, fixed = self.est(masks, depth, q)
        with self.lock:
            for k in np.flatnonzero(fixed):
                self.pos[k], self.yaw[k] = state[k, :3], state[k, 3]


class Recorder:
    """writes one mp4 per trial with the tracked masks"""

    def __init__(self, objects, out_dir, fps):
        self.objects, self.out_dir, self.period = objects, out_dir, 1.0 / fps
        self.cam = objects.cameras[0]
        self.writer, self.path, self.recording, self.text = None, None, False, ""
        self.lock = threading.Lock()
        os.makedirs(out_dir, exist_ok=True)
        threading.Thread(target=self._run, daemon=True).start()

    def start(self, trial):
        with self.lock:
            if not self.recording:
                self.path, self.recording = os.path.join(self.out_dir, f"trial_{trial:03d}.mp4"), True

    def stop(self, label=None):
        """finish the video; label True/False adds _success/_failure to the file name"""
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                if label is not None:
                    path = self.path.replace(".mp4", "_success.mp4" if label else "_failure.mp4")
                    os.replace(self.path, path)
                    self.path = path
                say(f"Video: {self.path}")
            path = self.path if self.writer is not None else None
            self.writer, self.recording = None, False
            return path

    def _run(self):
        while True:
            start = time.time()
            with self.lock:
                frames = self.objects.sub.get_last_frames() if self.recording else None
                if frames is not None:
                    frame = draw_masks(frames[f"{self.cam}_img_rgb"][::2, ::2].copy(), self.objects.masks)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, self.text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    if self.writer is None:
                        h, w = frame.shape[:2]
                        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / self.period, (w, h))
                    self.writer.write(frame)
            time.sleep(max(0.0, self.period - (time.time() - start)))


class Keys:
    """single key presses from the terminal"""

    def __init__(self):
        self.q = queue.Queue()
        self.fd, self.old = None, None
        if sys.stdin.isatty():
            import tty
            import termios
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                return
            self.q.put(ch.lower())

    def get(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        if self.fd is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def limit(dq, max_dq):
    """scale dq so no joint moves more than max_dq"""
    return dq * min(1.0, max_dq / max(np.abs(dq).max(), 1e-12))


class MeanPolicy:
    """mean of the actions of several models trained on the same dataset"""
    def __init__(self, paths, device, n_sample_steps):
        self.policies = [Policy(p, device, n_sample_steps) for p in paths.split(",")]
        self.stats, self.kind = self.policies[0].stats, self.policies[0].kind
        for p in self.policies[1:]:
            assert all(np.array_equal(p.stats[k], self.stats[k]) for k in ("state_min", "state_max", "action_min", "action_max")), \
                "--model: the models were trained on different datasets"

    def __call__(self, state):
        return np.mean([p(state) for p in self.policies], axis=0)


def policy_objects(stats, flat):
    drop = set(stats.get("no_yaw", []))
    if not drop:
        return flat
    s = np.asarray(flat).reshape(-1, object_dim)
    return np.concatenate([row[:3] if label in drop else row for label, row in zip(stats["labels"], s)])


def policy_state(stats, q, gripper, flat):
    """policy state: arm, gripper, objects, and with stats["relative"] those objects' xyz minus the tcp"""
    assert stats.get("obs_window", 1) == 1 and stats.get("chunk", 1) == 1, "deploy runs single-step policies (obs_window = chunk = 1)"
    if stats.get("yaw_relative"):
        # wrapped to +-45 deg
        flat = np.array(flat, dtype=np.float64).reshape(-1, object_dim)
        for k in stats["yaw_relative"]:
            i = list(stats["labels"]).index(k)
            flat[i, 3] = wrap90(flat[i, 3] - tcp_yaw(fk(q)[:3, :3]))
        flat = flat.reshape(-1)
    parts = [arm_state(q, stats.get("state_space", "joint")), [gripper], policy_objects(stats, flat)]
    if stats.get("relative"):
        xyz = np.asarray(flat).reshape(-1, object_dim)[:, :3]
        tcp = fk(q)[:3, 3]
        parts += [xyz[list(stats["labels"]).index(k)] - tcp for k in stats["relative"]]
    return np.concatenate(parts)


def z_floor(args, stats):
    """tcp z floor (m) from --z_floor and --table_z, the higher wins; None if off"""
    floors = []
    if getattr(args, "table_z", None) is not None:
        floors.append(args.table_z + args.z_buffer)
    z = getattr(args, "z_floor", None)
    if z == "demos":
        if stats.get("state_space", "joint") == "cartesian":
            floors.append(float(stats["state_min"][2]))
        else:
            import glob
            import h5py
            paths = glob.glob(os.path.join(args.demo_dir, "demo_*.hdf5"))
            assert paths, f"no demos in {args.demo_dir} for --z_floor demos"
            low = []
            for path in paths:
                with h5py.File(path, "r") as f:
                    low.append(fk(f["obs/joint_state"][:, :n_joints])[:, 2, 3].min())
            floors.append(float(min(low)))
    elif z is not None:
        floors.append(float(z))
    return max(floors) if floors else None


def deploy(robot, objects, policy, keys, args, recorder=None, viewer=None):
    """run rollouts with soft restarts; 1/0 labels a trial and the success rate is printed at the end"""
    stats = policy.stats
    state_space, action_space = stats.get("state_space", "joint"), stats.get("action_space", "joint")
    home = list(args.home if args.home is not None else stats["home"])
    period = 1.0 / stats["control_hz"]
    # _abs policies predict the arm target this many steps ahead
    target_steps = stats.get("target_steps", 1)
    floor = z_floor(args, stats)
    if floor is not None:
        say(f"z floor: tcp z >= {floor:.3f} m", "yellow")
    if viewer is not None:
        viewer.update(z_floor=floor)

    def apply_floor(q, dq):
        """raise a command that would go below the floor to the floor height, keeping xy and rotation"""
        T = fk(q + dq)
        if T[2, 3] >= floor:
            return dq, False
        T[2, 3] = floor
        dq = limit(ik(q + dq, T) - q, args.max_dq)
        if fk(q + dq)[2, 3] < min(fk(q)[2, 3], floor) - 0.002:
            dq = np.zeros_like(dq)
        return dq, True

    def go_home():
        if viewer is not None:
            viewer.update(state="homing")
        if args.dry_run:
            say(f"Dry run: not going home {np.round(home, 3)}", "yellow")
            return
        say("Going home")
        robot.go2position(home)
        robot.send2gripper("o")

    def print_step(step, q, action, dq_raw, dq, floored):
        """print the policy action, the joint change it asks for, and what is sent"""
        tcp = fk(q)[:3, 3]
        if action_space == "cartesian":
            a = f"dx {np.round(100 * action[:3], 2)} cm, rot {np.degrees(np.linalg.norm(action[3:6])):.2f} deg"
        elif action_space == "cartesian_abs":
            a = f"target {np.round(action[:3], 3)} m ({target_steps} steps ahead)"
        elif action_space == "joint_abs":
            a = f"target q {np.round(action[:-1], 3)} ({target_steps} steps ahead)"
        else:
            a = f"dq {np.round(action[:-1], 3)}"
        sent = f"{min(np.abs(dq).max() / period, robot.max_speed):.2f} rad/s" if robot.mode == "velocity" else f"{np.abs(dq).max():.3f} rad"
        say(f"step {step}: tcp {np.round(tcp, 3)}, action {a}, gripper {action[-1]:+.1f} | joint change max "
            f"{np.abs(dq_raw).max():.3f} rad{' (scaled down)' if np.abs(dq_raw).max() > args.max_dq else ''}"
            f", tcp change {np.round(100 * (fk(q + dq)[:3, 3] - tcp), 2)} cm, sent max {sent}"
            + (" (z floor)" if floored else "") + (" (dry run: not sent)" if args.dry_run else ""))

    def report(step_times):
        if step_times:
            late = np.mean(np.array(step_times) > 1.1 * period)
            say(f"{len(step_times)} steps at {1 / np.median(step_times):.1f} Hz (policy trained at {1 / period:g} Hz), "
                f"{100 * late:.0f}% of steps late", "green")

    results = {}

    def summary():
        if results:
            ok = [results[t][0] for t in sorted(results)]
            say(f"Success {sum(ok)}/{len(ok)} ({100 * np.mean(ok):.0f}%): "
                + " ".join(f"{t}:{'1' if s else '0'}" for t, (s, _) in sorted(results.items())), "green")

    floor_steps, videos, retracks = {}, {}, {}
    if args.action_noise:
        assert action_space in ("cartesian", "cartesian_abs"), "--action_noise needs a cartesian action space"
    noise_rng, noise = np.random.default_rng(args.noise_seed), np.zeros(3)
    obs_noise = getattr(args, "obs_noise", 0.0)
    obs_rng = np.random.default_rng(None if args.noise_seed is None else args.noise_seed + 1)
    results_path = None
    if args.results_dir and not args.dry_run:
        info = model_info(args.model, policy)
        name = f"{info['set'] or 'model'}_s{info['seed']}_{info['method'].lower().replace(' ', '_')}_{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(args.results_dir, exist_ok=True)
        results_path = os.path.join(args.results_dir, name + ".json")
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        settings = {k: getattr(args, k, None) for k in ("max_dq", "max_speed", "max_steps", "max_trials", "n_sample_steps",
                                                         "z_floor", "table_z", "z_buffer", "perception", "mirror_live", "no_spill_guard",
                                                         "video_dir", "home", "action_noise", "noise_hold", "noise_seed",
                                                         "retrack", "obs_noise")}
        settings["floor"] = floor

    def save_results():
        if results_path is None:
            return
        trials = [{"trial": t, "success": r[0], "steps": r[1], "floor_steps": floor_steps.get(t, 0),
                   "retracks": retracks.get(t, 0), "video": videos.get(t)} for t, r in sorted(results.items())]
        n_ok = sum(t["success"] for t in trials)
        out = {**info, "started": started, "controller": getattr(robot, "mode", None), "settings": settings,
               "n": len(trials), "successes": n_ok, "success_rate": n_ok / len(trials) if trials else None, "trials": trials}
        with open(results_path + ".tmp", "w") as f:
            json.dump(out, f, indent=1)
        os.replace(results_path + ".tmp", results_path)

    logs, log_dir = {}, getattr(args, "video_dir", None) if not args.dry_run else None
    last_switch, asks = {}, {}

    def save_log(ep):
        rows = logs.get(ep)
        if not rows or not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"trial_{ep:03d}_log.npz")
        np.savez_compressed(path, labels=np.array(stats["labels"]), model=getattr(args, "model", ""), success=results.get(ep, (None,))[0],
                            **{k: np.array([r[k] for r in rows]) for k in rows[0]})
        say(f"Log: {path}")

    if results_path is not None:
        say(f"Results: {results_path}")
    go_home()
    hold, q, gripper, running, waiting, resuming = None, None, 1.0, False, True, False
    episode, step, step_times = 1, 0, []
    wait_start = time.time()
    say(f"Keys: {KEYS}")
    say(f"Trial {episode}: waiting for objects")
    last_time = time.time()
    try:
        while True:
            q = robot.readState()["q"][:n_joints]
            if hasattr(objects, "set_robot"):
                objects.set_robot(q, gripper)
            key = keys.get()
            if key == "q":
                break
            if key in ("s", " ", "1", "0") and running:
                running, hold = False, q
                report(step_times)
                if key in ("s", " "):
                    say(f"Stopped at step {step}. g resumes, 1/0 labels the trial, r resets, h homes", "yellow")
            if key in ("1", "0"):
                if step == 0:
                    say("Nothing to label yet", "yellow")
                else:
                    results[episode] = (key == "1", step)
                    save_results()
                    save_log(episode)
                    say(f"Trial {episode}: {'success' if key == '1' else 'failure'} ({step} steps). r for the next trial", "green")
                    if args.max_trials and len(results) >= args.max_trials:
                        say(f"{len(results)} trials done", "green")
                        break
            elif key in ("s", " ") and resuming:
                resuming = False
                say("Resume cancelled. g re-detects again", "yellow")
            elif key == "g" and not running and args.retrack and step and hasattr(objects, "retrack"):
                objects.retrack()
                retracks[episode] = retracks.get(episode, 0) + 1
                resuming = True
                say("Re-detecting the fruit, resumes once they are fixed (keep hands out of view)", "yellow")
            elif key == "g" and not running:
                if not objects.all_seen():
                    say(f"Not seen yet: {objects.missing()}", "yellow")
                else:
                    running, step_times = True, []
                    if recorder is not None:
                        recorder.start(episode)
                    say(f"Trial {episode}: {'resuming' if step else 'running'}")
            elif key in ("h", "r") and not running:
                if key == "r":
                    if step and episode not in results:
                        say(f"Trial {episode} has no label, not counted", "yellow")
                    if recorder is not None:
                        videos[episode] = recorder.stop(results[episode][0] if episode in results else None)
                        save_results()
                    save_log(episode)
                go_home()
                hold, gripper, resuming = None, 1.0, False
                if key == "r":
                    objects.reset(idle=lambda: robot.stop(robot.readState()["q"][:n_joints]))
                    episode, step, waiting, wait_start = episode + 1, 0, True, time.time()
                    say(f"Trial {episode}: waiting for objects")
                else:
                    say("Home. g resumes from here, r starts the next trial")
                last_time = time.time()
                continue
            elif key == "p":
                say(f"q {np.round(q, 3)}, tcp {np.round(fk(q)[:3, 3], 3)}, gripper {gripper:+.0f}")
                for label, s in zip(stats["labels"], objects.state().reshape(-1, object_dim)):
                    say(f"  {label}: xyz {np.round(s[:3], 3)}, yaw {s[3]:.2f}")

            if resuming and objects.all_seen():
                resuming, running, step_times = False, True, []
                if recorder is not None:
                    recorder.start(episode)
                say(f"Trial {episode}: fruit fixed, resuming")
            if waiting and objects.all_seen():
                waiting = False
                say("Objects found. g starts")
            elif waiting and time.time() - wait_start > args.wait_objects:
                say(f"Still not seen: {objects.missing()} (fix the scene and press r, or q)", "yellow")
                wait_start = time.time()

            if recorder is not None:
                recorder.text = f"trial {episode}  step {step}  " + (
                    ("success" if results[episode][0] else "failure") if episode in results else "running" if running else "stopped")
            if viewer is not None:
                viewer.update(trial=episode, step=step, max_steps=args.max_steps, max_trials=args.max_trials, q=q.tolist(),
                              results=[[t, bool(r[0])] for t, r in sorted(results.items())],
                              control_hz=1 / np.median(step_times[-20:]) if step_times else None,
                              state=("success" if results[episode][0] else "failure") if episode in results
                              else "running" if running else "waiting for objects" if waiting
                              else "re-detecting fruit" if resuming
                              else "ready" if step == 0 else "stopped")
            if running:
                obs = objects.state()
                if obs_noise:
                    obs = np.array(obs, dtype=np.float64).reshape(-1, object_dim)
                    obs[:, :3] += obs_rng.normal(0.0, obs_noise, (len(obs), 3))
                    obs = obs.reshape(-1)
                state = policy_state(stats, q, gripper, obs)
                action = policy(state)
                if stats.get("chunk_relative"):
                    action[:3] += fk(q)[:3, 3]
                if stats.get("rot_relative"):
                    action[3:6] = so3_log(so3_exp(action[3:6]) @ fk(q)[:3, :3] @ R_REF.T)
                if getattr(args, "grip_samples", 1) > 1:
                    action[-1] = np.mean([action[-1]] + [policy(state)[-1] for _ in range(args.grip_samples - 1)])
                target = action[:-1]
                if args.action_noise:
                    if step % args.noise_hold == 0:
                        noise = noise_rng.normal(0.0, args.action_noise, 3)
                    target = target.copy()
                    target[:3] += noise
                dq_raw = (joint_target(q, target, action_space) - q) / target_steps
                dq = limit(dq_raw, args.max_dq)
                floored = False
                if floor is not None:
                    dq, floored = apply_floor(q, dq)
                    floor_steps[episode] = floor_steps.get(episode, 0) + floored
                if args.dry_run or args.verbose:
                    print_step(step, q, action, dq_raw, dq, floored)
                if args.dry_run:
                    robot.stop(q)
                else:
                    robot.move(q, q + dq, period)
                command = 1.0 if action[-1] >= 0.0 else -1.0
                asks[episode] = asks.get(episode, 0) + 1 if command != gripper else 0
                if asks[episode] < getattr(args, "grip_confirm", 1):
                    command = gripper
                if getattr(args, "grip_gate", None) is not None and np.linalg.norm(action[:3] - fk(q)[:3, 3]) >= args.grip_gate:
                    command = gripper
                if step - last_switch.get(episode, -10 ** 6) < getattr(args, "grip_hold", 0):
                    command = gripper
                if command != gripper:
                    last_switch[episode] = step
                if log_dir:
                    T = fk(q)
                    logs.setdefault(episode, []).append(dict(
                        t=time.time(), step=step, q=np.array(q, dtype=np.float64), tcp=T[:3, 3], R=T[:3, :3].reshape(-1),
                        gripper=gripper, command=command, state=np.asarray(state, dtype=np.float32),
                        action=np.asarray(action, dtype=np.float32), dq=np.asarray(dq, dtype=np.float64), floored=floored,
                        noise=np.asarray(noise if args.action_noise else np.zeros(3), dtype=np.float32),
                        objects=np.asarray(objects.state(), dtype=np.float32)))
                if command != gripper:
                    if not args.dry_run:
                        robot.send2gripper("o" if command > 0 else "c")
                    gripper = command
                step += 1
                if step >= args.max_steps:
                    running, hold = False, q + dq
                    report(step_times)
                    say(f"Trial {episode} reached max_steps ({step}). 1/0 labels it, r resets", "green")
            else:
                if hold is None:
                    hold = q
                robot.stop(hold)

            remaining = period - (time.time() - last_time)
            if remaining > 0:
                time.sleep(remaining)
            if running:
                step_times.append(time.time() - last_time)
            last_time = time.time()
    finally:
        robot.stop(hold if hold is not None else q)
        if running:
            report(step_times)
        if recorder is not None:
            videos[episode] = recorder.stop(results[episode][0] if episode in results else None)
        save_results()
        save_log(episode)
        summary()
        if results_path is not None and results:
            say(f"Results: {results_path}", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. models/diffusion_weights1; a,b,c: the mean of several seeds")
    parser.add_argument("--max_steps", type=int, default=600, help="per trial")
    parser.add_argument("--max_trials", type=int, default=None, help="stop after this many labeled trials")
    parser.add_argument("--video_dir", default=None, help="save an mp4 and a per-step log (npz) per trial here")
    parser.add_argument("--video_fps", type=float, default=15)
    parser.add_argument("--results_dir", default="results", help="a json per run with every trial's label ('' = off)")
    parser.add_argument("--z_floor", default=None, help="demos or a tcp z in m (off by default)")
    parser.add_argument("--demo_dir", default="demos/push_v3", help="for --z_floor demos with a joint-state model")
    parser.add_argument("--table_z", type=float, default=None, help="tcp z (m) with the fingers on the table, read with p")
    parser.add_argument("--z_buffer", type=float, default=0.01)
    # largest joint change per control step (rad)
    parser.add_argument("--max_dq", type=float, default=0.05)
    # cap on any joint velocity sent to a velocity controller (rad/s)
    parser.add_argument("--max_speed", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true", help="read the robot and run the policy, print actions, never move")
    parser.add_argument("--verbose", action="store_true", help="print the action at every step")
    parser.add_argument("--home", type=float, nargs=n_joints, default=None)
    parser.add_argument("--n_sample_steps", type=int, default=10)
    parser.add_argument("--grip_samples", type=int, default=1, help="gripper command from the mean of this many policy samples")
    parser.add_argument("--grip_hold", type=int, default=0, help="steps a new gripper command is kept")
    parser.add_argument("--grip_confirm", type=int, default=1, help="a gripper switch needs this many asks in a row")
    parser.add_argument("--grip_gate", type=float, default=None, help="the gripper switches only with a commanded move under this (m)")
    parser.add_argument("--wait_objects", type=float, default=30.0)
    parser.add_argument("--host", default="172.16.0.3")
    parser.add_argument("--robot_port", type=int, default=8080)
    parser.add_argument("--gripper_port", type=int, default=8081)
    parser.add_argument("--perception", choices=["codata", "push", "stack", "close"], default="codata")
    parser.add_argument("--cameras", nargs="+", default=["left_realsense", "right_realsense"])
    parser.add_argument("--calibration", default=None,
                        help="default: /workspace/src/config/calibration/cameras.yaml, push: calibration/orbbec.yaml, stack: calibration/orbbec_stack.yaml")
    parser.add_argument("--camera_addr", default="localhost")
    parser.add_argument("--camera_port", type=int, default=8082)
    parser.add_argument("--sam3_model", default="facebook/sam3")
    parser.add_argument("--edgetam_model", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--det_threshold", type=float, default=0.5)
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    # push: largest fruit move between frames (m)
    parser.add_argument("--max_step", type=float, default=0.03)
    # live frames have to be mirrored to match the recordings
    parser.add_argument("--mirror_live", action="store_true")
    parser.add_argument("--no_spill_guard", action="store_true")
    # stack: cube edge (m)
    parser.add_argument("--cube_size", type=float, default=0.05)
    parser.add_argument("--static_cubes", type=int, default=0, help="stack: each cube fixed at the median of its first N "
                        "measurements, re-measured when let go or bumped (0: running median)")
    parser.add_argument("--close_static", action="store_true", help="close: the handle fixed too (close_v1 models)")
    parser.add_argument("--retrack", action="store_true", help="close: g after a mid-trial stop re-detects the fruit")
    parser.add_argument("--action_noise", type=float, default=0.0, help="std (m) of noise on the commanded tcp target")
    parser.add_argument("--noise_hold", type=int, default=1, help="steps each noise draw is kept")
    parser.add_argument("--noise_seed", type=int, default=None)
    parser.add_argument("--obs_noise", type=float, default=0.0, help="std (m) of noise on the object positions the policy sees, per step")
    parser.add_argument("--device", default="cuda")
    # live view in the browser, 0 = off
    parser.add_argument("--viewer_port", type=int, default=8088)
    parser.add_argument("--open_browser", action="store_true", help="open the viewer in a browser")
    args = parser.parse_args()
    if args.calibration is None:
        args.calibration = ("calibration/orbbec_stack.yaml" if args.perception in ("stack", "close") and os.path.exists("calibration/orbbec_stack.yaml")
                            else "calibration/orbbec.yaml" if args.perception in ("push", "stack") else "/workspace/src/config/calibration/cameras.yaml")

    policy = (MeanPolicy if "," in args.model else Policy)(args.model, args.device, args.n_sample_steps)
    # first call is slow (cuda warmup); do it before the robot connects
    policy(np.zeros_like(policy.stats["state_min"]))
    say(f"{args.model}: {policy.stats.get('state_space', 'joint')} state, {policy.stats.get('action_space', 'joint')} action")
    objects = {"push": PushTracker, "stack": StackTracker, "close": CloseTracker}.get(args.perception, ObjectTracker)(args, policy.stats["labels"])
    viewer = None
    if args.viewer_port:
        viewer = Viewer(objects, model_info(args.model, policy), args.viewer_port, open_browser=args.open_browser)
        viewer.update(state="connecting to robot", dry_run=args.dry_run, max_trials=args.max_trials, max_steps=args.max_steps)
    robot = FrankaResearch3(host=args.host, robot_port=args.robot_port, gripper_port=args.gripper_port, max_speed=args.max_speed)
    keys = Keys()
    if viewer is not None:
        viewer.update(controller=robot.mode)
        viewer.on_key = keys.q.put
    recorder = Recorder(objects, args.video_dir, args.video_fps) if args.video_dir else None
    try:
        deploy(robot, objects, policy, keys, args, recorder, viewer)
    except KeyboardInterrupt:
        print()
    finally:
        robot.stop()
        say("Stopped", "yellow")
        keys.close()
        objects.close()
