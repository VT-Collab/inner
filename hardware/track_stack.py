# object tracking for the stack demos
# usage: python track_stack.py --demo_dir <dir>/stack/train [--click]   (writes <demo_dir>_privileged; skips tracked demos)
#        --states_only: recompute states from the saved masks
import os
import glob
import json
import argparse
import yaml
import h5py
import numpy as np
import cv2
from tqdm import tqdm
from objects import mask_points, points_yaw, unwrap_to
from track_pot import above_table, largest_blob, find_fruit, box_tracker
from robot_mask import RobotMask
from kinematics import fk

LABELS = ["tan_cube", "black_cube", "apple", "orange", "banana"]
CUBES = ["tan_cube", "black_cube"]
FRUIT = ["apple", "orange", "banana"]
COLORS = {"tan_cube": (255, 153, 0), "black_cube": (0, 200, 255), "apple": (180, 0, 255), "orange": (0, 90, 255),
          "banana": (120, 255, 120)}
# a cube's yaw repeats every 90 deg, a fruit's long axis every 180
PERIOD = {"tan_cube": np.pi / 2, "black_cube": np.pi / 2}


def period(label):
    return PERIOD.get(label, np.pi)


def wrap_to(yaw, p):
    return (yaw + p / 2) % p - p / 2


def best_window(proj, size, step=0.001):
    """fraction of points in the best size-wide window along each row of sorted proj, and that window's center"""
    n = proj.shape[1]
    lo, hi = proj[:, 0:1], proj[:, -1:]
    starts = lo - size + step * np.arange(int(np.ceil((hi - lo).max() / step)) + int(size / step) + 1)
    counts = np.stack([np.searchsorted(p, s + size) - np.searchsorted(p, s) for p, s in zip(proj, starts)])
    best = counts >= 0.99 * counts.max(1, keepdims=True)
    return counts.max(1) / n, (starts * best).sum(1) / best.sum(1) + size / 2


def cube_pose(points, cam_xy, size, n_max=2000, q=98, step_deg=0.25):
    """center xy, top z and yaw in [-pi/4, pi/4) of a cube of edge size, from points on any visible face"""
    top = np.percentile(points[:, 2], 95)
    xy = points[:, :2].astype(np.float64)
    if len(xy) > n_max:
        xy = xy[np.linspace(0, len(xy) - 1, n_max).astype(int)]
    th = np.deg2rad(np.arange(0.0, 90.0, step_deg))
    axes = np.stack([np.cos(th), np.sin(th)], 1)
    pu, pv = axes @ xy.T, np.stack([-axes[:, 1], axes[:, 0]], 1) @ xy.T
    extent = lambda p: np.subtract(*np.percentile(p, [q, 100 - q], 1))
    yaw = th[np.argmin(extent(pu) * extent(pv))]
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-u[1], u[0]])
    upright = np.subtract(*np.percentile(points[:, 2], [95, 5])) > 0.4 * size
    center = []
    for a in (u, v):
        p = np.sort(xy @ a)
        lo, hi = np.percentile(p, [100 - q, q])
        if hi - lo <= 0.6 * size and upright:
            mid = (lo + hi) / 2
            center.append(mid + np.sign(mid - cam_xy @ a) * size / 2)
        else:
            center.append(best_window(p[None], size)[1][0])
    return np.array([*(center[0] * u + center[1] * v), top], dtype=np.float32), wrap_to(yaw, np.pi / 2)


def cube_outline(p, yaw, size, c, x0, y0, shape):
    """image outline mask of a cube (center xy, top z, yaw) in a crop with origin (x0, y0)"""
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    sq = size / 2 * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) @ R.T + p[:2]
    uv = project(np.r_[np.c_[sq, np.full(4, p[2])], np.c_[sq, np.full(4, p[2] - size)]], c) - [x0, y0]
    out = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(np.round(uv * 4).astype(np.int32)), 1, shift=2)
    return out.view(bool)


def cube_fit(mask, occluders, c, size, p, yaw, margin=30):
    """xy and yaw of a cube from its mask outline, refined from (p, yaw) by compass search; height stays p[2]"""
    v, u = np.nonzero(mask)
    if len(u) < 50:
        return p, yaw
    h, w = mask.shape
    x0, y0 = max(u.min() - margin, 0), max(v.min() - margin, 0)
    x1, y1 = min(u.max() + margin + 1, w), min(v.max() + margin + 1, h)
    obs = mask[y0:y1, x0:x1]
    occ = occluders[y0:y1, x0:x1]
    n, blobs = cv2.connectedComponents(obs.astype(np.uint8))
    if n > 2:
        near = cv2.dilate(cube_outline(p, yaw, size, c, x0, y0, obs.shape).astype(np.uint8), np.ones((21, 21), np.uint8))
        keep = np.unique(blobs[(near > 0) & obs])
        if len(keep):
            stray = obs & ~np.isin(blobs, keep)
            obs, occ = obs & ~stray, occ | stray
    free = ~obs & ~occ
    n_obs = obs.sum()

    def cost(x):
        m = cube_outline(np.array([x[0], x[1], p[2]]), x[2], size, c, x0, y0, obs.shape)
        return (m & free).sum() + n_obs - (m & obs).sum()

    x = np.array([p[0], p[1], yaw], dtype=np.float64)
    best = cost(x)
    step = np.array([0.004, 0.004, np.deg2rad(4.0)])
    while step[0] > 0.0004:
        moved = False
        for i in range(3):
            for s in (1, -1):
                y = x.copy()
                y[i] += s * step[i]
                cy = cost(y)
                if cy < best:
                    x, best, moved = y, cy, True
        if not moved:
            step /= 2
    return np.array([x[0], x[1], p[2]], dtype=np.float32), wrap_to(x[2], np.pi / 2)


def inside_cube(points, p, yaw, size, tol=0.01):
    """share of points inside the cube, with tol (m) of slack"""
    d = points[:, :2] - p[:2]
    u = d @ np.array([np.cos(yaw), np.sin(yaw)])
    v = d @ np.array([-np.sin(yaw), np.cos(yaw)])
    z = points[:, 2]
    h = size / 2 + tol
    return np.mean((np.abs(u) < h) & (np.abs(v) < h) & (z > p[2] - size - tol) & (z < p[2] + tol))


def cube_colors(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {"tan_cube": (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] >= 40) & (hsv[..., 1] <= 150)
                        & (hsv[..., 2] > 110),
            "black_cube": hsv[..., 2] < 70}


def cube_prompt(rgb, depth, c, label, box, size):
    """EdgeTAM prompt for a whole cube from its first-frame box: (box, (n, 2) face points or None)"""
    h, w = depth.shape
    x0, y0, x1, y1 = box.astype(int)
    inbox = np.zeros((h, w), bool)
    inbox[y0:y1 + 1, x0:x1 + 1] = True
    raised = above_table(depth, c, 0.015, 0.15) & inbox
    m = cube_colors(rgb)[label] & raised
    points = mask_points(m if m.sum() >= 30 else raised, depth, c["K"], c["depth_scale"], c["T_base_cam"], erode_px=1)
    if points is None:
        return box, None
    T = c["T_base_cam"]
    p, yaw = cube_pose(points, T[:2, 3], size)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    sq = size / 2 * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) @ R.T + p[:2]
    corners = np.r_[np.c_[sq, np.full(4, p[2])], np.c_[sq, np.full(4, p[2] - size)]]
    faces = [p] + [np.array([*(p[:2] + size / 2 * n), p[2] - size / 2]) for n in (R[:, 0], R[:, 1], -R[:, 0], -R[:, 1])]
    normals = [np.array([0, 0, 1.0])] + [np.array([*n, 0]) for n in (R[:, 0], R[:, 1], -R[:, 0], -R[:, 1])]
    faces = [f for f, n in zip(faces, normals) if n @ (T[:3, 3] - f) / np.linalg.norm(T[:3, 3] - f) > 0.3]
    uv = project(np.array(corners), c)
    pad_px = 0.1 * (uv.max(0) - uv.min(0))
    box = np.r_[np.maximum(uv.min(0) - pad_px, 0), np.minimum(uv.max(0) + pad_px, [w - 1, h - 1])].astype(np.float32)
    base, valid = base_points(depth, c)
    keep = []
    for (u, v) in project(np.array(faces), c).round().astype(int):
        if box[0] <= u <= box[2] and box[1] <= v <= box[3] and valid[v, u] and 0.008 < base[v, u, 2] < p[2] + 0.01:
            keep.append([u, v])
    return box, (np.array(keep, dtype=np.float32) if keep else None)


def project(X, c):
    T = c["T_base_cam"]
    cam = (np.atleast_2d(X) - T[:3, 3]) @ T[:3, :3]
    return np.stack([c["K"][0, 0] * cam[:, 0] / cam[:, 2] + c["K"][0, 2], c["K"][1, 1] * cam[:, 1] / cam[:, 2] + c["K"][1, 2]], 1)


def cube_prompts(rgb, depth, c, boxes, size):
    """whole-cube prompts for the cubes in boxes: (boxes, {cube: points or None})"""
    boxes, points = dict(boxes), {}
    for k in CUBES:
        if boxes.get(k) is not None:
            boxes[k], points[k] = cube_prompt(rgb, depth, c, k, boxes[k], size)
    return boxes, points


def pad(blob, w, h, f=0.1):
    x, y, bw, bh, _ = blob
    return np.array([max(x - f * bw, 0), max(y - f * bh, 0), min(x + (1 + f) * bw, w - 1), min(y + (1 + f) * bh, h - 1)],
                    dtype=np.float32)


def base_points(depth, c):
    h, w = depth.shape
    v, u = np.mgrid[:h, :w]
    z = depth.astype(np.float32) * c["depth_scale"]
    cam = np.stack([(u - c["K"][0, 2]) * z / c["K"][0, 0], (v - c["K"][1, 2]) * z / c["K"][1, 1], z], axis=-1)
    return cam @ c["T_base_cam"][:3, :3].T + c["T_base_cam"][:3, 3], z > 0.1


def mat_rect(rgb, depth, c, margin=0.02, min_pixels=20000):
    """mat as (4, 2) base-frame xy corners shrunk by margin (m), or None"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    base, valid = base_points(depth, c)
    flat = valid & (np.abs(base[..., 2]) < 0.01) & (hsv[..., 1] < 50) & (hsv[..., 2] > 90) & (base[..., 0] > 0.25)
    flat = cv2.morphologyEx(flat.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(flat)
    if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < min_pixels:
        return None
    xy = base[labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])][::10, :2].astype(np.float32)
    center, (w, h), deg = cv2.minAreaRect(xy)
    return cv2.boxPoints((center, (max(w - 2 * margin, 0), max(h - 2 * margin, 0)), deg))


def on_mat(depth, c, corners):
    """(H, W) bool mask of pixels whose base-frame xy lies inside the convex corners"""
    base, valid = base_points(depth, c)
    p = base[..., :2]
    side = [np.cross(b - a, p - a) for a, b in zip(corners, np.roll(corners, -1, axis=0))]
    return valid & (np.all([s >= 0 for s in side], 0) | np.all([s <= 0 for s in side], 0))


def mat_pixels(corners, c):
    """mat corners projected into the image, (4, 2) pixels"""
    return project(np.c_[corners, np.zeros(4)], c)


def find_cubes(rgb, depth, c, min_pixels=300, mat=None):
    """{label: xyxy box or None} for the tan and black cube on the first frame"""
    h, w = depth.shape
    table = above_table(depth, c, 0.015, 0.15) & (np.ones((h, w), bool) if mat is None else mat)
    boxes = {}
    for label, m in cube_colors(rgb).items():
        blob = largest_blob(m & table, min_pixels)
        boxes[label] = None if blob is None else pad(blob, w, h)
    return boxes


def find_objects(rgb, depth, c):
    """all first-frame boxes, only on the mat when one is found"""
    corners = mat_rect(rgb, depth, c)
    mat = None if corners is None else on_mat(depth, c, corners)
    boxes = find_cubes(rgb, depth, c, mat=mat)
    exclude = np.zeros(depth.shape, bool) if mat is None else ~mat
    # the fruit search skips the cubes (a tan cube can pass for an orange in shadow)
    for b in boxes.values():
        if b is not None:
            x0, y0, x1, y1 = b.astype(int)
            exclude[y0:y1 + 1, x0:x1 + 1] = True
    boxes.update(find_fruit(rgb, depth, c, exclude=exclude))
    return boxes


def ray_point(mask, c, z):
    """xy where the mask centroid's ray meets the plane z (m), for masks with no depth; (3,) or None"""
    v, u = np.nonzero(mask)
    if len(u) < 30:
        return None
    K, T = c["K"], c["T_base_cam"]
    ray = T[:3, :3] @ np.array([(u.mean() - K[0, 2]) / K[0, 0], (v.mean() - K[1, 2]) / K[1, 1], 1.0])
    if abs(ray[2]) < 1e-6:
        return None
    s = (z - T[2, 3]) / ray[2]
    return (T[:3, 3] + s * ray).astype(np.float32) if s > 0 else None


class CubeGuard:
    """re-prompts a cube's session with the mask minus the robot when its mask grows onto the arm"""

    def __init__(self, on_robot=0.2, grow=1.5, keep=0.3):
        self.on_robot, self.grow, self.keep, self.a0 = on_robot, grow, keep, None

    def __call__(self, tracker, mask, robot):
        if not mask.any():
            return False
        self.a0 = mask.sum() if self.a0 is None else self.a0
        cleaned = mask & ~robot
        if (mask & robot).sum() > self.on_robot * mask.sum() and mask.sum() > self.grow * self.a0 and cleaned.sum() > self.keep * self.a0:
            tracker.reprompt(0, cleaned)
            return True
        return False


def cube_tracker(seg, label, box, points):
    """one cube in its own EdgeTAM session, prompted with a box and optional face points"""
    from perception import CameraTracker

    class CubeTracker(CameraTracker):
        def _start(self, image):
            proc = self.seg.edgetam_processor
            self.session = proc.init_video_session(inference_device=self.seg.device, dtype=self.seg.dtype)
            self.session.video_height, self.session.video_width = image.shape[:2]
            self.obj_ids = [0]
            extra = {} if points is None else {"input_points": [[points.tolist()]], "input_labels": [[[1] * len(points)]]}
            proc.add_inputs_to_inference_session(self.session, frame_idx=self.t, obj_ids=[0], input_boxes=[[box.tolist()]],
                                                 original_size=image.shape[:2], **extra)
            return True

        def reprompt(self, k, mask):
            self.seg.edgetam_processor.add_inputs_to_inference_session(
                self.session, frame_idx=self.t, obj_ids=[k], input_masks=[mask.astype(np.float32)], original_size=mask.shape)

    return CubeTracker(seg, [label])


def sessions(edgetam, labels, boxes, points=None):
    """EdgeTAM sessions as [(tracker, label indices)]: each cube alone, the fruit together"""
    points = points or {}
    out = [(cube_tracker(edgetam, k, boxes[k], points.get(k)), [labels.index(k)]) for k in labels if k in CUBES]
    fruit = [k for k in labels if k not in CUBES]
    return out + ([(box_tracker(edgetam, fruit, boxes), [labels.index(k) for k in fruit])] if fruit else [])


class StackState:
    """per-frame (x, y, z, yaw) of each object from masks, depth, joints and gripper; unseen objects keep their last state"""

    def __init__(self, labels, c, correction=None, draw=None, cube_size=0.05, max_step=0.03, grasp_xy=0.10, grasp_z=0.20,
                 agree=0.03,
                 margin=6, shape=(1080, 1920), center="ray", window=5, table_z=0.0, static=0, move=(0.015, np.deg2rad(10)),
                 move_frames=5):
        # center: ray, median or outline (cube_fit)
        self.labels, self.c, self.correction, self.window, self.table_z = list(labels), c, correction, window, table_z
        self.center, self.fit = center, center == "outline"
        self.robot = RobotMask(draw, c["T_base_cam"], shape=shape, margin=margin) if draw is not None else None
        self.cube_size, self.max_step, self.grasp_xy, self.grasp_z, self.agree = cube_size, max_step, grasp_xy, grasp_z, agree
        # static: fix each cube at the median of its first N measurements, 0 for a running median
        self.static, self.move, self.move_frames = static, move, move_frames
        self.reset()

    def reset(self):
        n = len(self.labels)
        self.state = np.full((n, 4), np.nan, dtype=np.float32)
        self.first_area = np.zeros(n)
        self.held, self.closed = {}, False
        # what each cube rests on: None for the table, else a cube index
        self.support = {}
        self.recent = {k: [] for k in range(len(self.labels))}
        self.fixed, self.acc, self.off = {}, {k: [] for k in range(len(self.labels))}, {k: 0 for k in range(len(self.labels))}

    def robot_mask(self, q, shape):
        if self.robot is None or q is None:
            return np.zeros(shape, bool)
        return self.robot(q)

    def support_height(self, k, mask, depth, occluders, z, c):
        """top of cube k on the table (median height around its mask) or on another cube"""
        m = mask.astype(np.uint8)
        ring = cv2.dilate(m, np.ones((31, 31), np.uint8)).astype(bool) & ~cv2.dilate(m, np.ones((11, 11), np.uint8)).astype(bool)
        table = mask_points(ring & ~occluders, depth, c["K"], c["depth_scale"], c["T_base_cam"], erode_px=0)
        heights = {None: (np.median(table[:, 2]) if table is not None and len(table) >= 50 else 0.0) + self.cube_size}
        for j in self.other_cubes(k):
            heights[j] = self.state[j, 2] + self.cube_size
        if k not in self.support:
            self.support[k] = min(heights, key=lambda j: abs(heights[j] - z))
        return heights.get(self.support[k], heights[None])

    def support_top(self, k, z):
        """top of cube k from what it rests on, without depth"""
        heights = {None: self.table_z + self.cube_size}
        for j in self.other_cubes(k):
            heights[j] = self.state[j, 2] + self.cube_size
        if k not in self.support:
            self.support[k] = min(heights, key=lambda j: abs(heights[j] - z))
        return heights.get(self.support[k], heights[None])

    def settle(self, k, m, T):
        """static mode: cube k's fixed state, None until it has static measurements"""
        if m is not None and k in self.fixed:
            f = self.fixed[k]
            far = np.linalg.norm(m[:2] - f[:2]) > self.move[0] or abs(wrap_to(m[3] - f[3], np.pi / 2)) > self.move[1]
            clear = T is None or np.linalg.norm(f[:2] - T[:2, 3]) > self.grasp_xy
            if far and clear:
                self.off[k] += 1
                self.acc[k].append(m)
                if self.off[k] >= self.move_frames:
                    del self.fixed[k]
                    self.off[k] = 0
            else:
                self.off[k], self.acc[k] = 0, []
        elif m is not None:
            self.acc[k].append(m)
            if len(self.acc[k]) >= self.static:
                r = np.array(self.acc[k])
                yaws = r[0, 3] + wrap_to(r[:, 3] - r[0, 3], np.pi / 2)
                self.fixed[k] = np.array([*np.median(r[:, :3], axis=0), np.median(yaws)], dtype=np.float32)
                self.acc[k], self.off[k] = [], 0
        return self.fixed.get(k)

    def corrected(self, p, yaw):
        p = np.array(p, dtype=np.float32)
        if self.correction is not None:
            A, b = self.correction[:, :2], self.correction[:, 2]
            if not np.isnan(yaw):
                d = A @ np.array([np.cos(yaw), np.sin(yaw)])
                yaw = wrap_to(np.arctan2(d[1], d[0]), np.pi / 2)
            p[:2] = A @ p[:2] + b
        return p, yaw

    def other_cubes(self, k):
        return [j for j, label in enumerate(self.labels)
                if label in CUBES and j != k and j not in self.held and not np.isnan(self.state[j, 2])]

    def measure(self, k, mask, depth, occluders=None, height=None):
        label, c = self.labels[k], self.c
        area = mask.sum()
        if area == 0:
            return None
        if self.first_area[k] == 0:
            self.first_area[k] = area
        a0 = self.first_area[k]
        if label in CUBES:
            if not 0.3 * a0 < area < 2 * a0:
                return None
            # a mostly covered cube has not moved: keep its last state
            if height is None and area < 0.6 * a0 and not np.isnan(self.state[k, 0]):
                return None
        elif not 0.5 * a0 < area < 1.5 * a0:
            return None
        if label in CUBES:
            rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
            h, w = mask.shape
            x0, x1, y0, y1 = max(cols[0] - 40, 0), min(cols[-1] + 41, w), max(rows[0] - 40, 0), min(rows[-1] + 41, h)
            K = c["K"].copy()
            K[0, 2] -= x0
            K[1, 2] -= y0
            c = dict(c, K=K)
            mask, depth = mask[y0:y1, x0:x1], depth[y0:y1, x0:x1]
            occluders = np.zeros_like(mask) if occluders is None else occluders[y0:y1, x0:x1]
        if label in CUBES and self.center == "ray":
            points = mask_points(mask, depth, c["K"], c["depth_scale"], c["T_base_cam"])
            if points is not None and (len(points) < 5 or np.median(points[:, 2]) >= 0.5):
                points = None
            z = np.percentile(points[:, 2], 95) if points is not None else self.table_z + self.cube_size
            if height is None and not np.isnan(self.state[k, 0]):
                m8 = mask.astype(np.uint8)
                edge = cv2.dilate(m8, np.ones((5, 5), np.uint8)).astype(bool) & ~mask
                if (edge & occluders).sum() > 0.1 * edge.sum():
                    return None
            top = height if height is not None else self.support_top(k, z)
            q = ray_point(mask, c, top - self.cube_size / 2)
            if q is None:
                return None
            yaw = np.nan if points is None else cube_pose(points, c["T_base_cam"][:2, 3], self.cube_size)[1]
            p, yaw = self.corrected([q[0], q[1], top], yaw)
            return np.array([*p, self.state[k, 3] if np.isnan(yaw) else yaw], dtype=np.float32)
        points = mask_points(mask, depth, c["K"], c["depth_scale"], c["T_base_cam"])
        if points is None and label in CUBES and k not in self.held:
            # a black top face often has no depth: its centroid ray at the cube's top
            p = ray_point(mask, c, self.cube_size)
            points = None if p is None else p[None]
        if points is None:
            return None
        if label in CUBES:
            if np.median(points[:, 2]) >= 0.5:
                return None
            if len(points) < 5:
                p, yaw = points[0], np.nan
            else:
                p, yaw = cube_pose(points, c["T_base_cam"][:2, 3], self.cube_size, n_max=500 if self.fit else 2000,
                                   step_deg=2.0 if self.fit else 0.25)
                if not self.fit:
                    p = np.median(points, axis=0)
                else:
                    p[2] = height if height is not None else self.support_height(k, mask, depth, occluders, p[2], c)
                    fp, fyaw = cube_fit(mask, occluders, c, self.cube_size, p, yaw)
                    # reject fits that leave most depth points outside the cube
                    if inside_cube(points, fp, fyaw, self.cube_size) >= 0.5:
                        p, yaw = fp, fyaw
            p, yaw = self.corrected(p, yaw)
            return np.array([*p, self.state[k, 3] if np.isnan(yaw) else yaw], dtype=np.float32)
        if self.correction is not None:
            points = points.copy()
            points[:, :2] = points[:, :2] @ self.correction[:, :2].T + self.correction[:, 2]
        p = np.median(points, axis=0)
        if p[2] >= 0.08:
            return None
        last = self.state[k]
        if not np.isnan(last[0]) and np.linalg.norm(p[:2] - last[:2]) > self.max_step:
            return None
        return np.array([*p, points_yaw(points)], dtype=np.float32)

    def __call__(self, masks, depth, q=None, gripper=1.0):
        """returns state (n, 4) and seen (n,) from masks (n, H, W) bool, depth (H, W), q (7,) or None, gripper +-1"""
        robot = self.robot_mask(q, masks.shape[1:])
        seen = np.zeros(len(self.labels), bool)
        T = fk(np.asarray(q, dtype=np.float64)) if q is not None else None
        closed = gripper < 0 and T is not None
        if closed and not self.closed:
            # the cube point is ~12 cm under the tcp and a few cm toward the camera, so a 3D distance misses it
            near = [(np.linalg.norm(self.state[k, :2] - T[:2, 3]), k) for k, l in enumerate(self.labels)
                    if l in CUBES and not np.isnan(self.state[k, 0]) and abs(self.state[k, 2] - T[2, 3]) < self.grasp_z]
            near = [x for x in near if x[0] < self.grasp_xy]
            if near:
                k = min(near)[1]
                R, t = T[:3, :3], T[:3, 3]
                self.held[k] = (R.T @ (self.state[k, :3] - t), self.state[k, 3] - np.arctan2(R[1, 0], R[0, 0]))
        if not closed:
            for k in self.held:
                self.fixed.pop(k, None)
                self.acc[k], self.off[k] = [], 0
                stacks = [(abs(self.state[k, 2] - self.state[j, 2] - self.cube_size), j) for j in self.other_cubes(k)]
                stacks = [x for x in stacks if x[0] < self.cube_size / 2]
                self.support[k] = min(stacks)[1] if stacks else None
            self.held = {}
        self.closed = closed
        preds = {}
        for k, (offset, dyaw) in self.held.items():
            R, t = T[:3, :3], T[:3, 3]
            preds[k] = np.array([*(t + R @ offset), np.arctan2(R[1, 0], R[0, 0]) + dyaw], dtype=np.float32)
        anything = masks.any(0)
        meas = [self.measure(k, masks[k] & ~robot, depth, robot | (anything & ~masks[k]), preds[k][2] if k in preds else None)
                for k in range(len(self.labels))]
        for k, m in enumerate(meas):
            if k in preds:
                m = preds[k]
                self.recent[k] = []
            elif self.static and self.labels[k] in CUBES:
                m = self.settle(k, m, T)
            elif m is not None and self.labels[k] in CUBES and self.window > 1:
                self.recent[k] = (self.recent[k] + [m])[-self.window:]
                r = np.array(self.recent[k])
                yaws = m[3] + wrap_to(r[:, 3] - m[3], np.pi / 2)
                m = np.array([*np.median(r[:, :3], axis=0), np.median(yaws)], dtype=np.float32)
            if m is None:
                continue
            seen[k] = True
            prev = self.state[k, 3]
            yaw = m[3] if np.isnan(prev) else prev + wrap_to(m[3] - prev, period(self.labels[k]))
            self.state[k] = [*m[:3], yaw]
        return self.state.copy(), seen


def load_camera(path):
    from perception import load_calibration
    c = load_calibration(path, ["orbbec"])["orbbec"]
    y = yaml.safe_load(open(path))["orbbec"]
    corr = y.get("xy_correction")
    return c, None if corr is None else np.array(corr, dtype=np.float32), y.get("draw")


def load_edgetam(name, device):
    import torch
    from types import SimpleNamespace
    from transformers import EdgeTamVideoModel, Sam2VideoProcessor
    dtype = torch.bfloat16
    return SimpleNamespace(device=device, dtype=dtype,
                           edgetam=EdgeTamVideoModel.from_pretrained(name, dtype=dtype).to(device).eval(),
                           edgetam_processor=Sam2VideoProcessor.from_pretrained(name))


def click_box(rgb, label):
    """box drawn by hand on the first frame (enter/space confirms, c cancels)"""
    view = cv2.cvtColor(cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2)), cv2.COLOR_RGB2BGR)
    x, y, w, h = cv2.selectROI(f"draw the {label}", view, showCrosshair=False)
    cv2.destroyAllWindows()
    return None if w == 0 or h == 0 else np.array([2 * x, 2 * y, 2 * (x + w), 2 * (y + h)], dtype=np.float32)


def first_boxes(rgb, depth, c, name, saved, click):
    boxes = find_objects(rgb, depth, c)
    for k, b in saved.get(name, {}).items():
        boxes[k] = np.array(b, dtype=np.float32)
    for k in LABELS:
        if boxes[k] is None and click:
            boxes[k] = click_box(rgb, k)
            if boxes[k] is not None:
                saved.setdefault(name, {})[k] = boxes[k].tolist()
    return boxes


def overlay(frame, masks, labels):
    for i, k in enumerate(labels):
        frame[masks[i]] = (0.5 * frame[masks[i]] + 0.5 * np.array(COLORS[k])).astype(np.uint8)
    return frame


def track(raw_path, out_path, edgetam, boxes, args, cam):
    """track one demo; writes masks, states and an overlay video to the privileged demo file"""
    c, corr, draw = cam
    missing = [k for k in LABELS if boxes[k] is None]
    assert not any(k in CUBES for k in missing), f"{raw_path}: no {missing} on the first frame (draw them with --click)"
    labels = [k for k in LABELS if boxes[k] is not None]
    with h5py.File(raw_path, "r") as raw:
        rgb, depth = raw["obs"]["orbbec_img_rgb"], raw["obs"]["orbbec_img_depth"]
        q = raw["obs"]["joint_state"][:, :7]
        gripper = np.asarray(raw["obs"]["gripper_state"]).reshape(-1)
        n, h, w = rgb.shape[:3]
        boxes, points = cube_prompts(rgb[0], depth[0], c, boxes, args.cube_size)
        trackers = sessions(edgetam, labels, boxes, points)
        guards = {idx[0]: CubeGuard() for _, idx in trackers if labels[idx[0]] in CUBES}
        est = StackState(labels, c, corr, draw, args.cube_size, shape=(h, w), static=args.static)
        state = np.full((n, len(LABELS), 4), np.nan, dtype=np.float32)
        visible = np.zeros((n, len(LABELS)), bool)
        held = np.zeros((n, len(LABELS)), bool)
        reprompts = 0
        video = cv2.VideoWriter(out_path.replace(".hdf5", "_orbbec.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                float(raw.attrs.get("fps", 20)), (w // 2, h // 2))
        tmp_path = out_path + ".tmp"
        with h5py.File(tmp_path, "w") as out:
            masks_out = out.create_dataset("orbbec/masks", shape=(n, len(LABELS), h, w), dtype=bool, chunks=(1, 1, h, w),
                                           compression="gzip")
            scores_out = np.zeros((n, len(LABELS)), np.float32)
            for t in tqdm(range(n), desc=os.path.basename(raw_path), leave=False):
                image = rgb[t]
                m = np.zeros((len(labels), h, w), bool)
                s = np.zeros(len(labels), np.float32)
                for tracker, idx in trackers:
                    m[idx], s[idx] = tracker.step(image)
                m[s < args.vis_threshold] = False
                robot = est.robot_mask(q[t], (h, w))
                for (tracker, idx) in trackers:
                    if idx[0] in guards:
                        reprompts += guards[idx[0]](tracker, m[idx[0]], robot)
                st, seen = est(m, depth[t], q[t], gripper[t])
                cols = [LABELS.index(k) for k in labels]
                state[t, cols], visible[t, cols] = st, seen
                for k in est.held:
                    held[t, cols[k]] = True
                full = np.zeros((len(LABELS), h, w), bool)
                full[cols] = m
                masks_out[t] = full
                scores_out[t, cols] = s
                video.write(cv2.cvtColor(cv2.resize(overlay(image.copy(), m, labels), (w // 2, h // 2)), cv2.COLOR_RGB2BGR))
            out.create_dataset("orbbec/scores", data=scores_out)
            out["object_state"], out["object_visible"], out["object_held"] = state, visible, held
            out.attrs["source"] = os.path.realpath(raw_path)
            out.attrs["cameras"] = ["orbbec"]
            out.attrs["tracked"] = labels
            out.attrs["labels"] = LABELS
            out.attrs["synthetic"] = []
            out.attrs["missing"] = missing
            out.attrs["first_boxes"] = np.array([boxes[k] for k in labels])
            out.attrs["reprompts"] = reprompts
        video.release()
    os.replace(tmp_path, out_path)
    return state, visible, held, missing, reprompts


def jitter(state, held):
    """median and 95% frame-to-frame xy step (mm) of each cube while not held"""
    out = []
    for k in range(len(CUBES)):
        free = ~held[1:, k] & ~held[:-1, k]
        step = np.linalg.norm(np.diff(state[:, k, :2], axis=0), axis=1)[free]
        step = step[~np.isnan(step)]
        out.append((np.median(step) * 1000, np.percentile(step, 95) * 1000) if len(step) else (np.nan, np.nan))
    return out


def restate(priv_path, raw_path, args, cam):
    """recompute a tracked demo's states from its saved masks; originals kept as *_prev"""
    c, corr, draw = cam
    with h5py.File(priv_path, "r+") as out, h5py.File(raw_path, "r") as raw:
        labels = [str(k) for k in out.attrs["tracked"]]
        cols = [LABELS.index(k) for k in labels]
        depth = raw["obs"]["orbbec_img_depth"]
        q = raw["obs"]["joint_state"][:, :7]
        gripper = np.asarray(raw["obs"]["gripper_state"]).reshape(-1)
        masks = out["orbbec/masks"]
        n, h, w = len(q), depth.shape[1], depth.shape[2]
        est = StackState(labels, c, corr, draw, args.cube_size, shape=(h, w), static=args.static)
        state = np.full((n, len(LABELS), 4), np.nan, dtype=np.float32)
        visible = np.zeros((n, len(LABELS)), bool)
        held = np.zeros((n, len(LABELS)), bool)
        for t in tqdm(range(n), desc=os.path.basename(priv_path), leave=False):
            st, seen = est(masks[t][cols], depth[t], q[t], gripper[t])
            state[t, cols], visible[t, cols] = st, seen
            for k in est.held:
                held[t, cols[k]] = True
        before = jitter(np.asarray(out["object_state"]), np.asarray(out["object_held"]))
        for key, val in (("object_state", state), ("object_visible", visible), ("object_held", held)):
            if key + "_prev" not in out:
                out.copy(key, key + "_prev")
            del out[key]
            out[key] = val
        out.attrs["cube_size"] = args.cube_size
    return before, jitter(state, held)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", default="/media/benjamin/T7/inner-product/inner_product/stack/train")
    parser.add_argument("--out_dir", default=None, help="default: <demo_dir>_privileged")
    parser.add_argument("--calibration", default="calibration/orbbec_stack.yaml" if os.path.exists("calibration/orbbec_stack.yaml")
                        else "calibration/orbbec.yaml")
    parser.add_argument("--edgetam_model", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--cube_size", type=float, default=0.05, help="cube edge (m)")
    parser.add_argument("--static", type=int, default=0, help="cubes fixed at the median of their first N measurements (deploy --static_cubes)")
    parser.add_argument("--device", default="cuda")
    # draw boxes by hand where the color search finds nothing
    parser.add_argument("--click", action="store_true")
    # only write first-frame box images, to check before tracking
    parser.add_argument("--boxes_only", action="store_true")
    # recompute states of tracked demos from saved masks
    parser.add_argument("--states_only", action="store_true")
    args = parser.parse_args()

    cam = load_camera(args.calibration)
    out_dir = args.out_dir or args.demo_dir.rstrip("/") + "_privileged"
    if args.states_only:
        privs = sorted(glob.glob(os.path.join(out_dir, "demo_*.hdf5")), key=lambda p: int(os.path.basename(p)[5:-5]))
        assert privs, f"no tracked demos in {out_dir}"
        for p in tqdm(privs, desc="states"):
            raw_path = os.path.join(args.demo_dir, os.path.basename(p))
            if not os.path.exists(raw_path):
                raw_path = str(h5py.File(p, "r").attrs["source"])
            before, after = restate(p, raw_path, args, cam)
            print(f"{os.path.basename(p)}: cube step per frame (median / 95%, mm) " + ", ".join(
                f"{k} {b[0]:.1f}/{b[1]:.1f} -> {a[0]:.1f}/{a[1]:.1f}" for k, b, a in zip(CUBES, before, after)))
        raise SystemExit
    os.makedirs(out_dir, exist_ok=True)
    boxes_path = os.path.join(out_dir, "boxes.json")
    saved = json.load(open(boxes_path)) if os.path.exists(boxes_path) else {}
    paths = sorted(glob.glob(os.path.join(args.demo_dir, "demo_*.hdf5")), key=lambda p: int(os.path.basename(p)[5:-5]))
    assert paths, f"no demos in {args.demo_dir}"
    todo = [p for p in paths if args.boxes_only or not os.path.exists(os.path.join(out_dir, os.path.basename(p)))]
    edgetam = None if args.boxes_only or not todo else load_edgetam(args.edgetam_model, args.device)
    for p in tqdm(todo, desc="boxes" if args.boxes_only else "tracking"):
        name = os.path.basename(p)
        with h5py.File(p, "r") as raw:
            rgb0, depth0 = raw["obs"]["orbbec_img_rgb"][0], raw["obs"]["orbbec_img_depth"][0]
        boxes = first_boxes(rgb0, depth0, cam[0], name, saved, args.click)
        json.dump(saved, open(boxes_path, "w"), indent=1)
        if args.boxes_only:
            os.makedirs(os.path.join(out_dir, "first_frames"), exist_ok=True)
            img = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
            corners = mat_rect(rgb0, depth0, cam[0])
            if corners is not None:
                cv2.polylines(img, [mat_pixels(corners, cam[0]).astype(np.int32)], True, (255, 255, 255), 2)
            boxes, points = cube_prompts(rgb0, depth0, cam[0], boxes, args.cube_size)
            for k, pts in points.items():
                for u, v in ([] if pts is None else pts.astype(int)):
                    cv2.circle(img, (int(u), int(v)), 8, COLORS[k][::-1], -1)
            for k, b in boxes.items():
                if b is not None:
                    x0, y0, x1, y1 = b.astype(int)
                    color = COLORS[k][::-1]
                    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
                    cv2.putText(img, k, (x0, max(y0 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imwrite(os.path.join(out_dir, "first_frames", name.replace(".hdf5", ".jpg")), cv2.resize(img, (960, 540)))
            size = "no mat found" if corners is None else "mat %.2f x %.2f m" % tuple(
                np.linalg.norm(corners[1:3] - corners[:2], axis=1))
            print(f"{name}: {size}, " + ", ".join(f"{k} {'ok' if boxes[k] is not None else 'MISSING'}" for k in LABELS))
            continue
        state, visible, held, missing, reprompts = track(p, os.path.join(out_dir, name), edgetam, boxes, args, cam)
        moved = [np.linalg.norm(state[-1, k, :2] - state[np.argmax(visible[:, k]), k, :2]) if visible[:, k].any() else np.nan
                 for k in range(len(LABELS))]
        print(f"{name}: seen " + ", ".join(f"{k} {v:.0%}" for k, v in zip(LABELS, visible.mean(0)))
              + f"; held {', '.join(f'{k} {h:.0%}' for k, h in zip(CUBES, held[:, :2].mean(0)))}"
              + f"; moved " + ", ".join(f"{k} {m:.2f} m" for k, m in zip(LABELS, moved))
              + (f"; not on the table: {missing}" if missing else "") + f"; re-prompts {reprompts}")
