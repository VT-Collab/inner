import cv2
import numpy as np
from kinematics import DH, _link

# robot mask from joint angles; arm as capsules (joint i, joint j, radius in m), hand and fingers as boxes in the tcp frame
LINKS = [(0, 1, 0.09), (2, 3, 0.08), (3, 4, 0.08), (4, 5, 0.07), (5, 7, 0.07), (7, 8, 0.06)]
HAND = ((-0.04, 0.04), (-0.11, 0.11), (-0.1034, -0.045))
FINGERS = [((-0.012, 0.012), (0.025, 0.055), (-0.045, 0.03)), ((-0.012, 0.012), (-0.055, -0.025), (-0.045, 0.03))]


def frames(q):
    """joint frame origins (base to flange) and the tcp frame"""
    T, origins = np.eye(4), [np.zeros(3)]
    for i, (a, d, alpha) in enumerate(DH):
        T = T @ _link(a, d, alpha, np.array(q[i], dtype=np.float64))
        origins.append(T[:3, 3].copy())
    flange = T @ _link(0, 0.107, 0, np.array(0.0))
    origins.append(flange[:3, 3].copy())
    return np.array(origins), flange @ _link(0, 0.1034, 0, np.array(-np.pi / 4))


def _box(T, xs, ys, zs):
    corners = np.array([[x, y, z, 1.0] for x in xs for y in ys for z in zs])
    return (corners @ T.T)[:, :3]


def _capsule(a, b, r, n=24):
    axis = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    u = np.cross(axis, [0, 0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    return np.concatenate([a + ring, b + ring, [a - r * axis, b + r * axis]])


class RobotMask:
    """boolean image mask of the robot for joints q; margin (px) pads for camera model error"""

    def __init__(self, draw, T_base_cam, shape=(1080, 1920), margin=20):
        self.d, self.shape, self.margin = draw, shape, margin
        self.T_cam_base = np.linalg.inv(np.asarray(T_base_cam, dtype=np.float64))

    def project(self, p):
        cam = p @ self.T_cam_base[:3, :3].T + self.T_cam_base[:3, 3]
        z = np.maximum(cam[:, 2], 1e-3)
        u = self.d["fx"] * cam[:, 0] / z + self.d["cx"]
        if self.d.get("mirrored"):
            u = self.d["width"] - 1 - u
        return np.stack([u, self.d["fy"] * cam[:, 1] / z + self.d["cy"]], axis=1), cam[:, 2] > 0.05

    def parts(self, q):
        origins, tcp = frames(q)
        parts = [_capsule(origins[i], origins[j], r) for i, j, r in LINKS]
        parts += [_box(tcp, *HAND)] + [_box(tcp, *f) for f in FINGERS]
        return parts

    def __call__(self, q):
        mask = np.zeros(self.shape, np.uint8)
        for p in self.parts(q):
            uv, front = self.project(p)
            if front.all():
                cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 1)
        if self.margin > 0:
            k = 2 * self.margin + 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return mask.astype(bool)
