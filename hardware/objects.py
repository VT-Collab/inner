import cv2
import warnings
import numpy as np


def mask_points(mask, depth, K, depth_scale, T_base_cam, erode_px=3, min_pixels=30, depth_range=(0.1, 3.0)):
    """3D points (robot base frame) under a mask, None if too few valid pixels"""
    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float32) * depth_scale
    valid = (z > depth_range[0]) & (z < depth_range[1])
    if valid.sum() < min_pixels:
        return None
    u, v, z = u[valid], v[valid], z[valid]
    points = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return points @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]


def points_yaw(points):
    """yaw of the long axis of the points in the table plane, in [-pi/2, pi/2)"""
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    _, _, Vh = np.linalg.svd(xy, full_matrices=False)
    return wrap(np.arctan2(Vh[0, 1], Vh[0, 0]))


def wrap(yaw):
    return (yaw + np.pi / 2) % np.pi - np.pi / 2


def mean_yaw(yaws):
    """mean axis angle over cameras (axis 0), ignoring NaNs"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c, s = np.nanmean(np.cos(2 * yaws), axis=0), np.nanmean(np.sin(2 * yaws), axis=0)
    return np.arctan2(s, c) / 2


def unwrap_to(prev, yaw):
    """yaw + k * pi closest to prev"""
    return yaw + np.pi * np.round((prev - yaw) / np.pi)
