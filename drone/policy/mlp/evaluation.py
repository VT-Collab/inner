# MLP rollouts on the shared packaged test cases (20 seconds per case).
# experiments/mlp_run.py owns persistence and output naming.
# E4 loads the transformation from the checkpoint.
import csv
import json
import os
import sys

import numpy as np
import torch

from .config import (action_dim, extra_dims, experiments, n_train_demos, default_seed, default_experiment, default_n_demos,
                    ctrl_hz, pyb_hz, test_steps, success_xy, success_z, success_speed, success_tilt_deg,
                    success_stable_steps, hidden_dim, data_key, run_name)
from .models import MLPPolicy, StateScramble
from .data import test_cases as load_packaged_test_cases

MODEL_NAMES = {0: "BC", 1: "InneR", 2: "Frobenius", 3: "nuclear", 4: "InneR-C"}
# CPU threads of this process (the rollouts run on the CPU)
TORCH_THREADS = 2
# success rates are also reported for these shorter time limits (s)
SUCCESS_AT = (12, 15, 20)


# the test cases: the reset observation, the target, and the extra entries of the experiment at each control step
def load_test_cases(experiment):
    return load_packaged_test_cases(experiment)


# the policy network of a training checkpoint (e4: the saved scramble, then the MLP) and its normalization
def load_policy(checkpoint):
    model = MLPPolicy(checkpoint["input_dim"], hidden_dim, action_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint.get("scramble_state_dict") is not None:
        scramble = StateScramble(checkpoint["input_dim"])
        scramble.load_state_dict(checkpoint["scramble_state_dict"])
        model = torch.nn.Sequential(scramble, model)
    stats = {k: v.numpy() for k, v in checkpoint["normalization"].items()}
    return model.eval(), stats


# policy(obs20, target, step) -> motor RPM: the 19D state (the 20D observation without the previous RPM, and the target)
# and the extra entries, normalized with the training statistics; the output is the normalized RPM
def mlp_policy(model, stats, extra):
    mean, std = stats["state_mean"], stats["state_std"]
    a_mean, a_std = stats["action_mean"], stats["action_std"]

    def policy(obs20, target, step):
        state = np.concatenate((np.asarray(obs20, np.float64)[:16], np.asarray(target, np.float64),
                                np.asarray(extra(step), np.float64))).astype(np.float32)
        s = (state - mean) / std
        with torch.inference_mode():
            predicted = model(torch.from_numpy(s.astype(np.float32))[None, :])[0].numpy()
        return predicted * a_std + a_mean
    return policy


def ground_contact(bullet, env):
    return bool(bullet.getContactPoints(bodyA=int(env.DRONE_IDS[0]), bodyB=int(env.PLANE_ID),
                                        physicsClientId=int(env.CLIENT)))


# one flight from a test case; success = ground contact with the XY error, Z error, speed and tilt within the bounds of
# config.py for success_stable_steps consecutive control steps; the goal error is the final 3D distance to the target
def evaluate_case(initial_obs20, target, policy):
    import pybullet as p
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    initial = np.asarray(initial_obs20, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     initial_xyzs=initial[:3].reshape(1, 3).astype(float),
                     initial_rpys=initial[7:10].reshape(1, 3).astype(float),
                     physics=Physics.PYB, pyb_freq=pyb_hz, ctrl_freq=ctrl_hz,
                     gui=False, record=False, obstacles=False, user_debug_gui=False)
    try:
        obs, _ = env.reset()
        obs20 = np.asarray(obs[0], dtype=np.float32)
        stable_steps, steps, max_tilt_deg = 0, 0, 0.0
        first_contact_time = None
        success, termination = False, "timeout"
        for step in range(test_steps):
            raw_rpm = policy(obs20, target, step)
            if not np.isfinite(raw_rpm).all():
                termination = "non_finite_action"
                break
            rpm = np.clip(raw_rpm, 0.0, float(env.MAX_RPM))
            obs, _, terminated, truncated, _ = env.step(rpm.reshape(1, 4))
            obs20 = np.asarray(obs[0], dtype=np.float32)
            steps = step + 1
            if not np.isfinite(obs20).all():
                termination = "non_finite_state"
                break

            pos, rpy, velocity = obs20[:3], obs20[7:10], obs20[10:13]
            xy_error = float(np.linalg.norm(pos[:2] - target[:2]))
            z_error = float(abs(pos[2] - target[2]))
            speed = float(np.linalg.norm(velocity))
            tilt_deg = float(np.rad2deg(np.linalg.norm(rpy[:2])))
            max_tilt_deg = max(max_tilt_deg, tilt_deg)
            contact = ground_contact(p, env)
            if contact and first_contact_time is None:
                first_contact_time = steps / ctrl_hz

            stable = (contact and xy_error <= success_xy and z_error <= success_z
                      and speed <= success_speed and tilt_deg <= success_tilt_deg)
            stable_steps = stable_steps + 1 if stable else 0
            if stable_steps >= success_stable_steps:
                success, termination = True, "stable_landing"
                break
            if terminated or truncated:
                termination = "environment_terminated"
                break

        final_pos = obs20[:3]
        return {"success": bool(success), "termination": termination, "duration_s": steps / ctrl_hz,
                "final_xyz_error_m": float(np.linalg.norm(final_pos - target)),
                "final_xy_error_m": float(np.linalg.norm(final_pos[:2] - target[:2])),
                "final_z_error_m": float(abs(final_pos[2] - target[2])),
                "final_speed_m_s": float(np.linalg.norm(obs20[10:13])),
                "first_contact_time_s": first_contact_time, "max_tilt_deg": max_tilt_deg}
    finally:
        env.close()


def summarize(records):
    success = np.array([r["success"] for r in records])
    duration = np.array([r["duration_s"] for r in records])
    err = np.array([r["final_xyz_error_m"] for r in records])
    xy = np.array([r["final_xy_error_m"] for r in records])
    out = {f"success_at_{T}": float(np.mean(success & (duration <= T + 1e-9))) for T in SUCCESS_AT}
    out.update({"err_mean": float(err.mean()), "err_median": float(np.median(err)),
                "xy_error_mean": float(xy.mean()), "xy_error_median": float(np.median(xy)),
                "z_error_mean": float(np.mean([r["final_z_error_m"] for r in records])),
                "landing_time_median": float(np.median(duration[success])) if success.any() else None,
                "far_miss_rate": float(np.mean(xy > 0.10)),
                "flip_rate": float(np.mean([r["max_tilt_deg"] > 90 for r in records]))})
    return out

