import contextlib
import io
import argparse
import numpy as np
import torch
from . import drone_support
from .experiment_utils import episode_clutter_base, is_descent_phase
STATE_DIM = 23
ACTION_DIM = 4
def model_rpm(model, norm, obs20, target, clutter=None):
    state = np.concatenate((obs20, target)).astype(np.float32)
    standardized = (state - norm["state_mean"]) / norm["state_std"]
    if clutter is not None:
        standardized = np.concatenate((standardized, clutter.astype(np.float32)))
    if hasattr(model, "predict_normalized"):
        predicted = model.predict_normalized(standardized)
    else:
        with torch.inference_mode():
            predicted = model(torch.from_numpy(standardized)[None, :])[0].numpy()
    return predicted * norm["action_std"] + norm["action_mean"]


def ground_contact(bullet, env):
    return bool(bullet.getContactPoints(
        bodyA=int(env.DRONE_IDS[0]),
        bodyB=int(env.PLANE_ID),
        physicsClientId=int(env.CLIENT),
    ))


def evaluate_case(model, norm, case, args, method="bc"):
    """Run one closed-loop PyBullet flight on CPU; method='expert' uses the controller of get_dataset.py."""
    if method not in ("expert", "bc", "inner", "dp", "dp_inner", "dp_frobenius", "dp_nuclear"):
        raise ValueError(f"Unsupported method: {method}")
    if model is not None and hasattr(model, "reset"):
        # Reproducible per case, independent of method order or earlier flights.
        seed = np.random.SeedSequence([
            int(getattr(args, "sampling_seed", 2026)), int(case["case_id"])
        ]).generate_state(1)[0]
        model.reset(int(seed))
    import pybullet as p
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    original = case["initial_observation"]
    target = case["target"]
    with contextlib.redirect_stdout(io.StringIO()):
        env = CtrlAviary(
            drone_model=DroneModel.CF2X, num_drones=1,
            initial_xyzs=original[:3].reshape(1, 3).astype(float),
            initial_rpys=original[7:10].reshape(1, 3).astype(float),
            physics=Physics.PYB,
            pyb_freq=args.pyb_freq, ctrl_freq=args.ctrl_freq,
            gui=args.gui, record=False, obstacles=False, user_debug_gui=False,
        )
    try:
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        obs20 = np.asarray(obs[0], dtype=np.float32)
        if np.linalg.norm(obs20[:3] - original[:3]) > 0.01:
            raise RuntimeError("Reset XYZ differs from saved initial XYZ")
        if (np.linalg.norm(original[10:16]) > 0.03 or
                np.max(np.abs(original[16:20])) > 1.0):
            raise RuntimeError("First dataset row is not a stationary reset state")

        # Import expert dependencies only when evaluating the expert.
        if method == "expert":
            from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
            from get_dataset import expert_rpm

            expert_controller = DSLPIDControl(drone_model=DroneModel.CF2X)
            expert_controller.I_COEFF_FOR[:] = 0.0
            expert_controller.I_COEFF_TOR[:] = 0.0

        stable_steps = 0
        seen_contact = False
        first_contact_time = None
        first_contact_xy = None
        max_tilt_deg = 0.0
        clipped_components = 0
        total_components = 0
        success = False
        termination = "timeout"
        steps = 0
        first_descent_time_s = None
        approach_steps = 0
        descent_steps = 0
        if method == "expert":
            dim = 0
        else:
            input_dim = (model.state_dim if hasattr(model, "state_dim")
                         else model.pi_1.in_features)
            dim = input_dim - STATE_DIM
        base_clutter = episode_clutter_base(case["case_id"], dim,
                                            getattr(args, "clutter_seed", 2026))
        common_half_range = getattr(args, "clutter_half_range", 0.0)
        approach_half_range = getattr(args, "approach_clutter_half_range", None)
        descent_half_range = getattr(args, "descent_clutter_half_range", None)
        if approach_half_range is None:
            approach_half_range = common_half_range
        if descent_half_range is None:
            descent_half_range = common_half_range

        for step in range(args.max_steps):
            phase_is_descent = is_descent_phase(obs20, target)
            if phase_is_descent:
                descent_steps += 1
                if first_descent_time_s is None:
                    first_descent_time_s = step / args.ctrl_freq
            else:
                approach_steps += 1
            half_range = descent_half_range if phase_is_descent else approach_half_range
            clutter = base_clutter * half_range if dim else None
            if method == "expert":
                # Exactly the expert action rule used during data collection.
                # Its internal landed flag is NOT the evaluation success flag.
                raw_rpm, _expert_landed = expert_rpm(
                    obs20, target, expert_controller,
                    env.CTRL_TIMESTEP, env.MAX_RPM,
                )
            else:
                raw_rpm = model_rpm(model, norm, obs20, target, clutter)
            if not np.isfinite(raw_rpm).all():
                termination = "non_finite_action"
                break
            rpm = np.clip(raw_rpm, 0.0, float(env.MAX_RPM))
            clipped_components += int(np.count_nonzero(np.abs(raw_rpm - rpm) > 1e-6))
            total_components += ACTION_DIM
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
            if contact and not seen_contact:
                seen_contact = True
                first_contact_time = steps / args.ctrl_freq
                first_contact_xy = xy_error

            stable = (
                contact and xy_error <= args.xy_tol and z_error <= args.z_tol
                and speed <= args.speed_tol and tilt_deg <= args.tilt_tol_deg
            )
            stable_steps = stable_steps + 1 if stable else 0
            if stable_steps >= args.stable_steps:
                success = True
                termination = "stable_landing"
                break
            if terminated or truncated:
                termination = "environment_terminated"
                break

        final_pos = obs20[:3]
        final_vel = obs20[10:13]
        return {
            "case_id": int(case["case_id"]),
            "success": bool(success), "termination": termination,
            "steps": steps, "duration_s": steps / args.ctrl_freq,
            "final_xy_error_m": float(np.linalg.norm(final_pos[:2] - target[:2])),
            "final_z_error_m": float(abs(final_pos[2] - target[2])),
            "final_xyz_error_m": float(np.linalg.norm(final_pos - target)),
            "final_altitude_m": float(final_pos[2]),
            "final_speed_m_s": float(np.linalg.norm(final_vel)),
            "ground_contact_seen": bool(seen_contact),
            "entered_descent": first_descent_time_s is not None,
            "first_descent_time_s": first_descent_time_s,
            "approach_time_s": approach_steps / args.ctrl_freq,
            "descent_time_s": descent_steps / args.ctrl_freq,
            "first_contact_time_s": first_contact_time,
            "first_contact_xy_error_m": first_contact_xy,
            "max_tilt_deg": max_tilt_deg,
            "rpm_clipped_fraction": clipped_components / max(total_components, 1),
            "init_x": float(original[0]), "init_y": float(original[1]),
            "init_z": float(original[2]),
            "target_x": float(target[0]), "target_y": float(target[1]),
            "target_z": float(target[2]),
        }
    finally:
        env.close()


def summarize(records):
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    successful = [r for r in records if r["success"]]
    contacted = [r for r in records if r["ground_contact_seen"]]
    def mean(key, subset=records):
        return float(np.mean([r[key] for r in subset])) if subset else None
    return {
        "episodes": len(records),
        "successes": len(successful),
        "success_rate": len(successful) / len(records),
        "contact_rate": len(contacted) / len(records),
        "entered_descent_rate": sum(bool(r.get("entered_descent")) for r in records) / len(records),
        "mean_final_xy_error_m_all": mean("final_xy_error_m"),
        "mean_final_z_error_m_all": mean("final_z_error_m"),
        "mean_final_xyz_error_m_all": mean("final_xyz_error_m"),
        "median_final_xy_error_m_all": float(np.median([r["final_xy_error_m"] for r in records])),
        "mean_first_contact_xy_error_m": mean("first_contact_xy_error_m", contacted),
        "mean_landing_time_s_success_only": mean("duration_s", successful),
        "mean_rpm_clipped_fraction": mean("rpm_clipped_fraction"),
        "termination_counts": {k: sum(r["termination"] == k for r in records)
                               for k in sorted(set(r["termination"] for r in records))},
    }


def make_rollout_args(**overrides):
    """Defaults shared by training validation rollouts and test rollouts."""
    defaults = dict(
        pyb_freq=240, ctrl_freq=48, max_seconds=12.0,
        xy_tol=0.05, z_tol=0.06, speed_tol=0.15,
        tilt_tol_deg=20.0, stable_steps=10, gui=False,
        clutter_half_range=0.0, approach_clutter_half_range=None,
        descent_clutter_half_range=None, clutter_seed=2026,
        sampling_seed=2026,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults, max_steps=round(defaults["max_seconds"] * defaults["ctrl_freq"]))

