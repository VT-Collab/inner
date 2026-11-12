"""Collect one-drone expert demonstrations for precision landing.

Each saved row is [observation (20), landing_target (3), motor_rpm (4)].
The policy input is row[:23]; its supervised action label is row[23:27].
The dataset is a flat list of rows.
Trajectory lengths are saved separately for replay and visualization.

CtrlAviary's 20-value observation is:
  0:3   XYZ position (m)
  3:7   orientation quaternion, XYZW
  7:10  roll, pitch, yaw (rad)
  10:13 XYZ linear velocity (m/s)
  13:16 XYZ angular velocity (rad/s)
  16:20 previous four motor RPM commands
The target at 20:23 is the landing point [x, y, drone-center height] (m).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from policy import drone_support  # Select the local simulator checkout.

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from task import ALTITUDE, CONTROL_HZ, LANDING_HEIGHT, SIM_HZ, lengths_path


def expert_action(drone, target, speed_limit):
    """Outer position/velocity feedback produces a desired XYZ velocity (m/s)."""
    velocity = 1.2 * (target - drone[:3]) - 0.35 * drone[10:13]
    speed = np.linalg.norm(velocity)
    return velocity * min(1.0, speed_limit / max(speed, 1e-12))


def expert_rpm(drone, target, controller, control_timestep, max_rpm):
    """Compute motor RPM using only the current saved state and landing target."""
    horizontal_error = np.linalg.norm(drone[:2] - target[:2])
    horizontal_speed = np.linalg.norm(drone[10:12])
    descending = horizontal_error < 0.04 and horizontal_speed < 0.06
    landed = drone[2] < 0.016 and abs(drone[12]) < 0.10
    if landed:
        return np.zeros(4), True

    waypoint = target.copy()
    if not descending:
        waypoint[2] = ALTITUDE
    velocity = expert_action(drone, waypoint, 0.25)

    # DSLPIDControl normally remembers integral errors and the previous RPY.
    # Clear that history, disable its integral gains, and reconstruct the
    # current Euler rates from the angular velocity in the saved observation.
    controller.reset()
    roll, pitch, yaw = drone[7:10]
    body_rates = Rotation.from_quat(drone[3:7]).inv().apply(drone[13:16])
    p, q, r = body_rates
    cos_pitch = np.cos(pitch)
    euler_rates = np.array([
        p + np.sin(roll) * np.tan(pitch) * q + np.cos(roll) * np.tan(pitch) * r,
        np.cos(roll) * q - np.sin(roll) * r,
        (np.sin(roll) * q + np.cos(roll) * r) / cos_pitch,
    ])
    controller.last_rpy = drone[7:10] - euler_rates * control_timestep
    rpm, _, _ = controller.computeControl(
        control_timestep=control_timestep,
        cur_pos=drone[:3],
        cur_quat=drone[3:7],
        cur_vel=drone[10:13],
        cur_ang_vel=drone[13:16],
        target_pos=drone[:3],
        target_rpy=np.array([0.0, 0.0, yaw]),
        target_vel=velocity,
    )
    return np.clip(rpm, 0.0, max_rpm), False


def collect(n_demos, seed, steps, output, gui=False):
    rng = np.random.default_rng(seed)
    dataset = []
    episode_lengths = []
    for demo_index in range(n_demos):
        initial_position = np.array([
            rng.uniform(-0.10, 0.10),
            rng.uniform(-0.10, 0.10),
            ALTITUDE,
        ])
        target = np.array([
            rng.uniform(0.25, 0.45),
            rng.uniform(-0.25, 0.25),
            LANDING_HEIGHT,
        ])
        env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=initial_position[None, :],
            physics=Physics.PYB,
            pyb_freq=SIM_HZ,
            ctrl_freq=CONTROL_HZ,
            gui=gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        try:
            observation, _ = env.reset()
            controller = DSLPIDControl(drone_model=DroneModel.CF2X)
            controller.I_COEFF_FOR[:] = 0.0
            controller.I_COEFF_TOR[:] = 0.0
            demonstration = []
            steps_after_landing = 0
            for _ in range(steps):
                drone = observation[0]
                rpm, landed = expert_rpm(
                    drone, target, controller, env.CTRL_TIMESTEP, env.MAX_RPM
                )
                # Record (state, goal, action) before applying this exact RPM.
                # CtrlAviary expects one row per drone, hence rpm[None, :].
                demonstration.append(np.concatenate((drone, target, rpm)).tolist())
                observation, _, terminated, truncated, _ = env.step(rpm[None, :])
                if terminated or truncated:
                    break
                if landed:
                    steps_after_landing += 1
                    if steps_after_landing >= CONTROL_HZ:
                        break
            dataset.extend(demonstration)
            episode_lengths.append(len(demonstration))
            final_error = np.linalg.norm(observation[0, :2] - target[:2])
            print(f"demo {demo_index + 1}/{n_demos}: {len(demonstration)} pairs, "
                  f"landed={landed}, final XY error {final_error:.3f} m, "
                  f"height {observation[0, 2]:.3f} m")
        finally:
            env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(dataset, file)
    with lengths_path(output).open("wb") as file:
        pickle.dump(episode_lengths, file)
    print(f"saved {len(episode_lengths)} demos and {len(dataset)} pairs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=int, default=30)
    parser.add_argument("--steps", type=int, default=12 * CONTROL_HZ)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--usage", type=str, default="train", choices=["train", "test", "val"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.demos < 1 or args.steps < 1:
        parser.error("--demos and --steps must be positive")
    if args.output is None:
        args.output = Path(__file__).parent / "datasets" / f"{args.usage}_{args.demos}.pkl"
    collect(args.demos, args.seed, args.steps, args.output, args.gui)
