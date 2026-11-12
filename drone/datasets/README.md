# Drone precision landing demonstrations

Task: one Crazyflie drone starts airborne at 0.7 m, flies over a randomized
landing point on the ground, descends, and turns its motors off after touchdown.
There are no obstacles. A proportional velocity planner and a memoryless
version of `gym-pybullet-drones`'s `DSLPIDControl` compute four motor RPM commands.
The expert chooses whether to translate, descend, or stop from the current
state and target. Its integral gains are zero, and its rate term uses the
angular velocity in the current observation. This makes each action a function
of the saved state and target alone.
`CtrlAviary` applies those RPM commands directly to the simulator. The episode
records one second with motors off after touchdown.

After the install steps in the [main README](../README.md), collect demonstrations from `drone/`:

```bash
python get_dataset.py --demos 5 --seed 0
```

This writes `datasets/train_5.pkl`, a flat list of state-goal-action rows.
`datasets/train_5_lengths.pkl` stores the number of rows in each demonstration
so the visualization scripts can replay an individual flight. Each 27-value
row contains the simulator's complete drone state (20), landing point (3),
and expert motor RPM command (4). The landing point's Z coordinate is the
drone center height when resting on the ground.

| Row indices | Meaning | Units |
| --- | --- | --- |
| `0:3` | Drone XYZ position | m |
| `3:7` | Orientation quaternion `(x, y, z, w)` | unitless |
| `7:10` | Roll, pitch, yaw | rad |
| `10:13` | XYZ linear velocity | m/s |
| `13:16` | XYZ angular velocity | rad/s |
| `16:20` | Previous four motor commands | RPM |
| `20:23` | Landing point `(x, y, center height at rest)` | m |
| `23:27` | Current four motor commands, the imitation label | RPM |

The **state given to a policy** is `row[:23]`; the **action to predict** is
`row[23:27]`, in the simulator's motor order. Motor commands are bounded by
`[0, env.MAX_RPM]`, and zero RPM stops the motors. Each action is recorded
immediately before being applied to the simulator. The simulator advances at
48 control steps/s, and every state-action pair is saved.
The landing point is part of the policy input because it changes between
demonstrations. For training, normalize the state features and RPM labels using training-set
statistics; the saved dataset keeps physical units.

View a saved demonstration as an animated top-down GIF:

```bash
python visualize_demo.py --dataset datasets/train_5.pkl --demo 1
```

This writes `datasets/demo_1.gif`. To watch the drone move live in a window
showing PyBullet's 3D simulator camera, run:

```bash
python replay_demo.py --dataset datasets/train_5.pkl --demo 1
```

The live window uses PyBullet's software renderer, so it needs X11 but not
OpenGL. It loops by default. Left drag orbits the camera, right drag pans, and
the mouse wheel zooms. Press Space to pause, R to restart, C to reset the
camera, +/- to change speed, L to toggle looping, or Q to quit. `--once`
plays a single loop. The orange disk on the ground marks the target's XY
location. To save the 3D simulator view instead, run
`python replay_demo.py --dataset datasets/train_5.pkl --demo 1 --gif`; it writes
`datasets/simulator_demo_1.gif`. The native PyBullet GUI remains available
with `--gui` on a machine where OpenGL works. `--headless` runs only the
physics replay.

## Training and evaluation

The experiments in the [main README](../README.md) read the packaged datasets in `datasets/data/`, whose policy inputs omit the previous motor commands and include the irrelevant entries of each experiment.
The 27-column format above belongs to `get_dataset.py`, not to the packaged datasets.
