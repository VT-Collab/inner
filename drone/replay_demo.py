"""Replay saved commands in a live PyBullet software-rendered window."""

import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from policy import drone_support  # Select the local simulator checkout.

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from task import CONTROL_HZ, RENDER_EVERY, SIM_HZ, load_demo


def render_frame(client, view, projection, elapsed, title="Simulator replay"):
    """Render the simulated drone with PyBullet's software renderer."""
    width, height = 640, 480
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, viewMatrix=view, projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER, physicsClientId=client,
    )
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 270, 36), fill="#ffffff")
    draw.text((12, 9), f"{title}  |  {elapsed:.1f} s", fill="#111827",
              font=ImageFont.load_default())
    return image


def add_ground_marker(client, target):
    marker_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=0.06, length=0.006,
        rgbaColor=[1.0, 0.65, 0.0, 1.0], physicsClientId=client,
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_visual,
                      basePosition=[target[0], target[1], 0.015],
                      physicsClientId=client)


def live_viewer(env, data, client, target, midpoint, projection,
                loop=True, exit_on_finish=False, realtime=True):
    """Run the simulator and camera in a responsive Tk window."""
    import tkinter as tk
    from PIL import ImageTk

    window = tk.Tk()
    window.title("PyBullet drone replay")
    image_label = tk.Label(window)
    image_label.pack()
    controls = tk.Frame(window)
    controls.pack(fill="x")
    status = tk.StringVar()
    loop_enabled = tk.BooleanVar(value=loop and not exit_on_finish)
    camera = {"focus": midpoint.copy(), "yaw": 35.0, "pitch": -35.0, "distance": 0.9}
    state = {"index": 0, "loops": 0, "paused": False, "speed": 1.0, "drag": None}

    def show_frame():
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera["focus"].tolist(),
            distance=camera["distance"], yaw=camera["yaw"],
            pitch=camera["pitch"], roll=0, upAxisIndex=2,
        )
        elapsed = state["index"] / CONTROL_HZ
        frame = render_frame(client, view, projection, elapsed)
        photo = ImageTk.PhotoImage(frame)
        image_label.configure(image=photo)
        image_label.image = photo
        status.set(
            f"Frame {state['index']}/{len(data)}   Loop {state['loops'] + 1}   "
            f"Speed {state['speed']:g}x   {'Paused' if state['paused'] else 'Playing'}"
        )

    def restart():
        env.reset()
        add_ground_marker(client, target)
        state["index"] = 0
        state["paused"] = False
        show_frame()

    def toggle_pause():
        state["paused"] = not state["paused"]
        show_frame()

    def reset_camera():
        camera.update(focus=midpoint.copy(), yaw=35.0, pitch=-35.0, distance=0.9)
        show_frame()

    def on_press(event, button):
        state["drag"] = (button, event.x, event.y)

    def on_drag(event):
        if state["drag"] is None:
            return
        button, old_x, old_y = state["drag"]
        dx, dy = event.x - old_x, event.y - old_y
        state["drag"] = (button, event.x, event.y)
        if button == 1:
            camera["yaw"] += dx * 0.5
            camera["pitch"] = float(np.clip(camera["pitch"] + dy * 0.5, -85, -5))
        else:
            angle = np.deg2rad(camera["yaw"])
            right = np.array([np.cos(angle), -np.sin(angle), 0.0])
            up = np.array([np.sin(angle), np.cos(angle), 0.0])
            camera["focus"] += (-dx * right + dy * up) * camera["distance"] * 0.002
        show_frame()

    def zoom(factor):
        camera["distance"] = float(np.clip(camera["distance"] * factor, 0.25, 4.0))
        show_frame()

    def on_key(event):
        key = event.keysym.lower()
        if key == "space":
            toggle_pause()
        elif key == "r":
            restart()
        elif key == "l":
            loop_enabled.set(not loop_enabled.get())
        elif key == "c":
            reset_camera()
        elif key in ("plus", "equal"):
            state["speed"] = min(state["speed"] * 2, 4.0)
            show_frame()
        elif key == "minus":
            state["speed"] = max(state["speed"] / 2, 0.25)
            show_frame()
        elif key in ("escape", "q"):
            window.destroy()

    tk.Button(controls, text="Pause / Play", command=toggle_pause).pack(side="left")
    tk.Button(controls, text="Restart", command=restart).pack(side="left")
    tk.Checkbutton(controls, text="Loop", variable=loop_enabled).pack(side="left")
    tk.Label(controls, textvariable=status).pack(side="right")
    tk.Label(window, text="Left drag: orbit   Right drag: pan   Wheel: zoom   "
                          "Space: pause   R: restart   C: reset camera   +/-: speed   Q: quit").pack()
    image_label.bind("<ButtonPress-1>", lambda event: on_press(event, 1))
    image_label.bind("<ButtonPress-3>", lambda event: on_press(event, 3))
    image_label.bind("<B1-Motion>", on_drag)
    image_label.bind("<B3-Motion>", on_drag)
    image_label.bind("<ButtonRelease-1>", lambda event: state.update(drag=None))
    image_label.bind("<ButtonRelease-3>", lambda event: state.update(drag=None))
    image_label.bind("<MouseWheel>", lambda event: zoom(0.9 if event.delta > 0 else 1.1))
    image_label.bind("<Button-4>", lambda event: zoom(0.9))
    image_label.bind("<Button-5>", lambda event: zoom(1.1))
    window.bind("<KeyPress>", on_key)
    show_frame()

    def tick():
        if state["paused"]:
            window.after(50, tick)
            return
        if state["index"] >= len(data):
            final_position = env._computeObs()[0, :3]
            error = np.linalg.norm(final_position[:2] - target[:2])
            print(f"completed loop {state['loops'] + 1}; landing XY error: {error:.3f} m, "
                  f"height: {final_position[2]:.3f} m")
            if exit_on_finish:
                window.destroy()
                return
            if loop_enabled.get():
                state["loops"] += 1
                restart()
            else:
                state["paused"] = True
                show_frame()
                window.after(50, tick)
                return
        started = time.monotonic()
        for _ in range(RENDER_EVERY):
            if state["index"] >= len(data):
                break
            env.step(data[state["index"], 23:27][None, :])
            state["index"] += 1
        show_frame()
        delay = 1 if not realtime else max(
            1, int(1000 * RENDER_EVERY / CONTROL_HZ / state["speed"]
                   - 1000 * (time.monotonic() - started))
        )
        window.after(delay, tick)

    window.after(1, tick)
    window.mainloop()


def replay(dataset_path, demo_number, mode="live", output=None, realtime=True,
           exit_on_finish=False, loop=True):
    data = load_demo(dataset_path, demo_number)
    start_position = data[0, :3]
    target = data[0, 20:23]
    gui = mode == "gui"
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=start_position[None, :],
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
        client = env.getPyBulletClient()
        midpoint = (start_position + target) / 2
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.25, cameraYaw=35, cameraPitch=-45,
                cameraTargetPosition=midpoint.tolist(), physicsClientId=client,
            )
            marker = target.copy()
            marker[2] += 0.03
            p.addUserDebugPoints([marker.tolist()], [[1.0, 0.7, 0.0]], pointSize=18,
                                 physicsClientId=client)
            p.addUserDebugText("TARGET", (marker + [0, 0, 0.07]).tolist(),
                               textColorRGB=[0.9, 0.5, 0.0], textSize=1.5,
                               physicsClientId=client)
            for a, b in zip(data[:-1, :3], data[1:, :3]):
                p.addUserDebugLine(a.tolist(), b.tolist(), [0.2, 0.5, 1.0],
                                   lineWidth=2, physicsClientId=client)
        elif mode in ("live", "render"):
            add_ground_marker(client, target)
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=midpoint.tolist(), distance=0.9,
                yaw=35, pitch=-35, roll=0, upAxisIndex=2,
            )
            projection = p.computeProjectionMatrixFOV(
                fov=60, aspect=640 / 480, nearVal=0.01, farVal=10,
            )
            if mode == "render":
                frames = [render_frame(client, view, projection, 0.0)]
            else:
                live_viewer(env, data, client, target, midpoint, projection,
                            loop=loop, exit_on_finish=exit_on_finish, realtime=realtime)
                return
        for sample_index, row in enumerate(data):
            start_time = time.monotonic()
            observation, _, terminated, truncated, _ = env.step(row[23:27][None, :])
            if gui and realtime:
                time.sleep(max(0.0, 1 / CONTROL_HZ - (time.monotonic() - start_time)))
            if mode == "render" and ((sample_index + 1) % RENDER_EVERY == 0
                                     or sample_index + 1 == len(data)):
                frame = render_frame(
                    client, view, projection, (sample_index + 1) / CONTROL_HZ,
                )
                frames.append(frame)
            if terminated or truncated:
                break
        error = np.linalg.norm(observation[0, :2] - target[:2])
        print(f"replayed demo {demo_number}; landing XY error: {error:.3f} m, "
              f"height: {observation[0, 2]:.3f} m")
        if mode == "render":
            output = output or Path(__file__).parent / f"datasets/simulator_demo_{demo_number}.gif"
            output.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(output, save_all=True, append_images=frames[1:],
                           duration=1000 / 10, loop=0)
            print(f"saved {len(frames)} simulator-rendered frames to {output}")
        if gui:
            input("Replay finished. Press Enter to close the simulator...")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "datasets/dataset.pkl")
    parser.add_argument("--demo", type=int, default=1, help="one-based demonstration number")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="open the native PyBullet GUI (requires OpenGL)")
    mode.add_argument("--gif", action="store_true", help="save the 3D simulator replay as a GIF")
    mode.add_argument("--headless", action="store_true", help="run physics without rendering")
    parser.add_argument("--output", type=Path, default=None, help="output path for the 3D GIF")
    parser.add_argument("--fast", action="store_true", help="skip real-time pacing")
    parser.add_argument("--exit-on-finish", action="store_true", help="close the live window after replay")
    parser.add_argument("--once", action="store_true", help="play once instead of looping in the live window")
    args = parser.parse_args()
    selected_mode = "gui" if args.gui else "render" if args.gif else "headless" if args.headless else "live"
    replay(args.dataset, args.demo, mode=selected_mode, output=args.output,
           realtime=not args.fast, exit_on_finish=args.exit_on_finish, loop=not args.once)
