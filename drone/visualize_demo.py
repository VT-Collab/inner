"""Render a saved one-drone demonstration as an animated top-down GIF."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from task import CONTROL_HZ, RENDER_EVERY, load_demo


WIDTH, HEIGHT = 760, 700
PLOT = (80, 105, 680, 605)
BLUE = "#2563eb"
GOLD = "#ca8a04"


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_frame(positions, target, frame_index, frame_label="Demo"):
    xy = positions[:, :2]
    target_xy = target[:2]
    all_xy = np.vstack((xy, target_xy))
    minimum = all_xy.min(axis=0) - 0.10
    maximum = all_xy.max(axis=0) + 0.10
    span = np.maximum(maximum - minimum, 0.5)
    scale = min((PLOT[2] - PLOT[0]) / span[0], (PLOT[3] - PLOT[1]) / span[1])
    center = (minimum + maximum) / 2.0

    def point(xy_point):
        px = (PLOT[0] + PLOT[2]) / 2 + (xy_point[0] - center[0]) * scale
        py = (PLOT[1] + PLOT[3]) / 2 - (xy_point[1] - center[1]) * scale
        return (int(round(px)), int(round(py)))

    image = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.text((35, 23), "Drone precision landing", font=font(24), fill="#111827")
    draw.text((35, 59), f"{frame_label} frame {frame_index + 1}/{len(positions)}", font=font(15), fill="#4b5563")
    draw.rectangle(PLOT, outline="#d1d5db", width=2)

    goal = point(target_xy)
    draw.ellipse((goal[0] - 12, goal[1] - 12, goal[0] + 12, goal[1] + 12), outline=GOLD, width=3)
    draw.line((goal[0] - 8, goal[1], goal[0] + 8, goal[1]), fill=GOLD, width=2)
    draw.line((goal[0], goal[1] - 8, goal[0], goal[1] + 8), fill=GOLD, width=2)

    path = [point(p) for p in xy[:frame_index + 1]]
    if len(path) > 1:
        draw.line(path, fill=BLUE, width=3)
    start = point(xy[0])
    draw.ellipse((start[0] - 6, start[1] - 6, start[0] + 6, start[1] + 6),
                 fill="#ffffff", outline=BLUE, width=2)
    current = point(xy[frame_index])
    draw.ellipse((current[0] - 11, current[1] - 11, current[0] + 11, current[1] + 11),
                 fill=BLUE, outline="#111827", width=2)

    error = np.linalg.norm(positions[frame_index, :2] - target[:2])
    elapsed = frame_index / CONTROL_HZ
    draw.text((80, 624), f"Time: {elapsed:.1f} s   XY error: {error:.3f} m   Height: {positions[frame_index, 2]:.3f} m",
              font=font(16), fill="#111827")
    draw.text((80, 653), "Blue: drone path     Gold: target",
              font=font(14), fill="#4b5563")
    return image


def visualize(dataset_path, demo_number, output, fps):
    data = load_demo(dataset_path, demo_number)
    positions = data[:, :3]
    target = data[0, 20:23]
    frame_indices = list(range(0, len(data), RENDER_EVERY))
    if frame_indices[-1] != len(data) - 1:
        frame_indices.append(len(data) - 1)
    frames = [draw_frame(positions, target, i) for i in frame_indices]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=1000 / fps, loop=0)
    print(f"saved {len(frames)} frames from demo {demo_number} to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "datasets/dataset.pkl")
    parser.add_argument("--demo", type=int, default=1, help="one-based demonstration number")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    output = args.output or Path(__file__).parent / f"datasets/demo_{args.demo}.gif"
    visualize(args.dataset, args.demo, output, args.fps)
