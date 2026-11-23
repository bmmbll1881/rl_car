"""Render the environment map (with lidar rays) for the report."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

from autodrive.env import MiniDriveEnv
from autodrive.baselines import PurePursuitDriver
from autodrive.render import render_frame, _font


def drive_to(env, step_target):
    drv = PurePursuitDriver(env)
    env.reset(seed=0)
    while env.steps < step_target:
        _, _, term, trunc, _ = env.step(drv.act())
        if term or trunc:
            break
    return env


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
    os.makedirs(out, exist_ok=True)
    env = MiniDriveEnv(seed=0)
    frames = []
    for target, label in ((0, "A: start / finish line, v=0"), (200, "B: mid-lap, tight corner")):
        drive_to(env, target)
        img = render_frame(env, scale=5.0, hud=f"{label}   v={env.v:4.1f} m/s")
        frames.append(img)
    w = max(f.width for f in frames)
    h = sum(f.height for f in frames) + 8
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for f in frames:
        canvas.paste(f, (0, y))
        y += f.height + 8
    path = os.path.join(out, "track_map.png")
    canvas.save(path)
    print("wrote", path, canvas.size)


if __name__ == "__main__":
    main()
