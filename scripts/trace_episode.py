"""Trace one scripted episode step by step -- the environment debugging tool."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.env import MiniDriveEnv, ACTION_NAMES
from autodrive.baselines import PurePursuitDriver


def main(seed=0, every=20, algo="expert"):
    env = MiniDriveEnv(seed=seed)
    print(env.describe())
    drv = PurePursuitDriver(env)
    obs, _ = env.reset(seed=seed)
    acts = Counter()
    ret = 0.0
    print(f"{'step':>5} {'act':>11} {'v':>5} {'lat':>6} {'ds':>6} {'prog':>7} {'ret':>8}")
    done = False
    while not done:
        a = drv.act()
        acts[ACTION_NAMES[a]] += 1
        obs, r, term, trunc, info = env.step(a)
        ret += r
        if env.steps % every == 0 or term or trunc:
            print(
                f"{env.steps:5d} {ACTION_NAMES[a]:>11} {info['speed']:5.2f} {info['lateral']:6.2f} "
                f"{info['ds']:6.2f} {info['progress']:7.1f} {ret:8.2f}"
            )
        done = term or trunc
    print(
        f"\nexpert result: return={ret:.2f} steps={env.steps} progress={env.s_total:.1f}/"
        f"{env.track.length:.1f} m success={info['success']} crash={info['crash']}"
    )
    print("action histogram:", dict(acts))


if __name__ == "__main__":
    main()
