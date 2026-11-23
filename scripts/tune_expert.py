"""Validate the scripted reference driver on the fixed evaluation seeds (dev helper).

``train.py`` evaluates every agent on seeds 10000..10000+N-1 with the greedy
policy; this script runs the pure-pursuit reference on the same seeds so the
reported "expert" numbers can be reproduced independently.

    python scripts/tune_expert.py                 # default parameters, 10 seeds
    python scripts/tune_expert.py --grid          # small sweep
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.env import MiniDriveEnv
from autodrive.baselines import PurePursuitDriver


def run(drv_kwargs, n=10, base_seed=10_000, obstacles=3):
    env = MiniDriveEnv(seed=0, n_obstacles=obstacles)
    returns, laps, crashes, cols, speeds, gaps = [], [], 0, 0, [], []
    for i in range(n):
        drv = PurePursuitDriver(env, **drv_kwargs)
        obs, _ = env.reset(seed=base_seed + i)
        ret, done, info, v, g = 0.0, False, {}, [], 99.0
        while not done:
            obs, r, term, trunc, info = env.step(drv.act())
            ret += r
            v.append(info["speed"])
            g = min(g, info["npc_min_distance"])
            done = term or trunc
        returns.append(ret)
        laps.append(info["success"])
        crashes += int(info["crash"])
        cols += int(info["collision"])
        speeds.append(np.mean(v))
        gaps.append(g)
    return (np.mean(returns), float(np.mean(laps)), cols / n, crashes / n,
            float(np.mean(speeds)), float(np.mean(gaps)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--obstacles", type=int, default=3)
    args = ap.parse_args()

    if not args.grid:
        r = run({}, obstacles=args.obstacles)
        print(f"default -> return {r[0]:7.2f}  success {r[1]:.2f}  collision {r[2]:.2f} "
              f"crash {r[3]:.2f}  speed {r[4]:5.2f}  min_gap {r[5]:4.1f}")
    else:
        grid = [
            dict(avoid_offset=3.0, avoid_range=26.0, brake_range=8.0),
            dict(avoid_offset=2.4, avoid_range=26.0, brake_range=8.0),
            dict(avoid_offset=3.4, avoid_range=30.0, brake_range=9.0),
            dict(avoid_offset=3.0, avoid_range=30.0, brake_range=9.0, v_target=8.0),
            dict(avoid_offset=2.6, avoid_range=22.0, brake_range=7.0, a_lat_max=3.0),
        ]
        for g in grid:
            r = run(g, obstacles=args.obstacles)
            print(f"{g} -> return {r[0]:7.2f}  success {r[1]:.2f}  collision {r[2]:.2f} "
                  f"crash {r[3]:.2f}  speed {r[4]:5.2f}  min_gap {r[5]:4.1f}")
