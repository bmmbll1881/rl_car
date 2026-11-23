#!/usr/bin/env python
"""Check that a curriculum environment lets the ego actually catch a blocked lead.

This is an *environment* check, not a policy evaluation: the ego is driven by the
scripted :class:`PurePursuitDriver`, so the numbers describe the traffic layout
only.  For every step it reports whether the ego lane was blocked
(``lane_lead`` closer than ``--block-distance``) and whether the ego was closing
on that lead (ego speed > lead speed + 0.5 m/s).

Examples
--------
    python scripts/check_curriculum.py --env-json configs/env_overtake_stage1.json
    python scripts/check_curriculum.py --env-json configs/env_overtake_stage2.json \
        --episodes 10 --seed-base 20000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from autodrive.baselines import PurePursuitDriver
from autodrive.env import MiniDriveEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-json", required=True, help="environment overrides to check")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=20_000)
    p.add_argument("--block-distance", type=float, default=18.0,
                   help="[m] ego lane counts as blocked below this lead gap")
    args = p.parse_args()

    with open(args.env_json, encoding="utf-8") as fh:
        overrides = json.load(fh)
    env = MiniDriveEnv(seed=0, **overrides)
    driver = PurePursuitDriver(env)

    speeds, gaps = [], []
    blocked = closing = 0
    blocked_episodes = successes = collisions = steps = 0
    for ep in range(args.episodes):
        env.reset(seed=args.seed_base + ep)
        driver.reset()
        ep_blocked = False
        done = False
        while not done:
            gap, lead_speed = env.obstacles.lane_lead(env.lane_of(), env.prev_s)
            if gap < args.block_distance:
                blocked += 1
                gaps.append(float(gap))
                ep_blocked = True
                if env.v > lead_speed + 0.5:
                    closing += 1
            _, _, term, trunc, info = env.step(driver.act())
            speeds.append(float(info["speed"]))
            steps += 1
            done = term or trunc
        blocked_episodes += int(ep_blocked)
        successes += int(info["success"])
        collisions += int(info["collision"])

    print(json.dumps({
        "env_json": args.env_json,
        "npc_count": int(env.obstacles.n),
        "npc_speeds": [float(env.cfg["npc_v_min"]), float(env.cfg["npc_v_max"])],
        "ego_v_max": float(env.cfg["v_max"]),
        "episodes": args.episodes,
        "steps": steps,
        "ego_speed_mean": round(float(np.mean(speeds)), 3),
        "blocked_steps": blocked,
        "blocked_frac": round(blocked / max(steps, 1), 4),
        "closing_steps": closing,
        "closing_frac_of_blocked": round(closing / max(blocked, 1), 4),
        "mean_lead_gap_when_blocked": round(float(np.mean(gaps)), 2) if gaps else None,
        "episodes_blocked": blocked_episodes,
        "expert_success_rate": successes / max(args.episodes, 1),
        "expert_collision_rate": collisions / max(args.episodes, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
