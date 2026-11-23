"""Render a 3-frame strip of one greedy episode (for reports where video is not shown).

Frames are chosen automatically: the start, the closest interaction with another
vehicle, and the end of the episode (finish line or collision).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from autodrive.algos import make_algorithm
from autodrive.baselines import PurePursuitDriver
from autodrive.env import MiniDriveEnv
from autodrive.logging_utils import advance_agent, reset_agent
from autodrive.render import render_frame

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(algo, env):
    if algo == "expert":
        return PurePursuitDriver(env)
    run_dir = os.path.join(HERE, "runs", f"{algo}_s0")
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    agent = make_algorithm(cfg["algo"], env.obs_dim, env.n_actions, cfg=cfg.get("algorithm_cfg"),
                           seed=cfg["seed"])
    agent.load_state_dict(dict(np.load(os.path.join(run_dir, "ckpt_best.npz"))))
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--seed", type=int, default=20_000)
    args = ap.parse_args()

    for algo in ("expert", "ppo", "dqn"):
        env = MiniDriveEnv(seed=0)
        agent = load(algo, env)
        obs, _ = env.reset(seed=args.seed)
        reset_agent(agent, obs)
        shots, gaps, done, ret = {}, [], False, 0.0
        while not done:
            a = agent.act(obs, greedy=True)
            obs, r, term, trunc, info = env.step(a)
            advance_agent(agent, obs, term, trunc)
            ret += r
            gap = info.get("npc_min_distance", float("inf"))
            gaps.append((gap, env.steps, ret, env.v))
            if env.steps == 1:
                shots[1] = render_frame(env, scale=5.0, hud=f"start   v={env.v:4.1f} m/s")
            done = term or trunc
        # closest interaction with another vehicle
        g, st, r_at, v_at = min(gaps, key=lambda t: t[0])
        env2 = MiniDriveEnv(seed=0)
        agent2 = load(algo, env2)
        obs, _ = env2.reset(seed=args.seed)
        reset_agent(agent2, obs)
        done, ret2 = False, 0.0
        while not done:
            a = agent2.act(obs, greedy=True)
            obs, r, term, trunc, info = env2.step(a)
            advance_agent(agent2, obs, term, trunc)
            ret2 += r
            if env2.steps == st:
                shots[st] = render_frame(env2, scale=5.0,
                                         hud=f"closest pass: gap {g:4.1f} m   v={env2.v:4.1f} m/s")
            done = term or trunc
        tags = ("finish line" if info["success"] else
                ("collision" if info["collision"] else "off track"))
        shots[10 ** 9] = render_frame(env2, scale=5.0, hud=f"{tags}   ret={ret2:6.1f}")

        frames = [shots[k] for k in sorted(shots)]
        w = max(f.width for f in frames)
        h = sum(f.height for f in frames) + 8 * (len(frames) - 1)
        strip = Image.new("RGB", (w, h), (255, 255, 255))
        y = 0
        for f in frames:
            strip.paste(f, (0, y))
            y += f.height + 8
        path = os.path.join(args.out, f"{algo}_strip.png")
        strip.save(path)
        print(f"wrote {path}  (return {ret:.1f}, closest gap {g:.1f} m at step {st}, {tags})")


if __name__ == "__main__":
    main()
