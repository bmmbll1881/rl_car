#!/usr/bin/env python
"""Record a full autonomous-driving episode as a video (Motion-JPEG .avi).

    python scripts/make_video.py --run runs/ppo_s0 --out figs/video_ppo_lap.avi
    python scripts/make_video.py --algo expert --out figs/video_expert_lap.avi
    python scripts/make_video.py --run runs/dqn_s0 --obstacles 0 --out figs/video_dqn_empty.avi

Every control step becomes one frame (dt = 0.1 s  ->  10 fps = real time), so the
video covers the **whole episode**: start, traffic interactions, the finish line
(or the crash, if the policy fails).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.algos import make_algorithm
from autodrive.baselines import PurePursuitDriver
from autodrive.env import ACTION_NAMES, MiniDriveEnv
from autodrive.logging_utils import advance_agent, reset_agent
from autodrive.render import render_frame
from autodrive.video import write_mjpeg_avi


def load_agent(args, env):
    if args.algo in ("expert", "random"):
        if args.algo == "expert":
            return PurePursuitDriver(env), None
        rng = np.random.default_rng(0)

        class RandomAgent:
            def act(self, obs, greedy=False):
                return int(rng.integers(env.n_actions))

        return RandomAgent(), None
    run_dir = args.run
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    agent = make_algorithm(cfg["algo"], env.obs_dim, env.n_actions,
                           cfg=cfg.get("algorithm_cfg"), seed=cfg["seed"])
    ckpt = os.path.join(run_dir, args.ckpt)
    try:
        agent.load_state_dict(dict(np.load(ckpt)))
    except ValueError as exc:
        raise SystemExit(
            f"checkpoint {ckpt} is incompatible with this environment "
            f"({env.obs_dim}-dim observation): {exc}\n"
            f"Re-train first:  python scripts/run_batch.py --seeds 0 1 2 3 4 --steps 250000\n"
            f"(or record the scripted driver:  --algo expert)"
        )
    return agent, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--algo", default=None, help="expert | random (scripted baselines)")
    ap.add_argument("--ckpt", default="ckpt_best.npz")
    ap.add_argument("--seed", type=int, default=10_000, help="episode seed")
    ap.add_argument("--episodes", type=int, default=1, help="how many episodes to concatenate")
    ap.add_argument("--obstacles", type=int, default=None, help="override number of NPC vehicles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    algo = args.algo
    run_cfg = None
    if args.run:
        with open(os.path.join(args.run, "config.json"), encoding="utf-8") as fh:
            run_cfg = json.load(fh)
        algo = algo or run_cfg["algo"]
    env_over = dict((run_cfg or {}).get("env_cfg", {}))
    if args.obstacles is not None:
        env_over["n_obstacles"] = args.obstacles
    try:
        env = MiniDriveEnv(seed=0, **env_over)
    except ValueError as exc:
        raise SystemExit(
            f"this run's stored environment configuration is from an older version: {exc}\n"
            f"Either re-train it, or record the scripted driver with  --algo expert"
        )

    agent, _ = load_agent(args, env)
    if hasattr(agent, "bind_env"):
        agent.bind_env(env)      # RARL re-installs its (frozen) adversarial vehicle
    frames, summary = [], []
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, _ = env.reset(seed=seed)
        reset_agent(agent, obs)
        ret, done, steps = 0.0, False, 0
        while not done:
            a = agent.act(obs, greedy=True)
            obs, r, term, trunc, info = env.step(a)
            advance_agent(agent, obs, term, trunc)
            ret += r
            steps += 1
            frames.append(
                render_frame(
                    env, scale=args.scale, show_lidar=True,
                    hud=(f"{args.title}ep {ep + 1}/{args.episodes}  step {steps:3d}  "
                         f"v={env.v:4.1f} m/s  ret={ret:7.1f}  "
                         f"gap={info['npc_min_distance']:4.1f} m  {ACTION_NAMES[a]}"),
                )
            )
            done = term or trunc
        summary.append((seed, ret, steps, info["success"], info["collision"], info["crash"]))
        print(f"  ep{ep + 1} seed={seed} return={ret:7.2f} steps={steps} "
              f"success={info['success']} collision={info['collision']} crash={info['crash']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_mjpeg_avi(args.out, frames, fps=args.fps, quality=args.quality)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[video] {len(frames)} frames @ {args.fps} fps -> {args.out}  ({size_mb:.1f} MB)"
          f"  [{frames[0].size[0]}x{frames[0].size[1]}]")


if __name__ == "__main__":
    main()
