#!/usr/bin/env python
"""Evaluate a saved agent: metrics table, per-step trace, and an animated GIF.

Examples
--------
    python evaluate.py --run runs/dqn_s0 --episodes 10
    python evaluate.py --run runs/ppo_s0 --ckpt ckpt_best.npz --gif out/ppo.gif
    python evaluate.py --algo expert --gif out/expert.gif
    python evaluate.py --run runs/dqn_s0 --trace out/trace.csv --episode 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from autodrive.algos import make_algorithm
from autodrive.baselines import PurePursuitDriver
from autodrive.env import MiniDriveEnv, ACTION_NAMES
from autodrive.logging_utils import (OvertakeTracker, advance_agent, policy_lane_snapshot,
                                     reset_agent)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_agent(args, env):
    if args.algo in ("expert", "random"):
        if args.algo == "expert":
            return PurePursuitDriver(env)
        rng = np.random.default_rng(0)

        class RandomAgent:
            def act(self, obs, greedy=False):
                return int(rng.integers(env.n_actions))

        return RandomAgent()

    run_dir = args.run or os.path.join(HERE, "runs", f"{args.algo}_s0")
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        run_cfg = json.load(fh)
    ckpt = os.path.join(run_dir, args.ckpt)
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    algo_cfg = run_cfg.get("algorithm_cfg")
    if getattr(args, "config", None):
        with open(args.config, encoding="utf-8") as fh:
            algo_cfg = json.load(fh)
    agent = make_algorithm(run_cfg["algo"], env.obs_dim, env.n_actions,
                           cfg=algo_cfg, seed=run_cfg["seed"])
    try:
        agent.load_state_dict(dict(np.load(ckpt)))
    except ValueError as exc:
        raise SystemExit(
            f"checkpoint {ckpt} is incompatible with this environment "
            f"({env.obs_dim}-dim observation): {exc}\n"
            f"It was most likely trained on an older environment version "
            f"(the checkpoint run used n_obstacles={run_cfg.get('env_cfg', {}).get('n_obstacles')}, "
            f"n_lanes={run_cfg.get('env_cfg', {}).get('n_lanes', 'n/a')}).\n"
            f"Re-train with:  python scripts/run_batch.py --seeds 0 1 2 3 4 --steps 250000"
        )
    # environment overrides used for training must be reproduced
    return agent, run_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="run directory (with config.json + checkpoint)")
    p.add_argument("--algo", default=None, help="expert | random (scripted baselines)")
    p.add_argument("--ckpt", default="ckpt_best.npz")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=10_000)
    p.add_argument("--gif", default=None)
    p.add_argument("--video", default=None, help="write a .avi (Motion-JPEG) of the episode")
    p.add_argument("--fps", type=int, default=10, help="video frame rate (10 = real time)")
    p.add_argument("--gif-episodes", type=int, default=1)
    p.add_argument("--gif-stride", type=int, default=2, help="store every n-th frame")
    p.add_argument("--trace", default=None)
    p.add_argument("--episode", type=int, default=0, help="episode index to trace / animate")
    p.add_argument("--obstacles", type=int, default=None, help="override number of NPC vehicles")
    p.add_argument("--env-json", default=None, help="JSON file with environment overrides")
    p.add_argument("--config", default=None,
                   help="override the stored algorithm config (e.g. enable a TTC shield)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.gif or args.video:
        from autodrive.render import make_gif, render_frame
        from autodrive.video import write_mjpeg_avi
    else:
        make_gif = render_frame = write_mjpeg_avi = None

    algo = args.algo
    run_cfg = None
    if args.run:
        with open(os.path.join(args.run, "config.json"), encoding="utf-8") as fh:
            run_cfg = json.load(fh)
        algo = algo or run_cfg["algo"]
    algo = algo or "dqn"
    env_over = dict((run_cfg or {}).get("env_cfg", {}))
    if args.env_json:
        with open(args.env_json, encoding="utf-8") as fh:
            env_over.update(json.load(fh))
    if args.obstacles is not None:
        env_over["n_obstacles"] = args.obstacles
    try:
        env = MiniDriveEnv(seed=0, **env_over)
    except ValueError as exc:
        raise SystemExit(
            f"this run's stored environment configuration is from an older version: {exc}\n"
            f"Re-train it on the current environment:  "
            f"python scripts/run_batch.py --seeds 0 1 2 3 4 --steps 250000"
        )

    loaded = load_agent(argparse.Namespace(
        algo=algo, run=args.run, ckpt=args.ckpt, config=args.config
    ), env)
    agent = loaded[0] if isinstance(loaded, tuple) else loaded
    if hasattr(agent, "bind_env"):
        agent.bind_env(env)      # RARL re-installs its (frozen) adversarial vehicle

    rows = []
    frames = []
    tracker = OvertakeTracker()
    for ep in range(args.episodes):
        seed = args.seed_base + ep
        obs, info = env.reset(seed=seed)
        reset_agent(agent, obs)
        tracker.start_episode(env)
        ret, done, steps = 0.0, False, 0
        speeds, lats, acts = [], [], []
        trace_rows = [] if (args.trace and ep == args.episode) else None
        capture = (args.gif or args.video) and ep < args.gif_episodes
        stride = 1 if args.video else args.gif_stride
        while not done:
            a = agent.act(obs, greedy=True)
            lane = policy_lane_snapshot(agent)
            obs, r, term, trunc, info = env.step(a)
            # settle the agent first: it sets ``_lane_bonus_taken`` for the
            # verified lane change this step, which the tracker must see
            advance_agent(agent, obs, term, trunc)
            tracker.observe(env, a, lane=lane,
                            bonus=float(getattr(agent, "_lane_bonus_taken", 0.0)))
            ret += r
            steps += 1
            speeds.append(info["speed"])
            lats.append(abs(info["lateral"]))
            acts.append(a)
            if trace_rows is not None:
                lane_data = getattr(agent, "last_lane", {})
                trace_rows.append(
                    dict(step=steps, x=round(env.x, 3), y=round(env.y, 3), theta=round(env.theta, 4),
                         v=round(env.v, 3), action=a, action_name=ACTION_NAMES[a], reward=round(r, 4),
                         lateral=round(info["lateral"], 3), lane=env.lane_of(),
                         best_lane=lane_data.get("best_lane", ""),
                         current_gap=round(float(lane_data.get("current_gap", 0.0)), 2),
                         best_gap=round(float(lane_data.get("best_gap", 0.0)), 2),
                         overtake=int(lane_data.get("overtake_available", False)),
                         lock=int(getattr(agent, "_overtake_target", None) is not None),
                         hold_steps=int(getattr(agent, "_overtake_hold_steps", 0)),
                         progress=round(info["progress"], 2),
                         crash=int(info["crash"]), success=int(info["success"]))
                )
            if capture and steps % stride == 0:
                frames.append(
                    render_frame(env, scale=4.0,
                                 hud=f"step {steps:3d}  v={env.v:4.1f} m/s  ret={ret:7.1f}  "
                                     f"gap={info.get('npc_min_distance', float('nan')):4.1f} m")
                )
            done = term or trunc
        rows.append(
            dict(episode=ep, seed=seed, return_=round(ret, 3), steps=steps, success=int(info["success"]),
                 crash=int(info["crash"]), collision=int(info.get("collision", False)),
                 distance=round(info["progress"], 1),
                 mean_speed=round(float(np.mean(speeds)), 3),
                 lateral_rms=round(float(np.sqrt(np.mean(np.square(lats)))), 3),
                 lap_time_s=round(steps * env.dt, 1))
        )
        if trace_rows is not None and args.trace:
            os.makedirs(os.path.dirname(os.path.abspath(args.trace)), exist_ok=True)
            with open(args.trace, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(trace_rows[0].keys()))
                w.writeheader()
                w.writerows(trace_rows)
            print(f"[trace] {len(trace_rows)} steps -> {args.trace}")

    if args.gif and frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.gif)), exist_ok=True)
        make_gif(frames, args.gif, duration=70)
        print(f"[gif] {len(frames)} frames -> {args.gif}")
    if args.video and frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
        write_mjpeg_avi(args.video, frames, fps=args.fps)
        print(f"[video] {len(frames)} frames @ {args.fps} fps -> {args.video}")

    agg = {
        "algorithm": algo,
        "episodes": args.episodes,
        "return_mean": float(np.mean([r["return_"] for r in rows])),
        "return_std": float(np.std([r["return_"] for r in rows])),
        "return_min": float(np.min([r["return_"] for r in rows])),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "crash_rate": float(np.mean([r["crash"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "distance_mean": float(np.mean([r["distance"] for r in rows])),
        "speed_mean": float(np.mean([r["mean_speed"] for r in rows])),
        "lap_time_mean": float(np.mean([r["lap_time_s"] for r in rows if r["success"]])) if any(
            r["success"] for r in rows) else float("nan"),
        "lateral_rms": float(np.mean([r["lateral_rms"] for r in rows])),
    }
    agg.update(tracker.summary())

    if not args.quiet:
        print(f"\n=== {algo} | {args.episodes} episodes | seeds {args.seed_base}.. ===")
        print(f"{'ep':>3} {'seed':>6} {'return':>9} {'steps':>6} {'dist':>7} {'v':>5} {'lat':>5} {'success':>7} {'crash':>5}")
        for r in rows:
            print(f"{r['episode']:3d} {r['seed']:6d} {r['return_']:9.2f} {r['steps']:6d} {r['distance']:7.1f} "
                  f"{r['mean_speed']:5.2f} {r['lateral_rms']:5.2f} {r['success']:7d} {r['crash']:5d}")
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                          for k, v in agg.items()}, indent=2))
    else:
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()}))


if __name__ == "__main__":
    main()
