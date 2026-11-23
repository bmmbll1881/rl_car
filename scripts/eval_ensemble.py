#!/usr/bin/env python
"""Evaluate an ensemble of independently trained checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from autodrive.algos import make_algorithm
from autodrive.env import ACTION_NAMES, MiniDriveEnv
from autodrive.logging_utils import advance_agent, reset_agent


def _policy_logits(agent, obs):
    if hasattr(agent, "_features"):
        policy_obs = agent._features(obs)
        return agent.policy.actor.forward(policy_obs)
    return agent.actor.forward(obs)


def _masked_probabilities(agent, obs):
    logits = _policy_logits(agent, obs)
    if hasattr(agent, "_action_mask"):
        mask = agent._action_mask()
        logits = np.where(mask, logits, -1e9)
    logits = logits - np.max(logits)
    p = np.exp(logits)
    return p / p.sum()


def combine_probabilities(probabilities, method="mean_prob"):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2:
        raise ValueError("probabilities must be [members, actions]")
    mean = p.mean(axis=0)
    if method == "mean_prob":
        return int(np.argmax(mean))
    if method == "majority":
        votes = np.argmax(p, axis=1)
        counts = np.bincount(votes, minlength=p.shape[1])
        return int(np.argmax(counts + 1e-9 * mean))
    raise ValueError(f"unknown ensemble method: {method}")


def load_member(run_dir, env, ckpt):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    agent = make_algorithm(cfg["algo"], env.obs_dim, env.n_actions,
                           cfg=cfg.get("algorithm_cfg"), seed=cfg["seed"])
    agent.load_state_dict(dict(np.load(os.path.join(run_dir, ckpt))))
    if hasattr(agent, "bind_env"):
        agent.bind_env(env)
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--ckpt", default="ckpt_best.npz")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=20_000)
    p.add_argument("--combine", choices=("mean_prob", "majority"), default="mean_prob")
    p.add_argument("--out", default=None)
    p.add_argument("--video", default=None,
                   help="write one representative episode as an MJPEG AVI")
    p.add_argument("--video-episode", type=int, default=0)
    p.add_argument("--video-scale", type=float, default=3.0)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--env-json", default=None,
                   help="JSON file with environment overrides shared by every member")
    args = p.parse_args()

    with open(os.path.join(args.runs[0], "config.json"), encoding="utf-8") as fh:
        base_cfg = json.load(fh)
    env = MiniDriveEnv(seed=0, **base_cfg["env_cfg"])
    if args.env_json:
        with open(args.env_json, encoding="utf-8") as fh:
            env_over = json.load(fh)
        env = MiniDriveEnv(seed=0, **{**base_cfg["env_cfg"], **env_over})
    agents = [load_member(run_dir, env, args.ckpt) for run_dir in args.runs]

    if args.video:
        from autodrive.render import render_frame
        from autodrive.video import write_mjpeg_avi
    frames = []
    rows = []
    for episode in range(args.episodes):
        seed = args.seed_base + episode
        obs, _ = env.reset(seed=seed)
        for agent in agents:
            reset_agent(agent, obs)
        done, ret, steps = False, 0.0, 0
        speeds, lats = [], []
        while not done:
            probabilities = [_masked_probabilities(agent, obs) for agent in agents]
            action = combine_probabilities(probabilities, args.combine)
            obs, reward, term, trunc, info = env.step(action)
            for agent in agents:
                advance_agent(agent, obs, term, trunc)
            ret += reward
            steps += 1
            speeds.append(info["speed"])
            lats.append(abs(info["lateral"]))
            if (args.video and episode == args.video_episode
                    and steps % max(1, int(args.frame_stride)) == 0):
                frames.append(render_frame(
                    env, scale=args.video_scale, show_lidar=True,
                    hud=(f"RARL v2 ensemble  ep {episode + 1}/{args.episodes}  "
                         f"step {steps:3d}  v={env.v:4.1f} m/s  "
                         f"gap={info.get('npc_min_distance', float('nan')):4.1f} m  "
                         f"{ACTION_NAMES[action]}"),
                ))
            done = term or trunc
        rows.append({
            "seed": seed,
            "return": ret,
            "steps": steps,
            "success": int(info["success"]),
            "collision": int(info.get("collision", False)),
            "speed": float(np.mean(speeds)) if speeds else 0.0,
            "lateral_rms": float(np.sqrt(np.mean(np.square(lats)))) if lats else 0.0,
            "lap_time": steps * env.dt,
        })

    successes = [r for r in rows if r["success"]]
    summary = {
        "method": args.combine,
        "members": len(agents),
        "episodes": args.episodes,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "speed_mean": float(np.mean([r["speed"] for r in rows])),
        "lateral_rms": float(np.mean([r["lateral_rms"] for r in rows])),
        "lap_time_mean": float(np.mean([r["lap_time"] for r in successes]))
        if successes else float("nan"),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))
    if args.video and frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
        stride = max(1, int(args.frame_stride))
        fps = float(args.fps) if args.fps is not None else 10.0 / stride
        write_mjpeg_avi(args.video, frames, fps=fps)
        print(f"[video] {len(frames)} frames @ {fps:g} fps -> {args.video}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
