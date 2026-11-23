"""Debug-oriented figures: per-step traces, action usage, and the PPO instability story.

    python scripts/make_debug_figs.py --runs runs --out figs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

from autodrive.algos import make_algorithm
from autodrive.baselines import PurePursuitDriver
from autodrive.env import MiniDriveEnv, ACTION_NAMES
from autodrive.logging_utils import advance_agent, reset_agent
from autodrive.render import PALETTE, bar_chart, line_chart, _font

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_agent(run_dir, env):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    ckpt = os.path.join(run_dir, "ckpt_best.npz")
    agent = make_algorithm(cfg["algo"], env.obs_dim, env.n_actions, cfg=cfg.get("algorithm_cfg"),
                           seed=cfg["seed"])
    agent.load_state_dict(dict(np.load(ckpt)))
    return agent, cfg


def rollout_trace(env, agent, seed):
    obs, _ = env.reset(seed=seed)
    reset_agent(agent, obs)
    rows, done = [], False
    while not done:
        a = agent.act(obs, greedy=True)
        obs, r, term, trunc, info = env.step(a)
        advance_agent(agent, obs, term, trunc)
        rows.append(dict(step=len(rows) + 1, action=a, speed=info["speed"], lateral=info["lateral"],
                         progress=info["progress"], reward=r,
                         gap=info.get("npc_min_distance", float("nan")),
                         collision=int(info.get("collision", False))))
        done = term or trunc
    return rows


def stack(images, gap=10):
    w = max(i.width for i in images)
    h = sum(i.height for i in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas


def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(HERE, "runs"))
    p.add_argument("--out", default=os.path.join(HERE, "figs"))
    p.add_argument("--seed", type=int, default=10_000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = MiniDriveEnv(seed=0)
    traces, actions = {}, {}
    for algo in ("dqn", "ppo"):
        run_dir = os.path.join(args.runs, f"{algo}_s0")
        if not os.path.exists(os.path.join(run_dir, "ckpt_best.npz")):
            print(f"[skip] {run_dir} has no checkpoint")
            continue
        agent, _ = load_agent(run_dir, env)
        rows = rollout_trace(env, agent, args.seed)
        traces[algo] = rows
        actions[algo] = rows

    expert = PurePursuitDriver(env)
    traces["expert"] = rollout_trace(env, expert, args.seed)
    actions["expert"] = traces["expert"]

    # trace CSVs (evidence a reader can re-inspect)
    for algo, rows in traces.items():
        path = os.path.join(args.out, f"trace_{algo}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", path)

    def series(key, label_suffix=""):
        out = []
        for algo in ("expert", "ppo", "dqn"):
            if algo not in traces:
                continue
            rows = traces[algo]
            out.append({"x": [r["step"] for r in rows], "y": [r[key] for r in rows],
                        "label": f"{algo.upper()}{label_suffix}", "color": PALETTE[algo]})
        return out

    line_chart(os.path.join(args.out, "trace_speed.png"), series("speed"),
               title=f"Speed profile on evaluation seed {args.seed} (greedy policy)",
               xlabel="control step (0.1 s)", ylabel="speed [m/s]")
    line_chart(os.path.join(args.out, "trace_lateral.png"), series("lateral"),
               title=f"Lateral offset from the centre line on seed {args.seed} (band = +/-4.5 m road edge)",
               xlabel="control step (0.1 s)", ylabel="lateral offset [m]",
               hlines=[{"y": 4.5, "label": "road edge", "color": (200, 60, 60)},
                       {"y": -4.5, "label": "", "color": (200, 60, 60)},
                       {"y": 0.0, "label": "centre line", "color": (170, 170, 170)}])
    line_chart(os.path.join(args.out, "trace_gap.png"), series("gap"),
               title=f"Distance to the closest other vehicle on seed {args.seed} (car-following behaviour)",
               xlabel="control step (0.1 s)", ylabel="gap [m]",
               hlines=[{"y": 2.0, "label": "collision", "color": (200, 60, 60)}])
    print("wrote", os.path.join(args.out, "trace_speed.png"), ", trace_lateral.png and trace_gap.png")

    # action usage
    groups = []
    for i, name in enumerate(ACTION_NAMES):
        vals = []
        for algo in ("dqn", "ppo", "expert"):
            if algo not in actions:
                continue
            rows = actions[algo]
            vals.append((algo.upper(), 100.0 * np.mean([r["action"] == i for r in rows]), PALETTE[algo]))
        groups.append({"label": name, "values": vals, "ymax": 100.0})
    bar_chart(os.path.join(args.out, "action_usage.png"), groups,
              title=f"Action usage on seed {args.seed} (% of control steps)", ylabel="% of steps")
    print("wrote", os.path.join(args.out, "action_usage.png"))

    # PPO instability: entropy / approx_kl from episodes.csv + eval success
    ppo_dir = os.path.join(args.runs, "ppo_s0")
    eps = read_csv(os.path.join(ppo_dir, "episodes.csv"))
    evs = read_csv(os.path.join(ppo_dir, "evals.csv"))
    if eps and evs:
        ent_rows = [(e["global_step"], e["entropy"]) for e in eps
                    if isinstance(e.get("global_step"), float) and isinstance(e.get("entropy"), float)]
        ent = [{"x": [p[0] for p in ent_rows], "y": [p[1] for p in ent_rows],
                "label": "policy entropy", "color": PALETTE["ppo"]}]
        path1 = os.path.join(args.out, "_entropy.png")
        line_chart(path1, ent, title="PPO (seed 0): policy entropy collapses over training",
                   xlabel="environment steps", ylabel="entropy [nats]", legend=False)
        s2 = [{"x": [e["global_step"] for e in evs], "y": [e["eval_success_rate"] for e in evs],
               "label": "validation success", "color": (46, 160, 67)}]
        path2 = os.path.join(args.out, "_evalsuccess.png")
        line_chart(path2, s2, title="PPO (seed 0): validation success rate oscillates with it",
                   xlabel="environment steps", ylabel="success rate", ylim=(0, 1.05), legend=False)
        stack([Image.open(path1), Image.open(path2)]).save(os.path.join(args.out, "ppo_instability.png"))
        os.remove(path1)
        os.remove(path2)
        print("wrote", os.path.join(args.out, "ppo_instability.png"))


if __name__ == "__main__":
    main()
