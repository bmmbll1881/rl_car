#!/usr/bin/env python
"""Evaluate every saved checkpoint on a dedicated 100-episode probe and plot the curves.

Unlike the 5-episode eval that runs *during* training, this script is the
"measurement" pass: for every ``ckpt_step*.npz`` of every seed it runs
``--episodes`` fresh episodes on fixed probe seeds and records

* the **success rate** = successful episodes / episodes, and
* the **mean speed of the successful episodes only** (a failed episode has no
  lap to be fast at, so it is excluded from the speed average).

Two charts are written, one per metric, with the per-seed spread and a Wilson
interval for the rate.  Checkpoints are evaluated in parallel processes.

    python scripts/eval_curve.py --algos dqn ppo --episodes 100
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np

COLORS = {"dqn": (205, 92, 60), "ppo": (40, 105, 175), "rarl": (60, 150, 90)}
EXPERT_SPEED = None          # filled in by main() if the expert probe was run
EXPERT_SUCCESS = None


def _wilson(k, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate_checkpoint(job):
    """Run ``job['episodes']`` episodes with one checkpoint; return a row."""
    from autodrive.env import MiniDriveEnv
    from autodrive.algos import make_algorithm
    from autodrive.logging_utils import advance_agent, reset_agent

    run_dir = job["run"]
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    env = MiniDriveEnv(seed=0, **cfg["env_cfg"])
    agent = make_algorithm(cfg["algo"], env.obs_dim, env.n_actions,
                           cfg=cfg["algorithm_cfg"], seed=cfg.get("seed", 0))
    agent.load_state_dict(dict(np.load(job["ckpt"])))
    if hasattr(agent, "bind_env"):
        agent.bind_env(env)

    successes = 0
    speeds = []
    laps = []
    collisions = 0
    for i in range(job["episodes"]):
        obs, _ = env.reset(seed=job["seed_base"] + i)
        reset_agent(agent, obs)
        done = False
        v_sum = 0.0
        n = 0
        while not done:
            action = agent.act(obs, greedy=True)
            obs, r, term, trunc, info = env.step(action)
            advance_agent(agent, obs, term, trunc)
            v_sum += env.v
            n += 1
            done = term or trunc
        if info.get("success"):
            successes += 1
            speeds.append(v_sum / max(n, 1))
            laps.append(env.steps * env.dt)
        if info.get("collision"):
            collisions += 1

    return {
        "run": os.path.basename(run_dir),
        "algo": cfg["algo"],
        "seed": cfg.get("seed", 0),
        "step": job["step"],
        "episodes": job["episodes"],
        "successes": successes,
        "collisions": collisions,
        "speed_success": float(np.mean(speeds)) if speeds else float("nan"),
        "speed_success_std": float(np.std(speeds)) if len(speeds) > 1 else float("nan"),
        "lap_success": float(np.mean(laps)) if laps else float("nan"),
    }


def collect_jobs(runs_dir, algos, step_limit):
    jobs = []
    for algo in algos:
        for run in sorted(glob.glob(os.path.join(runs_dir, f"{algo}_s*"))):
            for ckpt in sorted(glob.glob(os.path.join(run, "ckpt_step*.npz")),
                               key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()))):
                step = int("".join(ch for ch in os.path.basename(ckpt) if ch.isdigit()))
                if step > step_limit:
                    continue
                jobs.append({"run": run, "ckpt": ckpt, "step": step})
    return jobs


def group_by_step(rows, bucket=10_000):
    """``{algo: {step: [rows]}}`` — checkpoint steps snapped to the eval grid.

    Seeds finish their 10k checkpoints at slightly different steps (e.g. 10038 vs
    10134), so rows are bucketed onto the nominal grid before pooling.
    """
    out = {}
    for r in rows:
        key = int(round(float(r["step"]) / bucket)) * bucket
        out.setdefault(r["algo"], {}).setdefault(key, []).append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["dqn", "ppo"])
    ap.add_argument("--episodes", type=int, default=100, help="episodes per checkpoint (the probe)")
    ap.add_argument("--seed-base", type=int, default=10_000, help="probe seeds are seed_base..+episodes-1")
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--step-limit", type=int, default=250_000)
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count()//2")
    ap.add_argument("--reuse", action="store_true", help="reuse an existing eval_curve.csv")
    ap.add_argument("--no-band", action="store_true", help="plain lines: no shaded band")
    ap.add_argument("--no-per-seed", action="store_true", help="hide the thin per-seed lines")
    ap.add_argument("--suffix", default="", help="suffix for the output file names")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "eval_curve.csv")
    jobs = collect_jobs(args.runs, args.algos, args.step_limit)
    print(f"[probe] {len(jobs)} checkpoints x {args.episodes} episodes "
          f"({len(args.algos)} algos)")

    rows = []
    if args.reuse and os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = [{k: (int(v) if k in ("seed", "step", "episodes", "successes", "collisions") else
                         (v if k in ("run", "algo") else float(v)))
                     for k, v in r.items()} for r in csv.DictReader(fh)]
        print(f"[probe] reused {len(rows)} rows from {csv_path}")
    else:
        for j in jobs:
            j["episodes"] = args.episodes
            j["seed_base"] = args.seed_base

        workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(evaluate_checkpoint, j) for j in jobs]
            for fut in as_completed(futures):
                rows.append(fut.result())
                done += 1
                if done % 10 == 0 or done == len(futures):
                    print(f"  {done}/{len(futures)} checkpoints done", flush=True)

        fields = ["run", "algo", "seed", "step", "episodes", "successes", "collisions",
                  "speed_success", "speed_success_std", "lap_success"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in sorted(rows, key=lambda d: (d["algo"], d["seed"], d["step"])):
                w.writerow({k: r.get(k) for k in fields})
        print(f"[probe] wrote {csv_path}")

    # ---- expert reference on the same probe seeds ------------------------
    try:
        from autodrive.env import MiniDriveEnv
        from autodrive.baselines import PurePursuitDriver
        env = MiniDriveEnv(seed=0)
        drv = PurePursuitDriver(env)
        ok = 0
        spd = []
        for i in range(args.episodes):
            obs, _ = env.reset(seed=args.seed_base + i)
            done = False
            v_sum, n = 0.0, 0
            while not done:
                obs, r, term, trunc, info = env.step(drv.act())
                v_sum += env.v
                n += 1
                done = term or trunc
            if info.get("success"):
                ok += 1
                spd.append(v_sum / max(n, 1))
        global EXPERT_SPEED, EXPERT_SUCCESS
        EXPERT_SUCCESS = ok / args.episodes
        EXPERT_SPEED = float(np.mean(spd)) if spd else float("nan")
        print(f"[expert] {ok}/{args.episodes} success, mean speed {EXPERT_SPEED:.2f} m/s")
    except Exception as exc:                      # expert is only a reference line
        print(f"[expert] skipped ({exc})")

    grouped = group_by_step(rows)

    from autodrive.render import line_chart

    def per_seed(algo, steps_map, steps, key):
        """``{seed: [(step, value), ...]}`` for one metric."""
        out = {}
        for st in steps:
            for b in steps_map[st]:
                out.setdefault(b["seed"], []).append((st, b[key]))
        return out

    for metric in ("success", "speed"):
        series = []
        for algo in args.algos:
            steps_map = grouped.get(algo)
            if not steps_map:
                continue
            steps = sorted(steps_map)
            base = COLORS.get(algo, (90, 90, 90))
            light = tuple(int(c + (255 - c) * 0.5) for c in base)
            x = np.array(steps, float)

            if metric == "success":
                k = np.array([sum(b["successes"] for b in steps_map[st]) for st in steps], float)
                n = np.array([sum(b["episodes"] for b in steps_map[st]) for st in steps], float)
                y = k / np.maximum(n, 1.0)
                ci = np.array([_wilson(a, b) for a, b in zip(k, n)])
                band = (ci[:, 0], ci[:, 1])
                ylabel = "success rate (pooled probe episodes)"
                title = f"Success rate per checkpoint — {int(n[0])} probe episodes pooled over seeds"
                ylim = (0.0, 1.0)
                hlines = ([{"y": EXPERT_SUCCESS, "label": f"expert {EXPERT_SUCCESS:.2f}",
                            "color": (130, 130, 130)}] if EXPERT_SUCCESS is not None else [])
                thin_key = None                     # handled below from counts
            else:
                vals, slow, shigh = [], [], []
                for st in steps:
                    good = [b for b in steps_map[st] if math.isfinite(b["speed_success"])]
                    if good:
                        w = np.array([b["successes"] for b in good], float)
                        v = np.array([b["speed_success"] for b in good], float)
                        vals.append(float((w * v).sum() / w.sum()))
                        slow.append(float(v.min()))
                        shigh.append(float(v.max()))
                    else:
                        vals.append(float("nan"))
                        slow.append(float("nan"))
                        shigh.append(float("nan"))
                y = np.array(vals)
                band = (np.array(slow), np.array(shigh))
                ylabel = "mean speed of successful laps (m/s)"
                title = "Speed when succeeding — failed episodes excluded, band = min..max across seeds"
                ylim = None
                hlines = ([{"y": EXPERT_SPEED, "label": f"expert {EXPERT_SPEED:.2f}", "color": (130, 130, 130)}]
                          if EXPERT_SPEED is not None and math.isfinite(EXPERT_SPEED) else [])

            # thin lines: one per seed
            if not args.no_per_seed:
                if metric == "success":
                    for s, pts in per_seed(algo, steps_map, steps, "successes").items():
                        xs = np.array([p[0] for p in pts], float)
                        ks = np.array([p[1] for p in pts], float)
                        ns = np.array([next(b["episodes"] for b in steps_map[int(p[0])] if b["seed"] == s)
                                       for p in pts], float)
                        series.append({"x": xs, "y": ks / ns, "color": light, "bold": False, "band": None})
                else:
                    for s, pts in per_seed(algo, steps_map, steps, "speed_success").items():
                        series.append({"x": np.array([p[0] for p in pts], float),
                                       "y": np.array([p[1] for p in pts], float),
                                       "color": light, "bold": False, "band": None})

            if args.no_band:
                band = None
                title = title.split(" — ")[0]
            series.append({"x": x, "y": y,
                           "label": f"{algo.upper()} ({len({b['seed'] for b in steps_map[steps[0]]})} seeds)",
                           "color": base, "band": band, "bold": True})

        if not series:
            continue
        suffix = args.suffix or ("_plain" if args.no_band else "")
        path = os.path.join(args.out, f"curve_probe_{metric}_{'_'.join(args.algos)}{suffix}.png")
        line_chart(path, series, title=title, xlabel="environment steps", ylabel=ylabel,
                   ylim=ylim, xlim=(0.0, float(args.step_limit)), hlines=hlines,
                   pad_frac=0.0 if ylim else 0.06)
        print(f"[fig] {path}")


if __name__ == "__main__":
    main()
