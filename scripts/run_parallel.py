#!/usr/bin/env python
"""Run a matrix of (algorithm, seed) training jobs concurrently.

One process per run keeps the runs independent (no shared RNG / buffers), and
running several at once is what makes a 3-algorithm comparison practical on a
laptop:

    python scripts/run_parallel.py --algos dqn ppo rarl --seeds 0 1 2 --steps 250000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["dqn", "ppo", "rarl"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    ap.add_argument("--config", default=None, help="use this config for every run")
    ap.add_argument("--tag-prefix", default="", help="prefix for the run tag (e.g. 'lo_')")
    ap.add_argument("--init-pattern", default=None,
                    help="per-seed checkpoint template, e.g. 'runs/rarl_s{seed}/ckpt_best.npz'")
    ap.add_argument("--bc-episodes", type=int, default=0,
                    help="pass scripted-expert behavior-cloning episodes to each run")
    ap.add_argument("--bc-epochs", type=int, default=12)
    ap.add_argument("--env-json", default=None,
                    help="environment override JSON passed to every training job")
    ap.add_argument("--jobs", type=int, default=0, help="max concurrent runs (0 = all at once)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    todo = [(a, s) for a in args.algos for s in args.seeds]
    if args.jobs and args.jobs > 0:
        todo_batches = [todo[i:i + args.jobs] for i in range(0, len(todo), args.jobs)]
    else:
        todo_batches = [todo]

    t0 = time.perf_counter()
    for batch in todo_batches:
        procs = []
        for algo, seed in batch:
            tag = f"{args.tag_prefix}{algo}_s{seed}"
            cfg = args.config or os.path.join(HERE, "configs", f"{algo}.json")
            cmd = [sys.executable, os.path.join(HERE, "train.py"),
                   "--algo", algo, "--seed", str(seed), "--steps", str(args.steps),
                   "--eval-every", str(args.eval_every), "--eval-episodes", "5",
                   "--tag", tag, "--out", args.out, "--quiet"]
            if os.path.exists(cfg):
                cmd += ["--config", cfg]
            if args.init_pattern:
                init_ckpt = args.init_pattern.format(seed=seed)
                if not os.path.exists(init_ckpt):
                    raise FileNotFoundError(init_ckpt)
                cmd += ["--init-checkpoint", init_ckpt]
            if args.bc_episodes > 0:
                cmd += ["--bc-episodes", str(args.bc_episodes),
                        "--bc-epochs", str(args.bc_epochs)]
            if args.env_json:
                cmd += ["--env-json", args.env_json]
            log = os.path.join(args.out, f"{tag}.log")
            fh = open(log, "w", encoding="utf-8")
            procs.append((tag, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
            print(f"[start] {tag}", flush=True)
        for tag, proc, fh in procs:
            rc = proc.wait()
            fh.close()
            print(f"[done ] {tag}  exit={rc}", flush=True)
    print(f"[all] {len(todo)} runs finished in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
