#!/usr/bin/env python
"""Reproduce the whole experiment matrix (2 algorithms x N seeds, identical budget).

    python scripts/run_batch.py --seeds 0 1 2 3 4 --steps 150000

Each run is an independent process, so the runs cannot leak state into each
other.  Results land in ``runs/<algo>_s<seed>/``; ``aggregate.py`` then builds the
comparison figures.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="+", default=["dqn", "ppo"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--out", default=os.path.join(HERE, "runs"))
    p.add_argument("--obstacles", type=int, default=None,
                   help="number of NPC vehicles (omit for the environment default, 0 = empty track)")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for algo in args.algos:
        for seed in args.seeds:
            tag = f"{algo}_s{seed}"
            run_dir = os.path.join(args.out, tag)
            if args.skip_existing and os.path.exists(os.path.join(run_dir, "summary.json")):
                print(f"[skip] {tag} already finished")
                continue
            cmd = [
                sys.executable, os.path.join(HERE, "train.py"),
                "--algo", algo, "--seed", str(seed), "--steps", str(args.steps),
                "--eval-every", str(args.eval_every), "--eval-episodes", "5",
                "--config", os.path.join(HERE, "configs", f"{algo}.json"),
                "--tag", tag, "--out", args.out, "--quiet",
            ]
            if args.obstacles is not None:
                cmd += ["--obstacles", str(args.obstacles)]
            print(f"[run] {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
    print("all runs finished")


if __name__ == "__main__":
    main()
