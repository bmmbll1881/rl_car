#!/usr/bin/env python
"""Training curves for the eval metrics, one chart per metric.

Reads ``runs/<algo>_s<seed>/evals.csv`` (written every ``--eval-every`` steps),
interpolates every seed onto a common step grid, and draws the mean with a
mean±std band.  Pillow only, like the rest of the project's figures.

    python scripts/make_metric_curves.py --algos dqn ppo
    python scripts/make_metric_curves.py --algos dqn ppo rarl --metrics success speed
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np

from autodrive.render import line_chart

# metric name -> (csv column, chart title, y label, units)
METRICS = {
    "success": ("eval_success_rate", "Eval success rate", "success rate", ""),
    "speed": ("eval_speed_mean", "Eval mean speed", "mean speed", " m/s"),
    "lap": ("eval_lap_time_mean", "Eval lap time (successful episodes)", "lap time", " s"),
    "return": ("eval_return_mean", "Eval return (time cost + bonus)", "return", ""),
    "collision": ("eval_collision_rate", "Eval collision rate", "collision rate", ""),
}
COLORS = {"dqn": (205, 92, 60), "ppo": (40, 105, 175), "rarl": (60, 150, 90),
          "expert": (120, 120, 120)}


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = []
        for r in csv.DictReader(fh):
            try:
                x = float(r["global_step"])
                y = float(r[col])
            except (KeyError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                rows.append((x, y))
    return rows


def curves_for(algo, column, runs_dir):
    """Per-seed eval series of one algorithm: ``[(x, y), ...]`` per seed."""
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, f"{algo}_s*", "evals.csv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = []
            for r in csv.DictReader(fh):
                try:
                    x, y = float(r["global_step"]), float(r[column])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    rows.append((x, y))
        if rows:
            out.append(rows)
    return out


def mean_band(series, grid):
    """Interpolate every seed onto ``grid`` and return (mean, std)."""
    if not series:
        return np.zeros_like(grid), np.zeros_like(grid)
    a = np.vstack([interp_seed(rows, grid) for rows in series])
    return a.mean(axis=0), a.std(axis=0)


def interp_seed(rows, grid):
    """One seed's eval series resampled on ``grid``."""
    xs = np.array([r[0] for r in rows], float)
    ys = np.array([r[1] for r in rows], float)
    order = np.argsort(xs)
    return np.interp(grid, xs[order], ys[order])


BOUNDS = {"success": (0.0, 1.0), "collision": (0.0, 1.0)}
RATE_METRICS = ("success", "collision")


def eval_episodes(runs_dir, algo):
    """How many evaluation episodes one eval point averages over."""
    for path in sorted(glob.glob(os.path.join(runs_dir, f"{algo}_s*", "config.json"))):
        try:
            import json
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            return int(cfg.get("eval_seeds", {}).get("episodes", 5))
        except Exception:
            continue
    return 5


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def pooled_rate_band(algo, column, runs_dir, grid, n_episodes):
    """Pool every seed's episodes per eval point and return (rate, lo, hi) with Wilson CI.

    A success rate is ``successes / episodes``: the right uncertainty is a
    binomial confidence interval on the pooled counts, not a std over seeds.
    """
    buckets = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, f"{algo}_s*", "evals.csv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    x = float(r["global_step"])
                    rate = float(r[column])
                except (KeyError, ValueError):
                    continue
                if not (math.isfinite(x) and math.isfinite(rate)):
                    continue
                k = int(round(rate * n_episodes))
                key = int(round(x / 10_000.0)) * 10_000
                b = buckets.setdefault(key, [0, 0])
                b[0] += k
                b[1] += n_episodes
    rate, lo, hi = np.full_like(grid, np.nan), np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    for i, g in enumerate(grid):
        b = buckets.get(int(g))
        if b and b[1] > 0:
            p, l, h = wilson(b[0], b[1])
            rate[i], lo[i], hi[i] = p, l, h
    ok = np.isfinite(rate)
    if ok.sum() < 2:
        return None
    rate = np.interp(grid, grid[ok], rate[ok])
    lo = np.interp(grid, grid[ok], lo[ok])
    hi = np.interp(grid, grid[ok], hi[ok])
    return rate, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["dqn", "ppo"])
    ap.add_argument("--metrics", nargs="+", default=["success", "speed"], choices=sorted(METRICS))
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--step-limit", type=int, default=250_000, help="x range (default: the budget)")
    ap.add_argument("--band", choices=["auto", "wilson", "std", "minmax", "none"], default="auto",
                    help="auto = Wilson CI for rates (success/collision), std across seeds otherwise")
    ap.add_argument("--no-per-seed", action="store_true",
                    help="hide the thin per-seed lines")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grid = np.arange(10_000, args.step_limit + 1, 10_000, dtype=float)

    for metric in args.metrics:
        column, title, ylabel, unit = METRICS[metric]
        bounds = BOUNDS.get(metric)
        mode = args.band
        if mode == "auto":
            mode = "wilson" if metric in RATE_METRICS else "std"

        series = []
        for algo in args.algos:
            per_seed = curves_for(algo, column, args.runs)
            if not per_seed:
                print(f"[warn] no evals.csv for {algo}, skipped")
                continue
            base = COLORS.get(algo, (90, 90, 90))
            if not args.no_per_seed:
                light = tuple(int(c + (255 - c) * 0.55) for c in base)
                for rows in per_seed:
                    xs = np.array([r[0] for r in rows], float)
                    ys = np.array([r[1] for r in rows], float)
                    order = np.argsort(xs)
                    series.append({"x": xs[order], "y": ys[order], "color": light,
                                   "bold": False, "band": None})

            band = None
            if mode == "wilson":
                n_ep = eval_episodes(args.runs, algo)
                pooled = pooled_rate_band(algo, column, args.runs, grid, n_ep)
                if pooled is not None:
                    mean, lo, hi = pooled
                    band = (lo, hi)
                    note = f"pooled k/{n_ep * len(per_seed)} episodes, Wilson 95% CI"
                else:
                    mean, std = mean_band(per_seed, grid)
                    band, note = (mean - std, mean + std), "mean ± std (no counts found)"
            else:
                mean, std = mean_band(per_seed, grid)
                note = "mean ± std over seeds"
                if mode == "std":
                    band = (mean - std, mean + std)
                elif mode == "minmax":
                    stack = np.vstack([interp_seed(rows, grid) for rows in per_seed])
                    band = (stack.min(axis=0), stack.max(axis=0))
                    note = "mean, band = min..max over seeds"
            if bounds and band is not None:
                band = (np.clip(band[0], *bounds), np.clip(band[1], *bounds))
                mean = np.clip(mean, *bounds)
            series.append({
                "x": grid, "y": mean,
                "label": f"{algo.upper()} ({len(per_seed)} seeds)",
                "color": base, "band": band, "bold": True,
            })
            print(f"  {algo.upper():5} {metric:8} [{mode:6}]: "
                  + "  ".join(f"{int(g/1000)}k={m:.3f}" for g, m in zip(grid[::5], mean[::5])))

        if not series:
            continue
        path = os.path.join(args.out, f"curve_train_{metric}_{'_'.join(args.algos)}.png")
        line_chart(
            path, series,
            title=f"{title} — {note if args.algos else ''}",
            xlabel="environment steps", ylabel=ylabel + unit,
            ylim=bounds, xlim=(0.0, float(args.step_limit)),
            pad_frac=0.0 if bounds else 0.06,
        )
        print(f"[fig] {path}")

    # small text summary of the last grid point
    print("\nfinal eval (mean over seeds, at the end of the grid):")
    for algo in args.algos:
        per_seed = curves_for(algo, METRICS[args.metrics[0]][0], args.runs)
        if per_seed:
            row = []
            for metric in args.metrics:
                col = METRICS[metric][0]
                m, s = mean_band(curves_for(algo, col, args.runs), grid)
                row.append(f"{metric} {m[-1]:.3f}±{s[-1]:.3f}")
            print(f"  {algo.upper():5} " + "  ".join(row))


if __name__ == "__main__":
    main()
