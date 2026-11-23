#!/usr/bin/env python
"""Aggregate several runs into comparison figures + a summary table.

    python aggregate.py --runs runs --out figs

Reads every ``runs/<tag>/`` that has a ``config.json``, groups runs by
algorithm, and produces (Pillow only, no matplotlib):

    figs/curve_train_return.png   training return vs. environment steps (mean +- std)
    figs/curve_eval_return.png    greedy evaluation return vs. steps
    figs/curve_eval_success.png   lap success rate vs. steps
    figs/bar_final_metrics.png    final metrics side by side
    figs/summary.csv / summary.md machine- and human-readable summary
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
from PIL import Image

from autodrive.render import PALETTE, bar_chart, line_chart

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS_NUM = {
    "return", "global_step", "episode", "steps", "success", "crash", "distance", "mean_speed",
    "lateral_rms", "epsilon", "loss", "eval_return_mean", "eval_success_rate", "eval_crash_rate",
    "eval_lap_time_mean", "eval_speed_mean", "eval_distance_mean", "eval_lateral_rms",
}


def _read_csv(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                if k in FIELDS_NUM and v not in ("", None):
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = np.nan
                else:
                    out[k] = v
            rows.append(out)
    return rows


def load_runs(root):
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(root, "*", "config.json"))):
        run_dir = os.path.dirname(cfg_path)
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        eps = _read_csv(os.path.join(run_dir, "episodes.csv"))
        evs = _read_csv(os.path.join(run_dir, "evals.csv"))
        if not eps and not evs:
            continue
        summary = {}
        sp = os.path.join(run_dir, "summary.json")
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as fh:
                summary = json.load(fh)
        runs.append(dict(dir=run_dir, tag=cfg.get("tag", os.path.basename(run_dir)), cfg=cfg,
                         episodes=eps, evals=evs, summary=summary))
    return runs


def smooth(y, w=10):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def curve_matrix(runs_list, source, key, grid, smooth_window=0):
    """Mean/std curve over seeds, re-sampled onto ``grid`` (env steps)."""
    rows = []
    for r in runs_list:
        raw = r.get(source) or []
        pts = [(e["global_step"], e[key]) for e in raw
               if isinstance(e.get("global_step"), float) and isinstance(e.get(key), float)]
        if len(pts) < 2:
            continue
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 2:
            continue
        if smooth_window:
            y = smooth(y, smooth_window)
            x = x[-len(y):]
        rows.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
    if not rows:
        return None, None
    m = np.vstack(rows)
    return np.nanmean(m, axis=0), np.nanstd(m, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(HERE, "runs"))
    p.add_argument("--out", default=os.path.join(HERE, "figs"))
    p.add_argument("--bin", type=int, default=2500)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no runs found under {args.runs}")

    def pick_final(r):
        s = r["summary"]
        for key in ("test_eval_best", "test_eval_final", "validation_eval", "final_eval"):
            if s.get(key):
                return s[key]
        return None

    def pick_ref(r):
        s = r["summary"]
        return s.get("reference_test") or s.get("reference") or {}

    algos = sorted({r["cfg"].get("algo", "?") for r in runs}, key=lambda a: (a != "ppo", a))
    print(f"loaded {len(runs)} runs: " + ", ".join(f"{r['tag']}({len(r['episodes'])} eps)" for r in runs))

    by_algo = {a: [r for r in runs if r["cfg"].get("algo") == a] for a in algos}
    max_step = max([e["global_step"] for r in runs for e in r["episodes"]] +
                   [e["global_step"] for r in runs for e in r["evals"]] or [0])
    grid = np.arange(0, max_step + args.bin, args.bin)

    # ---- reference lines (same eval protocol) ------------------------------
    ref_lines = []
    for r in runs:
        ref = pick_ref(r)
        if "expert" in ref:
            ref_lines.append({"y": ref["expert"]["eval_return_mean"],
                              "label": f"expert {ref['expert']['eval_return_mean']:.0f}",
                              "color": PALETTE["expert"]})
            break

    series = []
    for a in algos:
        mean, std = curve_matrix(by_algo[a], "episodes", "return", grid, smooth_window=10)
        if mean is None:
            continue
        series.append({"x": grid, "y": mean, "band": (mean - np.nan_to_num(std), mean + np.nan_to_num(std)),
                       "label": f"{a.upper()} (n={len(by_algo[a])})", "color": PALETTE.get(a, (80, 80, 80))})
    if series:
        path = line_chart(os.path.join(args.out, "curve_train_return.png"), series,
                          title="Training return per episode (rolling mean over 10 episodes, mean ± std over seeds)",
                          xlabel="environment steps", ylabel="episode return", hlines=ref_lines[:1])
        print("wrote", path)

    for key, fname, title, ylabel, ylim in [
        ("eval_return_mean", "curve_eval_return.png", "Greedy evaluation return (5 fixed seeds)", "eval return", None),
        ("eval_success_rate", "curve_eval_success.png", "Lap success rate on the evaluation seeds", "success rate", (0, 1.05)),
        ("eval_lap_time_mean", "curve_eval_lap_time.png", "Mean lap time of successful episodes", "lap time [s]", None),
    ]:
        series = []
        for a in algos:
            mean, std = curve_matrix(by_algo[a], "evals", key, grid)
            if mean is None or np.all(np.isnan(mean)):
                continue
            series.append({"x": grid, "y": mean,
                           "band": (mean - np.nan_to_num(std), mean + np.nan_to_num(std)),
                           "label": f"{a.upper()} (n={len(by_algo[a])})", "color": PALETTE.get(a, (80, 80, 80))})
        if not series:
            continue
        hl = []
        for r in runs:
            ref = pick_ref(r)
            e = ref.get("expert")
            if not e:
                continue
            if key == "eval_lap_time_mean":
                hl = [{"y": e["eval_lap_time_mean"], "label": f"expert {e['eval_lap_time_mean']:.0f}s",
                       "color": PALETTE["expert"]}]
            elif key == "eval_return_mean":
                hl = [{"y": e["eval_return_mean"], "label": f"expert {e['eval_return_mean']:.0f}",
                       "color": PALETTE["expert"]}]
            elif key == "eval_success_rate":
                hl = [{"y": 1.0, "label": "expert 1.00", "color": PALETTE["expert"]}]
            break
        path = line_chart(os.path.join(args.out, fname), series, title=title,
                          xlabel="environment steps", ylabel=ylabel, ylim=ylim, hlines=hl)
        print("wrote", path)

    # ---- final metric bars --------------------------------------------------
    finals = {}
    for a in algos:
        rows = [pick_final(r) for r in by_algo[a] if pick_final(r)]
        if not rows:
            continue
        finals[a] = {k: (float(np.mean([row[k] for row in rows])), float(np.std([row[k] for row in rows])))
                     for k in rows[0] if isinstance(rows[0][k], (int, float))}

    if finals:
        specs = [
            ("eval_return_mean", "Final return (10 held-out seeds)"),
            ("eval_success_rate", "Lap success rate"),
            ("eval_lap_time_mean", "Lap time [s]  (lower is better)"),
            ("eval_lateral_rms", "Lateral RMS [m]  (lower = tracks the racing line)"),
        ]
        panels = []
        for i, (key, title) in enumerate(specs):
            vals = [(a.upper(), finals[a][key][0], PALETTE.get(a, (80, 80, 80)))
                    for a in algos if a in finals and key in finals[a] and np.isfinite(finals[a][key][0])]
            if not vals:
                continue
            path = os.path.join(args.out, f"_panel{i}.png")
            bar_chart(path, [{"label": "", "values": vals, "ymax": max(v for _, v, _ in vals) * 1.15}],
                      title=title, width=720, height=360)
            panels.append(path)
        if panels:
            ims = [Image.open(p) for p in panels]
            cols = 2
            rows = (len(ims) + 1) // 2
            w = max(i.width for i in ims) * cols
            h = max(i.height for i in ims) * rows
            canvas = Image.new("RGB", (w, h), (255, 255, 255))
            for k, im in enumerate(ims):
                canvas.paste(im, ((k % cols) * im.width, (k // cols) * im.height))
            out_path = os.path.join(args.out, "bar_final_metrics.png")
            canvas.save(out_path)
            for p in panels:
                os.remove(p)
            print("wrote", out_path)

    # ---- summary tables -----------------------------------------------------
    csv_path = os.path.join(args.out, "summary.csv")
    md_path = os.path.join(args.out, "summary.md")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["algo", "seeds", "budget_steps", "final_return", "final_return_std", "success_rate",
                    "crash_rate", "collision_rate", "lap_time_s", "mean_speed", "lateral_rms",
                    "min_npc_distance", "steps_per_s", "wall_s"])
        lines = ["| algorithm | seeds | budget | final return | success | collision | lap time (s) | mean speed | lateral RMS | min NPC gap (m) | step/s |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in algos:
            rs = by_algo[a]
            if not rs:
                continue
            final = finals.get(a, {})
            sps = [r["summary"].get("steps_per_s") for r in rs if r["summary"].get("steps_per_s")]
            wall = [r["summary"].get("wall_s") for r in rs if r["summary"].get("wall_s")]
            row = [
                a.upper(),
                len(rs),
                int(rs[0]["cfg"].get("steps", 0)),
                f"{final['eval_return_mean'][0]:.1f}±{final['eval_return_mean'][1]:.1f}" if "eval_return_mean" in final else "-",
                f"{final['eval_success_rate'][0]:.2f}" if "eval_success_rate" in final else "-",
                f"{final['eval_collision_rate'][0]:.2f}" if "eval_collision_rate" in final else "-",
                f"{final['eval_lap_time_mean'][0]:.1f}" if "eval_lap_time_mean" in final else "-",
                f"{final['eval_speed_mean'][0]:.2f}" if "eval_speed_mean" in final else "-",
                f"{final['eval_lateral_rms'][0]:.2f}" if "eval_lateral_rms" in final else "-",
                f"{final['eval_min_npc_distance'][0]:.1f}" if "eval_min_npc_distance" in final else "-",
                f"{np.mean(sps):.0f}" if sps else "-",
            ]
            w.writerow(row)
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        for r in runs:
            e = (pick_ref(r)).get("expert")
            if e:
                lines.append(f"| expert (scripted) | - | - | {e['eval_return_mean']:.1f} | "
                             f"{e['eval_success_rate']:.2f} | {e.get('eval_collision_rate', 0):.2f} | "
                             f"{e['eval_lap_time_mean']:.1f} | {e['eval_speed_mean']:.2f} | "
                             f"{e['eval_lateral_rms']:.2f} | {e.get('eval_min_npc_distance', float('nan')):.1f} | - |")
                w.writerow(["expert", "-", "-", f"{e['eval_return_mean']:.1f}", e["eval_success_rate"],
                            e.get("eval_collision_rate", 0), e["eval_lap_time_mean"], e["eval_speed_mean"],
                            e["eval_lateral_rms"], e.get("eval_min_npc_distance", 0), "-"])
                break
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", csv_path, "and", md_path)


if __name__ == "__main__":
    main()
