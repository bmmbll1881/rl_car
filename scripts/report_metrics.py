"""Compute the report numbers (sample efficiency, stability, test-set means).

    python scripts/report_metrics.py --runs runs --out figs

Produces ``figs/report_metrics.md`` (markdown tables ready for the report) and
``figs/report_metrics.json``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(path):
    if not os.path.exists(path):
        return []
    rows = []
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


def load(runs_dir):
    out = []
    for cfg_path in sorted(glob.glob(os.path.join(runs_dir, "*", "config.json"))):
        d = os.path.dirname(cfg_path)
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        sp = os.path.join(d, "summary.json")
        if not os.path.exists(sp):
            continue
        with open(sp, encoding="utf-8") as fh:
            summary = json.load(fh)
        out.append(dict(tag=cfg["tag"], cfg=cfg, summary=summary,
                        evals=read_csv(os.path.join(d, "evals.csv")),
                        episodes=read_csv(os.path.join(d, "episodes.csv"))))
    return out


def first_reach(evals, thr):
    for e in evals:
        if e.get("eval_success_rate", 0) >= thr:
            return e["global_step"]
    return None


def agg(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(HERE, "runs"))
    p.add_argument("--out", default=os.path.join(HERE, "figs"))
    args = p.parse_args()
    runs = load(args.runs)
    os.makedirs(args.out, exist_ok=True)

    algos = sorted({r["cfg"]["algo"] for r in runs}, key=lambda a: (a != "ppo", a))
    md = []
    rows_test = {}
    for a in algos:
        rs = [r for r in runs if r["cfg"]["algo"] == a]
        if not rs:
            continue
        best = [r["summary"].get("test_eval_best") or {} for r in rs]
        final = [r["summary"].get("test_eval_final") or {} for r in rs]
        vals = [r["summary"].get("validation_eval") or {} for r in rs]
        rows_test[a] = {
            "n": len(rs),
            "steps": rs[0]["cfg"]["steps"],
            "test_best": {k: agg([b.get(k) for b in best]) for k in
                          ("eval_return_mean", "eval_success_rate", "eval_collision_rate",
                           "eval_lap_time_mean", "eval_speed_mean", "eval_lateral_rms")},
            "test_final": {k: agg([b.get(k) for b in final]) for k in
                           ("eval_return_mean", "eval_success_rate", "eval_lap_time_mean", "eval_speed_mean")},
            "val_best_return": agg([v.get("eval_return_mean") for v in vals]),
            "steps_to_80": agg([first_reach(r["evals"], 0.8) for r in rs]),
            "steps_to_100": agg([first_reach(r["evals"], 1.0) for r in rs]),
            "stable_frac": agg([np.mean([e["eval_success_rate"] >= 0.99 for e in r["evals"] if
                                         e["global_step"] >= 0.5 * rs[0]["cfg"]["steps"]]) for r in rs]),
            "steps_per_s": agg([r["summary"].get("steps_per_s") for r in rs]),
            "wall_s": agg([r["summary"].get("wall_s") for r in rs]),
        }

    md.append("### 最终性能（独立测试集 seeds 20000–20009，10 episodes）\n")
    md.append("| 算法 | 种子数 | 回报 | 成功率 | 碰撞率 | 圈速 (s) | 平均车速 (m/s) | 横向 RMS (m) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for a, d in rows_test.items():
        b = d["test_best"]
        md.append(f"| {a.upper()}（best ckpt） | {d['n']} | "
                  f"{b['eval_return_mean'][0]:.1f} ± {b['eval_return_mean'][1]:.1f} | "
                  f"{b['eval_success_rate'][0]:.2f} ± {b['eval_success_rate'][1]:.2f} | "
                  f"{b['eval_collision_rate'][0]:.2f} | "
                  f"{b['eval_lap_time_mean'][0]:.1f} ± {b['eval_lap_time_mean'][1]:.1f} | "
                  f"{b['eval_speed_mean'][0]:.2f} ± {b['eval_speed_mean'][1]:.2f} | "
                  f"{b['eval_lateral_rms'][0]:.2f} |")
    for r in runs:
        e = (r["summary"].get("reference_test") or {}).get("expert")
        if e:
            md.append(f"| expert（脚本参考） | - | {e['eval_return_mean']:.1f} | {e['eval_success_rate']:.2f} | "
                      f"{e['eval_crash_rate']:.2f} | {e['eval_lap_time_mean']:.1f} | "
                      f"{e['eval_speed_mean']:.2f} | {e['eval_lateral_rms']:.2f} |")
            break

    md.append("\n### 训练结束时的权重（未做 checkpoint 选择）\n")
    md.append("| 算法 | 回报 | 成功率 | 圈速 (s) | 平均车速 (m/s) |")
    md.append("|---|---|---|---|---|")
    for a, d in rows_test.items():
        f = d["test_final"]
        md.append(f"| {a.upper()}（final） | {f['eval_return_mean'][0]:.1f} ± {f['eval_return_mean'][1]:.1f} | "
                  f"{f['eval_success_rate'][0]:.2f} ± {f['eval_success_rate'][1]:.2f} | "
                  f"{f['eval_lap_time_mean'][0]:.1f} | {f['eval_speed_mean'][0]:.2f} |")

    md.append("\n### 样本效率与稳定性（验证集，每 10k 步一次贪心评估）\n")
    md.append("| 算法 | 首次成功≥0.8 (步) | 首次成功=1.0 (步) | 后半程评估中成功率=1.0 的比例 | 吞吐 (steps/s) | 单次训练墙钟 (s) |")
    md.append("|---|---|---|---|---|---|")
    for a, d in rows_test.items():
        md.append(f"| {a.upper()} | {_fmt(d['steps_to_80'])} | {_fmt(d['steps_to_100'])} | "
                  f"{d['stable_frac'][0]:.2f} ± {d['stable_frac'][1]:.2f} | "
                  f"{d['steps_per_s'][0]:,.0f} | {d['wall_s'][0]:.0f} |")

    txt = "\n".join(md) + "\n"
    with open(os.path.join(args.out, "report_metrics.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(args.out, "report_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(rows_test, fh, indent=2, ensure_ascii=False)
    print(txt)


def _fmt(t):
    m, s = t
    if not np.isfinite(m):
        return "未达到"
    return f"{m:,.0f} ± {s:,.0f}" if s and np.isfinite(s) and s > 0 else f"{m:,.0f}"


if __name__ == "__main__":
    main()
