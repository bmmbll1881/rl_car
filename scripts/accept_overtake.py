#!/usr/bin/env python
"""Strict acceptance gate for the overtake task.

The gate is two independent counts over the *same* 10 held-out episodes:

* **clean laps** >= 6/10 -- the episode completed a lap with no collision and no
  off-track departure;
* **strict safe overtakes** >= 3/10 -- an episode contains at least one
  manoeuvre certified by :class:`autodrive.acceptance.StrictOvertakeMonitor`:
  safe lane change into an adjacent lane, the blocking car actually passed, the
  destination lane held until the pass, no collision and no off-track anywhere
  in the episode.

Both numbers are printed and written to JSON; the process exits non-zero when
either gate fails.  Nothing is inferred: a run that fails is reported as failed.

Examples
--------
Gate 0 -- prove the environment is solvable by the *demonstration* expert:

    ./runtime/python/python.exe scripts/accept_overtake.py --expert \
        --env-json configs/env_overtake_stage1.json --episodes 10

Gate 1 -- a trained checkpoint:

    ./runtime/python/python.exe scripts/accept_overtake.py \
        --run runs/bcft_stage1_20k --ckpt ckpt_best.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np

from autodrive.acceptance import gate
from autodrive.algos import make_algorithm
from autodrive.baselines import MemorylessExpertDriver, PurePursuitDriver
from autodrive.env import MiniDriveEnv
from autodrive.logging_utils import evaluate_agent

STAGE1 = os.path.join(HERE, "configs", "env_overtake_stage1.json")


def _load_env(args) -> MiniDriveEnv:
    env_cfg = {}
    if args.run:
        cfg_path = os.path.join(args.run, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                env_cfg = dict(json.load(fh).get("env_cfg") or {})
    if args.env_json:
        with open(args.env_json, encoding="utf-8") as fh:
            env_cfg.update(json.load(fh))
    return MiniDriveEnv(seed=0, **env_cfg)


def _load_agent(args, env):
    if args.expert:
        if args.legacy_expert:
            return PurePursuitDriver(env), "expert:expert"
        driver = MemorylessExpertDriver(env, block_distance=float(args.expert_block_distance))
        return driver, f"expert:{driver.name}"
    if not args.run:
        raise SystemExit("either --expert or --run is required")
    cfg_path = os.path.join(args.run, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"{cfg_path} not found; cannot rebuild the algorithm")
    with open(cfg_path, encoding="utf-8") as fh:
        run_cfg = json.load(fh)
    algorithm = run_cfg.get("algo", "rarl")
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(args.run, args.ckpt)
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint {ckpt} not found")
    agent = make_algorithm(algorithm, env.obs_dim, env.n_actions,
                           cfg=run_cfg.get("algorithm_cfg") or {}, seed=run_cfg.get("seed", 0))
    if hasattr(agent, "bind_env"):
        agent.bind_env(env)
    with np.load(ckpt) as state:
        agent.load_state_dict(dict(state))
    return agent, f"{algorithm}:{ckpt}"


def _fmt_bool(flag: bool) -> str:
    return "yes" if flag else "no "


def main() -> int:
    p = argparse.ArgumentParser(description="Strict overtake acceptance gate")
    p.add_argument("--expert", action="store_true", help="gate the scripted expert (Gate 0)")
    p.add_argument("--legacy-expert", action="store_true",
                   help="use the stateful PurePursuitDriver instead of the memoryless expert")
    p.add_argument("--run", default=None, help="run directory holding config.json + checkpoint")
    p.add_argument("--ckpt", default="ckpt_best.npz")
    p.add_argument("--env-json", default=None,
                   help=f"environment overrides (default for --expert: {STAGE1})")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=40_000,
                   help="held-out seeds; must not overlap training/eval/test/demo seeds")
    p.add_argument("--min-clean-laps", type=int, default=6)
    p.add_argument("--min-strict-overtakes", type=int, default=3)
    p.add_argument("--block-distance", type=float, default=18.0,
                   help="verifier: a lead closer than this blocks the source lane")
    p.add_argument("--expert-block-distance", type=float, default=22.0,
                   help="blocked threshold used by the expert's own lane choice "
                        "(match the collector's lane_block_distance)")
    p.add_argument("--pass-margin", type=float, default=3.0)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    if args.env_json is None and args.expert:
        args.env_json = STAGE1

    env = _load_env(args)
    agent, label = _load_agent(args, env)

    res = evaluate_agent(env, agent, args.episodes, base_seed=args.seed_base,
                         block_distance=args.block_distance,
                         pass_margin=args.pass_margin)
    rows = res.pop("_rows")

    clean_laps = sum(1 for r in rows if r["clean_lap"])
    strict = sum(1 for r in rows if r["strict_safe_overtake"])
    verdict = gate(clean_laps, strict, len(rows),
                   args.min_clean_laps, args.min_strict_overtakes)

    print(f"[accept] {label} | env {env.describe().split('|')[0].strip()}")
    print(f"[accept] {len(rows)} episodes, seeds {args.seed_base}..{args.seed_base + len(rows) - 1}")
    print(f"{'seed':>6} {'steps':>6} {'succ':>5} {'crash':>6} {'coll':>5} "
          f"{'clean_lap':>10} {'strict':>7} {'dist':>7} {'v':>5}")
    for r in rows:
        print(f"{r['seed']:6d} {r['steps']:6d} {r['success']:5d} {r['crash']:6d} "
              f"{r['collision']:5d} {_fmt_bool(r['clean_lap']):>10} "
              f"{r['strict_events']:7d} {r['distance']:7.1f} {r['mean_speed']:5.2f}")
    print(f"[accept] clean laps {clean_laps}/{len(rows)} "
          f"(>= {args.min_clean_laps}: {_fmt_bool(verdict['clean_lap_pass'])})")
    print(f"[accept] strict safe overtakes {strict}/{len(rows)} "
          f"(>= {args.min_strict_overtakes}: {_fmt_bool(verdict['strict_safe_overtake_pass'])})")
    print(f"[accept] collisions {verdict['episodes'] - res['eval_clean_episodes']}/"
          f"{verdict['episodes']}, off-track {res['eval_crash_rate'] - res['eval_collision_rate']:.2f}")

    detail = []
    for r in rows:
        detail.extend(r.get("strict_events_detail") or [])
    if detail:
        print("[accept] strict manoeuvres:")
        for e in detail:
            print(f"  seed {e['seed']:6d} step {e['start_step']:4d} lane {e['src_lane']}->"
                  f"{e['dst_lane']} blocker {e['blocker_id']} passed@{e['passed_step']} "
                  f"hold {e['hold_steps']} {e['reason']} valid={e['valid']}")

    summary = {
        "agent": label,
        "env": {"n_obstacles": env.cfg["n_obstacles"],
                "npc_v_min": env.cfg["npc_v_min"], "npc_v_max": env.cfg["npc_v_max"],
                "v_max": env.cfg["v_max"], "npc_lead_gap": env.cfg["npc_lead_gap"],
                "n_episodes": len(rows)},
        "criteria": {
            "clean_lap": "lap completed, no collision, no off-track",
            "strict_safe_overtake": ("safe lane change out of a blocked lane, blocker "
                                     "passed, destination lane held, episode clean"),
            "block_distance": args.block_distance,
            "pass_margin": args.pass_margin,
        },
        "gate": verdict,
        "metrics": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in res.items()},
        "episodes": [
            {"seed": r["seed"], "steps": r["steps"], "success": r["success"],
             "crash": r["crash"], "collision": r["collision"],
             "clean_lap": bool(r["clean_lap"]), "distance": round(float(r["distance"]), 2),
             "mean_speed": round(float(r["mean_speed"]), 3),
             "strict_events": int(r["strict_events"]),
             "strict_safe_overtake": bool(r["strict_safe_overtake"]),
             "overtake_passes": int(r["overtake"]["overtake_passes"]),
             "lane_bonus_events": int(r["overtake"]["lane_bonus_events"])}
            for r in rows
        ],
        "strict_manoeuvres": detail,
    }
    out = args.json_out or os.path.join(
        args.run if args.run else os.path.join(HERE, "runs"),
        f"acceptance_{'expert' if args.expert else 'policy'}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[accept] {'ACCEPTED' if verdict['passed'] else 'REJECTED'} -> {out}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
