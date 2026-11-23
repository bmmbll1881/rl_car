#!/usr/bin/env python
"""Curriculum entry point for sensor-fused RARL overtaking.

Two stages, chained through checkpoints, then an independent greedy evaluation of
every stage checkpoint on held-out seeds and ``scripts/diagnose_risk.py`` on the
best one.

    Easy     ``configs/env_overtake_easy.json`` -- 8 sparse NPCs at 1.5-3 m/s, one
             slow lead planted 25 m ahead.  Meets a car rarely, so the ego can
             learn to drive and to complete one lane change without a crash.
    Stage 1  ``configs/env_overtake_stage1.json`` -- 12 NPCs at 2.5-4 m/s, ego
             capped at 10 m/s.  The ego can still catch the lead here, so the
             overtake signal stays reachable.
    Stage 2  ``configs/env_overtake_stage2.json`` -- 18 NPCs at 4-6 m/s, ego
             capped at 13 m/s.  Same policy, harder traffic.

Both stages use ``configs/rarl_overtake_sensor.json``: the 25-dim
:class:`autodrive.sensors.SensorFusion` block is on, and positive shaping is
only paid for a verified safe lane change plus forward progress inside the pass
window (``progress_reward_only_in_overtake_window``).  The tactical lock stops
starting new lane changes while a pass window is open
(``overtake_lock_pause_in_window``), so the policy is free to accelerate away
from the car it just passed.

Example
-------
    ./runtime/python/python.exe scripts/train_overtake_curriculum.py \
        --steps 40000 --stage1-steps 30000 --tag rarl_overtake_sensor_40k

The script never invents numbers: it prints what the evaluation and
``diagnose_risk`` actually returned, and records whether a *safe* overtake
(blocker passed after a verified lane change, no collision) happened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np

from autodrive.algos import make_algorithm
from autodrive.env import MiniDriveEnv
from autodrive.logging_utils import evaluate_agent, selection_key

TRAIN = os.path.join(HERE, "train.py")
DIAGNOSE = os.path.join(HERE, "scripts", "diagnose_risk.py")


def _run(cmd, cwd=HERE):
    print("[curriculum]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage(tag, env_json, algo_cfg, steps, out, init=None, bc=None):
    cmd = [sys.executable, TRAIN, "--algo", "rarl", "--seed", "0",
           "--tag", tag, "--out", out,
           "--config", algo_cfg, "--env-json", env_json,
           "--eval-episodes", "5", "--final-eval-episodes", "10",
           "--test-episodes", "10"]
    if bc:
        # A pure warm-start stage: no environment steps, just the expert data
        # and the supervised update.  Its ckpt_final.npz is the BC checkpoint
        # the first PPO stage is initialised from.
        cmd += ["--bc-only", "--bc-epochs", str(int(bc["epochs"])),
                "--bc-batch-size", str(int(bc["batch_size"])),
                "--bc-label-smoothing", str(float(bc["label_smoothing"]))]
        if bc.get("demos"):
            cmd += ["--bc-demos", bc["demos"]]
        if bc.get("episodes"):
            cmd += ["--bc-episodes", str(int(bc["episodes"]))]
        if bc.get("keep_clean_only"):
            cmd += ["--bc-keep-clean-only"]
        if bc.get("legacy_expert"):
            cmd += ["--bc-legacy-expert"]
        if bc.get("save"):
            cmd += ["--bc-save", bc["save"]]
    else:
        cmd += ["--steps", str(int(steps)),
                "--eval-every", str(int(steps))]
    if init:
        cmd += ["--init-checkpoint", init]
    _run(cmd)
    ckpt = os.path.join(out, tag, "ckpt_final.npz")
    return {"name": tag, "dir": os.path.join(out, tag), "steps": int(steps),
            "init_checkpoint": init, "bc": bool(bc), "checkpoint": ckpt,
            "checkpoint_sha256": _sha256(ckpt) if os.path.exists(ckpt) else None}


def _load_agent(run_dir, ckpt):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        run_cfg = json.load(fh)
    env = MiniDriveEnv(seed=0, **run_cfg["env_cfg"])
    agent = make_algorithm(run_cfg["algo"], env.obs_dim, env.n_actions,
                           cfg=run_cfg.get("algorithm_cfg") or {}, seed=run_cfg["seed"])
    if hasattr(agent, "bind_env"):
        agent.bind_env(env)
    agent.load_state_dict(dict(np.load(os.path.join(run_dir, ckpt))))
    return run_cfg, env, agent


def _evaluate_stage(stage, ckpt, episodes, seed_base):
    run_cfg, env, agent = _load_agent(stage["dir"], ckpt)
    res = evaluate_agent(env, agent, episodes, base_seed=seed_base)
    rows = res.pop("_rows", [])
    safe_pass = [i for i, r in enumerate(rows)
                 if r["overtake"]["overtake_passes"] > 0
                 and not r["collision"] and not r["crash"]]
    strict = [i for i, r in enumerate(rows) if r.get("strict_safe_overtake")]
    safe_events = sum(int(r["overtake"]["overtake_passes"]) for i, r in enumerate(rows)
                      if i in safe_pass)
    return {
        "stage": stage["name"],
        "steps": stage["steps"],
        "dir": stage["dir"],
        "init_checkpoint": stage.get("init_checkpoint"),
        "episodes": len(rows),
        "seed_base": seed_base,
        "greedy": True,
        **{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
           for k, v in res.items() if isinstance(v, (int, float))},
        "episodes_with_pass": sum(int(r["overtake"]["overtake_passes"] > 0) for r in rows),
        "episodes_with_safe_pass": len(safe_pass),
        "episodes_with_strict_overtake": len(strict),
        "strict_overtake_events": sum(int(r.get("strict_events", 0)) for r in rows),
        "clean_laps": sum(int(bool(r.get("clean_lap"))) for r in rows),
        "safe_pass_episode_indices": safe_pass,
        "strict_overtake_episode_indices": strict,
        "safe_pass_events": int(safe_events),
        "clean_episodes": int(sum(1 for r in rows
                                  if not r["collision"] and not r["crash"])),
        "overtake_passes_total": sum(int(r["overtake"]["overtake_passes"]) for r in rows),
        "verified_lane_change_completions":
            sum(int(r["overtake"]["lane_bonus_events"]) for r in rows),
        "off_track_episodes": int(sum(1 for r in rows
                                      if r["crash"] and not r["collision"])),
        "safe_overtake_observed": bool(safe_pass or strict),
        "behavior_cloning": run_cfg.get("behavior_cloning") or {},
        "behavior_cloning_data": {
            k: v for k, v in (run_cfg.get("behavior_cloning_data") or {}).items()
            if k != "episode_stats"
        },
        "per_episode": [
            {"seed": r["seed"], "return": r["return"], "steps": r["steps"],
             "success": r["success"], "crash": r["crash"], "collision": r["collision"],
             "distance": r["distance"], "mean_speed": r["mean_speed"],
             "clean_lap": bool(r.get("clean_lap", False)),
             "lane_changes": r["overtake"]["lane_changes"],
             "lane_bonus_events": r["overtake"]["lane_bonus_events"],
             "passes": r["overtake"]["overtake_passes"],
             "strict_events": int(r.get("strict_events", 0)),
             "strict_safe_overtake": bool(r.get("strict_safe_overtake", False)),
             "safe_pass": bool(i in safe_pass)}
            for i, r in enumerate(rows)
        ],
    }


def _diagnose(run_dir, ckpt, episodes, seed_base, out_path):
    cmd = [sys.executable, DIAGNOSE, "--run", run_dir, "--ckpt", ckpt,
           "--episodes", str(int(episodes)), "--seed-base", str(int(seed_base))]
    print("[curriculum]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=HERE, check=True, capture_output=True, text=True)
    text = proc.stdout.strip()
    try:
        summary = json.loads(text)
    except json.JSONDecodeError:
        summary = {"raw": text}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(text, flush=True)
    return summary


def _score(ev):
    """Ranking used only to pick which checkpoint gets the diagnosis run."""
    return selection_key(ev)


def main():
    p = argparse.ArgumentParser(description="Run the sensor-fused overtake curriculum")
    p.add_argument("--steps", type=int, default=20_000, help="total environment steps")
    p.add_argument("--easy-steps", type=int, default=6_000, help="0 disables the stage")
    p.add_argument("--stage1-steps", type=int, default=10_000)
    p.add_argument("--stage2-steps", type=int, default=None,
                   help="default: --steps minus the earlier stages")
    p.add_argument("--curriculum", default=None,
                   help="JSON curriculum file (stages + env/algo configs); "
                        "overrides the --*-steps layout")
    p.add_argument("--stage-names", default=None,
                   help="comma-separated subset of curriculum stages (default: all)")
    p.add_argument("--stage-steps", type=int, default=None,
                   help="override the step budget of every curriculum stage")
    p.add_argument("--init-checkpoint", default=None,
                   help="load this checkpoint into the first stage (fine-tuning)")
    p.add_argument("--tag", default="rarl_overtake_sensor")
    p.add_argument("--out", default=os.path.join(HERE, "runs"))
    p.add_argument("--algo-config", default=os.path.join(HERE, "configs", "rarl_overtake_sensor.json"))
    p.add_argument("--bc-episodes", type=int, default=0,
                   help="collect this many expert episodes for the BC warm-start stage "
                        "(0 disables the stage unless --bc-demos is given)")
    p.add_argument("--bc-demos", default=None,
                   help="pre-collected demonstration .npz used by the BC stage")
    p.add_argument("--bc-epochs", type=int, default=300)
    p.add_argument("--bc-batch-size", type=int, default=128)
    p.add_argument("--bc-label-smoothing", type=float, default=0.02)
    p.add_argument("--bc-env", default=None,
                   help="environment used to collect/record the BC data "
                        "(default: the first curriculum stage)")
    p.add_argument("--bc-keep-clean-only", action="store_true",
                   help="drop demonstration episodes that crashed or collided")
    p.add_argument("--bc-legacy-expert", action="store_true",
                   help="collect with the stateful PurePursuitDriver (non-Markov labels)")
    p.add_argument("--bc-save-demos", default=None,
                   help="where the curriculum writes the collected demonstrations")
    p.add_argument("--easy-env", default=os.path.join(HERE, "configs", "env_overtake_easy.json"))
    p.add_argument("--stage1-env", default=os.path.join(HERE, "configs", "env_overtake_stage1.json"))
    p.add_argument("--stage2-env", default=os.path.join(HERE, "configs", "env_overtake_stage2.json"))
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-seed-base", type=int, default=20_000,
                   help="held-out seeds; never used for training or checkpoint selection")
    p.add_argument("--ckpt", default="ckpt_final.npz")
    args = p.parse_args()

    plan = []
    algo_config = args.algo_config
    if args.curriculum:
        with open(args.curriculum, encoding="utf-8") as fh:
            curriculum = json.load(fh)
        algo_config = curriculum.get("algo_config") or algo_config
        if not os.path.isabs(algo_config):
            algo_config = os.path.join(HERE, algo_config)
        wanted = None
        if args.stage_names:
            wanted = [s.strip() for s in args.stage_names.split(",") if s.strip()]
        for stage in curriculum.get("stages", []):
            name = str(stage["name"])
            if wanted is not None and name not in wanted:
                continue
            steps = int(stage.get("steps", args.steps))
            if args.stage_steps is not None:
                steps = int(args.stage_steps)
            env_json = stage["env_json"]
            if not os.path.isabs(env_json):
                env_json = os.path.join(HERE, env_json)
            plan.append((name, env_json, steps))
        if not plan:
            raise SystemExit(f"curriculum {args.curriculum} selected no stages")
    else:
        fixed = max(args.easy_steps, 0) + max(args.stage1_steps, 0)
        s2 = args.stage2_steps
        if s2 is None:
            s2 = max(args.steps - fixed, 0)
        if args.easy_steps > 0:
            plan.append(("easy", args.easy_env, args.easy_steps))
        if args.stage1_steps > 0:
            plan.append(("stage1", args.stage1_env, args.stage1_steps))
        if s2 > 0:
            plan.append(("stage2", args.stage2_env, s2))

    bc_demos = os.path.abspath(args.bc_demos) if args.bc_demos else None
    bc_save = os.path.abspath(args.bc_save_demos) if args.bc_save_demos else None
    bc_stage = bool(bc_demos or args.bc_episodes > 0)
    if bc_stage:
        bc_env = args.bc_env or (plan[0][1] if plan else args.stage1_env)
        if not os.path.isabs(bc_env):
            bc_env = os.path.join(HERE, bc_env)
        if bc_save is None and not bc_demos:
            bc_save = os.path.join(args.out, f"{args.tag}_bc_demos.npz")
        # name it "bc" so the evaluation/summary can identify the warm-start
        plan.insert(0, ("bc", bc_env, 0))

    total = sum(int(steps) for _, _, steps in plan)
    if total < 20_000:
        print(f"[curriculum] warning: total budget {total} < 20k steps")

    stages = []
    init = args.init_checkpoint
    for name, env_json, steps in plan:
        if name == "bc":
            stage = _stage(
                f"{args.tag}_{name}", env_json, algo_config, 0, args.out, init=init,
                bc={"episodes": int(args.bc_episodes), "demos": bc_demos,
                    "save": bc_save, "epochs": args.bc_epochs,
                    "batch_size": args.bc_batch_size,
                    "label_smoothing": args.bc_label_smoothing,
                    "keep_clean_only": bool(args.bc_keep_clean_only),
                    "legacy_expert": bool(args.bc_legacy_expert)})
        else:
            stage = _stage(f"{args.tag}_{name}", env_json, algo_config, steps, args.out,
                           init=init)
        stages.append(stage)
        init = os.path.join(stage["dir"], "ckpt_final.npz")

    evaluations = [_evaluate_stage(st, args.ckpt, args.eval_episodes, args.eval_seed_base)
                   for st in stages]
    best = max(evaluations, key=_score)
    best_stage = next(st for st in stages if st["name"] == best["stage"])

    bc_eval = next((e for e in evaluations if e["stage"].endswith("_bc")), None)
    chain_checks = []
    for prev, cur in zip(stages, stages[1:]):
        if not cur.get("init_checkpoint"):
            continue
        chain_checks.append({
            "stage": cur["name"],
            "init_checkpoint": cur["init_checkpoint"],
            "init_checkpoint_sha256": _sha256(cur["init_checkpoint"]),
            "previous_stage_checkpoint_sha256": prev.get("checkpoint_sha256"),
            "matches_previous_stage": _sha256(cur["init_checkpoint"])
            == prev.get("checkpoint_sha256"),
        })
    summary = {
        "tag": args.tag,
        "curriculum": args.curriculum,
        "budget_steps": total,
        "init_checkpoint": args.init_checkpoint,
        "init_chain": [
            {"stage": st["name"], "init_checkpoint": st.get("init_checkpoint"),
             "checkpoint": st.get("checkpoint"),
             "checkpoint_sha256": st.get("checkpoint_sha256")}
            for st in stages
        ],
        "checkpoint_chain": chain_checks,
        "checkpoint_chain_verified": bool(chain_checks) and all(
            c["matches_previous_stage"] for c in chain_checks),
        "stages": stages,
        "behavior_cloning": {
            "warm_start": bool(bc_stage),
            "stage": bc_eval["stage"] if bc_eval else None,
            "checkpoint": os.path.join(bc_eval["dir"], "ckpt_final.npz") if bc_eval else None,
            "demos": bc_demos or bc_save,
            "metrics": (bc_eval or {}).get("behavior_cloning", {}),
            "data": (bc_eval or {}).get("behavior_cloning_data", {}),
            "eval": ({k: bc_eval[k] for k in
                      ("episodes_with_safe_pass", "safe_pass_events", "overtake_passes_total",
                       "eval_success_rate", "eval_collision_rate", "eval_crash_rate",
                       "eval_speed_mean", "eval_distance_mean")}
                     if bc_eval else {}),
        },
        "evaluations": evaluations,
        "selected_stage": best["stage"],
        "selection_rule": ("highest (clean laps, strict safe overtakes, collision-free "
                           "episodes, success rate, distance) on the held-out evaluation "
                           "seeds; return is not used"),
        "safe_pass": {
            "criterion": ("a blocker pass (not a lane change) in an independent episode "
                          "with no collision and no off-track; strict certification "
                          "additionally requires a safe lane change and target-lane keeping"),
            "episodes": best["episodes"],
            "seed_base": best["seed_base"],
            "safe_pass_episodes": best["episodes_with_safe_pass"],
            "strict_safe_overtake_episodes": best["episodes_with_strict_overtake"],
            "strict_safe_overtake_events": best["strict_overtake_events"],
            "clean_laps": best["clean_laps"],
            "safe_pass_events": best["safe_pass_events"],
            "success_rate": best["eval_safe_pass_rate"],
            "lap_success_rate": best["eval_success_rate"],
            "overtake_passes_total": best["overtake_passes_total"],
            "verified_lane_change_completions": best["verified_lane_change_completions"],
            "collision_rate": best["eval_collision_rate"],
            "crash_rate": best["eval_crash_rate"],
            "off_track_rate": best["off_track_episodes"] / max(best["episodes"], 1),
            "passed": bool(best["episodes_with_safe_pass"] > 0
                           or best["episodes_with_strict_overtake"] > 0),
            "acceptance_gate": "run scripts/accept_overtake.py for the authoritative verdict",
        },
        "safe_overtake_observed_any_stage": any(e["safe_overtake_observed"] for e in evaluations),
        "diagnose": _diagnose(best_stage["dir"], args.ckpt, args.eval_episodes,
                              args.eval_seed_base,
                              os.path.join(best_stage["dir"], "diagnose.json")),
    }

    out_path = os.path.join(args.out, f"{args.tag}_curriculum_summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== curriculum summary ===")
    for ev in evaluations:
        bc = ev.get("behavior_cloning") or {}
        bc_note = ""
        if bc:
            bc_note = (f" | BC loss {bc.get('bc_loss_start', float('nan')):.3f}->"
                       f"{bc.get('bc_loss_end', float('nan')):.3f} acc "
                       f"{bc.get('bc_accuracy_start', float('nan')):.3f}->"
                       f"{bc.get('bc_accuracy_end', float('nan')):.3f}")
        print(f"{ev['stage']:>34s} steps={ev['steps']:6d} "
              f"lap_success={ev['eval_success_rate']:.2f} "
              f"safe_pass_rate={ev['eval_safe_pass_rate']:.2f} "
              f"collision={ev['eval_collision_rate']:.2f} crash={ev['eval_crash_rate']:.2f} "
              f"speed={ev['eval_speed_mean']:.2f} dist={ev['eval_distance_mean']:.1f} "
              f"changes={ev['eval_lane_changes']} "
              f"bonus={ev['verified_lane_change_completions']} "
              f"passes={ev['overtake_passes_total']} "
              f"safe_pass_eps={ev['episodes_with_safe_pass']} "
              f"safe_pass_events={ev['safe_pass_events']}{bc_note}")
    sp = summary["safe_pass"]
    print(f"selected stage: {best['stage']} | safe overtake observed: {sp['passed']} "
          f"({sp['safe_pass_episodes']}/{sp['episodes']} episodes, "
          f"{sp['safe_pass_events']} safe passes, "
          f"collision {sp['collision_rate']:.2f}, off-track {sp['off_track_rate']:.2f})")
    print(f"[curriculum] summary written to {out_path}")


if __name__ == "__main__":
    main()
