#!/usr/bin/env python
"""Train (or watch) one agent on MiniDrive.

Examples
--------
    python train.py --algo dqn --seed 0 --steps 120000 --tag dqn_s0
    python train.py --algo ppo --seed 0 --steps 120000 --tag ppo_s0
    python train.py --algo expert --watch          # scripted baseline, no learning
    python train.py --algo dqn --debug-trace 40    # print every step of episode 1

Everything a run produces lands in ``runs/<tag>/``:
    config.json      full configuration (env + algorithm + budget + git-free snapshot)
    episodes.csv     one row per training episode (return, crash, losses, ...)
    evals.csv        one row per greedy evaluation on fixed seeds
    ckpt_*.npz       parameters (numpy-only, no pickle)
    train.log        console transcript
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from autodrive.algos import REGISTRY, make_algorithm
from autodrive.baselines import PurePursuitDriver
from autodrive.demonstrations import collect_expert_demonstrations
from autodrive.env import MiniDriveEnv
from autodrive.logging_utils import (CSVLogger, OvertakeTracker, evaluate_agent,
                                     policy_lane_snapshot, save_config, selection_key)

HERE = os.path.dirname(os.path.abspath(__file__))

EPISODE_FIELDS = [
    "episode", "global_step", "return", "task_return", "steps", "success", "crash", "collision",
    "distance", "mean_speed", "lateral_rms", "min_npc_distance", "epsilon", "loss", "policy_loss",
    "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var",
    "risk_total", "risk_road", "risk_vehicle", "risk_ttc", "risk_ttc_front",
    "risk_ttc_rear", "ttc_min", "action_risk", "dense_weight",
    "lane_overtake", "lane_current_gap", "lane_best_gap",
    "overtake_lock", "overtake_cooldown", "overtake_hold_steps",
    "overtake_completions", "overtake_aborts", "overtake_state",
    "overtake_state_machine", "overtake_window", "progress_reward",
    "overtake_opportunity_steps", "overtake_toward_frac", "overtake_hold_frac",
    "overtake_brake_frac", "lane_changes", "lane_changes_toward", "lane_bonus_events",
    "overtake_window_steps", "overtake_passes",
    "sensor_safe_accel", "sensor_target_window", "sensor_ego_ttc", "sensor_lead_delta",
    "ttc_active_vehicles", "phase_is_ego", "adv_action", "wall_s",
]
EVAL_FIELDS = [
    "global_step", "episode", "eval_return_mean", "eval_return_min", "eval_success_rate",
    "eval_crash_rate", "eval_collision_rate", "eval_distance_mean", "eval_speed_mean",
    "eval_lateral_rms", "eval_min_npc_distance", "eval_lap_time_mean",
    "eval_overtake_opportunity_steps", "eval_overtake_toward_frac",
    "eval_overtake_hold_frac", "eval_overtake_brake_frac",
    "eval_lane_changes", "eval_lane_changes_toward", "eval_lane_bonus_events",
    "eval_overtake_window_steps", "eval_overtake_passes",
    "eval_clean_episodes", "eval_safe_pass_episodes", "eval_safe_pass_events",
    "eval_safe_pass_rate",
]


class Tee:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        msg = " ".join(str(p) for p in parts)
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()


def _sha256(path):
    """Content hash, so a summary can prove which checkpoint it started from."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_env(args) -> MiniDriveEnv:
    overrides = {}
    if args.env_json:
        with open(args.env_json, encoding="utf-8") as fh:
            overrides = json.load(fh)
    if args.no_random_start:
        overrides["random_start"] = False
    if args.lidar_noise is not None:
        overrides["lidar_noise"] = args.lidar_noise
    if args.obstacles is not None:
        overrides["n_obstacles"] = args.obstacles
    return MiniDriveEnv(seed=args.seed, **overrides)


def main():
    p = argparse.ArgumentParser(description="Train an agent on MiniDrive")
    p.add_argument("--algo", default="dqn", help="dqn | ppo | expert | random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=120_000, help="total environment steps (budget)")
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--final-eval-episodes", type=int, default=10)
    p.add_argument("--test-seed-base", type=int, default=20_000,
                   help="held-out test seeds: never used for model selection")
    p.add_argument("--test-episodes", type=int, default=10)
    p.add_argument("--out", default=os.path.join(HERE, "runs"))
    p.add_argument("--tag", default=None)
    p.add_argument("--config", default=None, help="JSON file with algorithm hyper-parameters")
    p.add_argument("--init-checkpoint", default=None,
                   help="load agent parameters before training (fine-tuning/transfer)")
    p.add_argument("--bc-episodes", type=int, default=0,
                   help="scripted-expert episodes used for actor warm-start")
    p.add_argument("--bc-demos", default=None,
                   help="pre-collected demonstration .npz (observations/actions); "
                        "overrides --bc-episodes and makes the warm-start reproducible")
    p.add_argument("--bc-save", default=None,
                   help="write the collected demonstrations to this .npz")
    p.add_argument("--bc-epochs", type=int, default=300)
    p.add_argument("--bc-batch-size", type=int, default=128)
    p.add_argument("--bc-label-smoothing", type=float, default=0.02)
    p.add_argument("--bc-keep-clean-only", action="store_true",
                   help="when collecting demonstrations, drop episodes that crashed "
                        "or collided (a crash is not a target behaviour)")
    p.add_argument("--bc-legacy-expert", action="store_true",
                   help="collect with the stateful PurePursuitDriver instead of the "
                        "default memoryless expert (non-Markov labels)")
    p.add_argument("--bc-only", action="store_true",
                   help="stop after the BC warm-start and evaluate it (steps forced to 0)")
    p.add_argument("--env-json", default=None, help="JSON file with environment overrides")
    p.add_argument("--no-random-start", action="store_true")
    p.add_argument("--lidar-noise", type=float, default=None)
    p.add_argument("--obstacles", type=int, default=None,
                   help="number of NPC vehicles on the track (0 = empty-track ablation)")
    p.add_argument("--no-checkpoints", action="store_true")
    p.add_argument("--debug-trace", type=int, default=0, metavar="N",
                   help="print a step-by-step table for the first N steps")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    tag = args.tag or f"{args.algo}_s{args.seed}"
    run_dir = os.path.join(args.out, tag)
    os.makedirs(run_dir, exist_ok=True)
    log = Tee(os.path.join(run_dir, "train.log"))

    algo_cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            algo_cfg = json.load(fh)

    env = build_env(args)
    agent = None
    if args.algo in REGISTRY:
        agent = make_algorithm(args.algo, env.obs_dim, env.n_actions, cfg=algo_cfg, seed=args.seed)
    elif args.algo == "expert":
        agent = PurePursuitDriver(env)
    elif args.algo == "random":
        rng = np.random.default_rng(args.seed)

        class RandomAgent:
            name = "random"

            def act(self, obs, greedy=False):
                return int(rng.integers(env.n_actions))

            def update(self, *a, **k):
                return None

        agent = RandomAgent()
    else:
        raise SystemExit(f"unknown --algo {args.algo}")

    # RARL (and any future algorithm that needs the simulation itself) binds here
    if hasattr(agent, "bind_env"):
        agent.bind_env(env)
    init_meta = {"path": None, "sha256": None}
    if args.init_checkpoint:
        if not hasattr(agent, "load_state_dict"):
            raise SystemExit("--init-checkpoint requires a learnable algorithm")
        with np.load(args.init_checkpoint) as checkpoint:
            agent.load_state_dict(dict(checkpoint))
        init_meta = {"path": os.path.abspath(args.init_checkpoint),
                     "sha256": _sha256(args.init_checkpoint)}
        source_cfg = os.path.join(os.path.dirname(os.path.abspath(args.init_checkpoint)),
                                  "config.json")
        if os.path.exists(source_cfg):
            try:
                with open(source_cfg, encoding="utf-8") as fh:
                    src = json.load(fh)
                init_meta["source_behavior_cloning"] = src.get("behavior_cloning") or {}
                init_meta["source_behavior_cloning_data"] = {
                    k: v for k, v in (src.get("behavior_cloning_data") or {}).items()
                    if k != "episode_stats"
                }
            except (OSError, json.JSONDecodeError):
                pass
        log(f"[init] loaded parameters from {args.init_checkpoint} "
            f"(sha256 {init_meta['sha256'][:12]})")
    bc_result = {}
    bc_stats = {}
    bc_masks = None
    bc_demos_path = args.bc_demos
    if args.bc_demos or args.bc_episodes > 0:
        if not hasattr(agent, "policy") and not hasattr(agent, "pretrain"):
            raise SystemExit("--bc-* requires an algorithm with a policy")
        if args.bc_demos:
            with np.load(args.bc_demos, allow_pickle=False) as demo:
                x = np.asarray(demo["observations"], dtype=float)
                y = np.asarray(demo["actions"], dtype=np.int64)
                if "masks" in demo.files:
                    bc_masks = np.asarray(demo["masks"], dtype=bool)
                raw_stats = demo["stats"] if "stats" in demo.files else None
            if raw_stats is not None:
                text = raw_stats.item() if getattr(raw_stats, "shape", ()) == () else raw_stats
                bc_stats = json.loads(str(text))
            else:
                bc_stats = {"source": "npz", "steps": int(x.shape[0])}
            bc_stats.setdefault("source", "npz")
            log(f"[bc] loaded {x.shape[0]} samples x {x.shape[1]} features from "
                f"{args.bc_demos}"
                + (f" (with safety masks)" if bc_masks is not None else ""))
        else:
            x, y, bc_stats = collect_expert_demonstrations(
                env, agent, args.bc_episodes, seed_base=30_000,
                memoryless=not args.bc_legacy_expert,
                keep_clean_only=args.bc_keep_clean_only)
            bc_masks = bc_stats.pop("masks", None)
            bc_stats["source"] = "collected"
            if args.bc_save:
                os.makedirs(os.path.dirname(os.path.abspath(args.bc_save)), exist_ok=True)
                save_kwargs = {"observations": x, "actions": y,
                               "stats": np.array(json.dumps(bc_stats))}
                if bc_masks is not None:
                    save_kwargs["masks"] = bc_masks
                np.savez_compressed(args.bc_save, **save_kwargs)
                bc_demos_path = args.bc_save
                log(f"[bc] demonstrations written to {args.bc_save}")
        pretrain = getattr(agent, "pretrain", None) or agent.policy.pretrain
        pretrain_kwargs = dict(epochs=args.bc_epochs, batch_size=args.bc_batch_size,
                               label_smoothing=args.bc_label_smoothing)
        if bc_masks is not None:
            import inspect
            if "action_mask" in inspect.signature(pretrain).parameters:
                pretrain_kwargs["action_mask"] = bc_masks
            else:
                log("[bc] WARNING: this algorithm's pretrain has no action_mask; "
                    "the recorded safety mask is ignored")
        bc_result = pretrain(x, y, **pretrain_kwargs)
        log(f"[bc] demos: {bc_stats.get('steps', x.shape[0])} steps / "
            f"{bc_stats.get('episodes', '?')} episodes "
            f"(kept {bc_stats.get('episodes_kept', '?')}) | "
            f"lane changes {bc_stats.get('lane_changes', 0)} "
            f"(blocked {bc_stats.get('blocked_lane_changes', 0)}) | "
            f"overtake opportunities {bc_stats.get('overtake_opportunity_steps', 0)} | "
            f"overtakes completed {bc_stats.get('overtakes_completed', 0)} | "
            f"mask conflicts {bc_stats.get('mask_conflicts', 0)}")
        log(f"[bc] {bc_result}")
        if int(bc_stats.get("overtakes_completed", 0)) == 0:
            log("[bc] WARNING: the scripted expert completed no overtake in these "
                "episodes; the demonstrations contain no overtake label.")
        # The warm-started actor is checkpointed before a single PPO step, so a
        # caller can prove the BC parameters are what the fine-tune starts from.
        if hasattr(agent, "state_dict"):
            _save(agent, os.path.join(run_dir, "ckpt_bc.npz"))
            log(f"[bc] checkpoint -> {os.path.join(run_dir, 'ckpt_bc.npz')}")
    if args.bc_only:
        args.steps = 0

    cfg_dump = {
        "tag": tag,
        "algo": args.algo,
        "seed": args.seed,
        "steps": args.steps,
        "algorithm_cfg": getattr(agent, "cfg", algo_cfg),
        "env_cfg": env.cfg,
        "eval_seeds": {"base": 10_000, "episodes": args.eval_episodes},
        "episode_seed_base": args.seed * 1_000_003,
        "init_checkpoint": args.init_checkpoint,
        "init_checkpoint_sha256": init_meta.get("sha256"),
        "behavior_cloning": bc_result,
        "behavior_cloning_data": bc_stats,
        "behavior_cloning_demos": bc_demos_path,
        "behavior_cloning_checkpoint":
            os.path.join(run_dir, "ckpt_bc.npz") if bc_result else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_config(os.path.join(run_dir, "config.json"), cfg_dump)

    log(f"=== {args.algo} | tag={tag} | seed={args.seed} | budget={args.steps} steps ===")
    log(env.describe())
    log("env:", json.dumps(env.cfg, ensure_ascii=False))
    if getattr(agent, "cfg", None):
        log("algo:", json.dumps(agent.cfg, ensure_ascii=False))

    episodes_csv = CSVLogger(os.path.join(run_dir, "episodes.csv"), EPISODE_FIELDS)
    evals_csv = CSVLogger(os.path.join(run_dir, "evals.csv"), EVAL_FIELDS)

    # ---- reference baselines on the same eval protocol ----------------------
    ref, ref_test = {}, {}
    for name, maker in (("random", None), ("expert", None)):
        if args.algo == name:
            continue
        ref_env = build_env(args)
        if name == "expert":
            ref_agent = PurePursuitDriver(ref_env)
        else:
            rng = np.random.default_rng(0)

            class R:
                def act(self, obs, greedy=False):
                    return int(rng.integers(ref_env.n_actions))
            ref_agent = R()
        ref[name] = evaluate_agent(ref_env, ref_agent, args.eval_episodes, base_seed=10_000)
        ref_test[name] = evaluate_agent(ref_env, ref_agent, args.test_episodes, base_seed=args.test_seed_base)
        log(f"[reference] {name:6s} return={ref[name]['eval_return_mean']:8.2f} "
            f"(test {ref_test[name]['eval_return_mean']:8.2f}) "
            f"success={ref[name]['eval_success_rate']:.2f} "
            f"speed={ref[name]['eval_speed_mean']:5.2f} m/s")

    # ---- training loop ------------------------------------------------------
    t_start = time.perf_counter()
    steps_done = 0
    episode = 0
    best_key = None
    next_eval = args.eval_every
    trace_left = args.debug_trace
    last_eval = {}

    if args.algo in ("expert", "random"):
        # no learning: single evaluation, logged as step 0
        res = evaluate_agent(env, agent, args.final_eval_episodes, base_seed=10_000)
        evals_csv.log(global_step=0, episode=0, **{k: v for k, v in res.items() if not k.startswith("_")})
        log(f"[{args.algo}] return={res['eval_return_mean']:.2f} success={res['eval_success_rate']:.2f} "
            f"lap_time={res['eval_lap_time_mean']:.1f}s")
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}))
        return

    prev_log = {}
    while steps_done < args.steps:
        ep_seed = args.seed * 1_000_003 + episode * 7919
        obs, _ = env.reset(seed=ep_seed)
        overtake = OvertakeTracker()
        overtake.start_episode(env)
        ep_return, ep_steps, done = 0.0, 0, False
        speeds, lats = [], []
        ep_return_task = 0.0

        while not done:
            a = agent.act(obs)
            lane = policy_lane_snapshot(agent)
            next_obs, r, term, trunc, info = env.step(a)
            out = agent.update(obs, a, r, next_obs, term, trunc, info)
            if out:
                prev_log = out
            overtake.observe(env, a, lane=lane,
                             bonus=float(getattr(agent, "_lane_bonus_taken", 0.0)))
            ep_return += r
            ep_return_task += r
            ep_steps += 1
            steps_done += 1
            speeds.append(info["speed"])
            lats.append(abs(info["lateral"]))

            if trace_left > 0:
                log(f"  [trace] step={ep_steps:4d} a={info['action']} v={info['speed']:5.2f} "
                    f"lat={info['lateral']:6.2f} ds={info['ds']:6.2f} r={r:7.3f} "
                    f"prog={info['progress']:7.1f} obs0..3={np.round(obs[:4], 2)}")
                trace_left -= 1

            obs = next_obs
            if (not (term or trunc)) and steps_done >= args.steps:
                trunc = True
            done = term or trunc

        episode += 1
        ep_info = {
            "episode": episode,
            "global_step": steps_done,
            "return": round(ep_return, 4),
            "task_return": round(ep_return_task, 4),
            "steps": ep_steps,
            "success": int(info["success"]),
            "crash": int(info["crash"]),
            "collision": int(info.get("collision", False)),
            "distance": round(info["progress"], 2),
            "mean_speed": round(float(np.mean(speeds)), 3) if speeds else 0.0,
            "lateral_rms": round(float(np.sqrt(np.mean(np.square(lats)))), 3) if lats else 0.0,
            "min_npc_distance": round(float(info.get("npc_min_distance", float("nan"))), 3),
            "wall_s": round(time.perf_counter() - t_start, 2),
        }
        ep_info.update({k: round(float(v), 5) for k, v in prev_log.items()})
        ep_info.update({k: round(float(v), 4) for k, v in agent.stats().items()})
        ep_info.update({k: round(float(v), 4) if isinstance(v, float) else int(v)
                        for k, v in overtake.episode_summary().items()})
        episodes_csv.log(**ep_info)

        if not args.quiet and (episode % 5 == 0):
            log(f"ep {episode:4d} | step {steps_done:7d} | return {ep_return:8.2f} | "
                f"dist {info['progress']:6.1f}m | speed {ep_info['mean_speed']:5.2f} | "
                f"success {ep_info['success']} crash {ep_info['crash']} "
                f"collision {ep_info['collision']} | "
                f"overtake opp {ep_info['overtake_opportunity_steps']:4d} "
                f"change {ep_info['overtake_toward_frac']:.2f} "
                f"hold {ep_info['overtake_hold_frac']:.2f} "
                f"brake {ep_info['overtake_brake_frac']:.2f} "
                f"lc {ep_info['lane_changes']} done {ep_info['lane_bonus_events']} "
                f"pass {ep_info['overtake_passes']} | "
                + " ".join(f"{k}={float(v):.3f}" for k, v in prev_log.items() if k in ("loss", "entropy")))

        # ---- periodic greedy evaluation ------------------------------------
        if steps_done >= next_eval or steps_done >= args.steps:
            last_eval = evaluate_agent(env, agent, args.eval_episodes, base_seed=10_000)
            evals_csv.log(global_step=steps_done, episode=episode,
                          **{k: v for k, v in last_eval.items() if not k.startswith("_")})
            log(f"[eval] step {steps_done:7d} | return {last_eval['eval_return_mean']:8.2f} | "
                f"success {last_eval['eval_success_rate']:.2f} | crash {last_eval['eval_crash_rate']:.2f} | "
                f"collision {last_eval['eval_collision_rate']:.2f} | "
                f"speed {last_eval['eval_speed_mean']:5.2f} m/s | "
                f"overtake opp {last_eval['eval_overtake_opportunity_steps']} "
                f"change {last_eval['eval_overtake_toward_frac']:.2f} "
                f"hold {last_eval['eval_overtake_hold_frac']:.2f} "
                f"brake {last_eval['eval_overtake_brake_frac']:.2f} "
                f"lc {last_eval['eval_lane_changes']} "
                f"done {last_eval['eval_lane_bonus_events']} "
                f"pass {last_eval['eval_overtake_passes']}")
            if not args.no_checkpoints:
                _save(agent, os.path.join(run_dir, f"ckpt_step{steps_done}.npz"))
                key = selection_key(last_eval)
                if best_key is None or key > best_key:
                    best_key = key
                    _save(agent, os.path.join(run_dir, "ckpt_best.npz"))
                    log(f"[select] new best at step {steps_done}: "
                        f"clean_laps={key[0]} strict_overtakes={key[1]} "
                        f"clean={key[2]} success={key[3]:.2f} distance={key[4]:.1f}")
            while next_eval <= steps_done:
                next_eval += args.eval_every

    # ---- final evaluation ---------------------------------------------------
    # validation set (used for checkpoint selection) ...
    final = evaluate_agent(env, agent, args.final_eval_episodes, base_seed=10_000)
    evals_csv.log(global_step=steps_done, episode=episode,
                  **{k: v for k, v in final.items() if not k.startswith("_")})
    if not args.no_checkpoints:
        _save(agent, os.path.join(run_dir, "ckpt_final.npz"))

    # ... and a *held-out* test set that model selection never touched
    test_final = evaluate_agent(env, agent, args.test_episodes, base_seed=args.test_seed_base)
    test_best = {}
    best_path = os.path.join(run_dir, "ckpt_best.npz")
    if (not args.no_checkpoints) and os.path.exists(best_path):
        probe = make_algorithm(args.algo, env.obs_dim, env.n_actions, cfg=algo_cfg, seed=args.seed)
        probe.load_state_dict(dict(np.load(best_path)))
        if hasattr(probe, "bind_env"):
            probe.bind_env(env)      # RARL re-attaches its frozen adversary
        test_best = evaluate_agent(env, probe, args.test_episodes, base_seed=args.test_seed_base)
        log(f"[test] best-checkpoint  return {test_best['eval_return_mean']:8.2f} | "
            f"success {test_best['eval_success_rate']:.2f} | lap {test_best['eval_lap_time_mean']:5.1f}s")
    log(f"[test] final-weights    return {test_final['eval_return_mean']:8.2f} | "
        f"success {test_final['eval_success_rate']:.2f} | lap {test_final['eval_lap_time_mean']:5.1f}s")

    elapsed = time.perf_counter() - t_start
    clean = lambda d: {k: round(float(v), 4) for k, v in d.items() if not k.startswith("_")}

    def safe_pass_summary(ev):
        """Safe-pass accounting for one evaluation set (validation or held-out)."""
        rows = ev.get("_rows", []) or []
        n = len(rows)
        clean_rows = [r for r in rows if r.get("clean")]
        safe_rows = [r for r in rows if r.get("safe_pass")]
        strict_rows = [r for r in rows if r.get("strict_safe_overtake")]
        clean_laps = [r for r in rows if r.get("clean_lap")]
        return {
            "episodes": n,
            "safe_pass_episodes": len(safe_rows),
            "safe_pass_events": int(sum(int(r.get("safe_pass_events", 0)) for r in rows)),
            "safe_pass_rate": (len(safe_rows) / n) if n else 0.0,
            "strict_safe_overtake_episodes": len(strict_rows),
            "strict_safe_overtake_rate": (len(strict_rows) / n) if n else 0.0,
            "strict_safe_overtake_events":
                int(sum(int(r.get("strict_events", 0)) for r in rows)),
            "overtake_passes": int(sum(int(r["overtake"]["overtake_passes"]) for r in rows)),
            "verified_lane_change_completions":
                int(sum(int(r["overtake"]["lane_bonus_events"]) for r in rows)),
            "clean_episodes": len(clean_rows),
            "clean_laps": len(clean_laps),
            "clean_lap_rate": (len(clean_laps) / n) if n else 0.0,
            "lap_success_rate": (sum(int(r["success"]) for r in rows) / n) if n else 0.0,
            "collision_rate": (sum(int(r["collision"]) for r in rows) / n) if n else 0.0,
            "off_track_rate":
                (sum(1 for r in rows if r["crash"] and not r["collision"]) / n) if n else 0.0,
            "crash_rate": (sum(int(r["crash"]) for r in rows) / n) if n else 0.0,
            "episodes_detail": [
                {"seed": r["seed"], "steps": r["steps"], "distance": round(float(r["distance"]), 1),
                 "success": r["success"], "crash": r["crash"], "collision": r["collision"],
                 "clean_lap": bool(r.get("clean_lap", False)),
                 "passes": int(r["overtake"]["overtake_passes"]),
                 "lane_bonus_events": int(r["overtake"]["lane_bonus_events"]),
                 "safe_pass": bool(r.get("safe_pass", False)),
                 "strict_events": int(r.get("strict_events", 0)),
                 "strict_safe_overtake": bool(r.get("strict_safe_overtake", False))}
                for r in rows
            ],
        }

    summary = {
        "algo": args.algo,
        "seed": args.seed,
        "steps": steps_done,
        "episodes": episode,
        "wall_s": round(elapsed, 2),
        "steps_per_s": round(steps_done / max(elapsed, 1e-9), 1),
        "init_checkpoint": args.init_checkpoint,
        "warm_start": init_meta,
        "behavior_cloning": bc_result,
        "behavior_cloning_data": bc_stats,
        "behavior_cloning_checkpoint":
            os.path.join(run_dir, "ckpt_bc.npz") if bc_result else None,
        "validation_eval": clean(final),
        "validation_safe_pass": safe_pass_summary(final),
        "test_eval_final": clean(test_final),
        "test_safe_pass_final": safe_pass_summary(test_final),
        "test_eval_best": clean(test_best),
        "test_safe_pass_best": safe_pass_summary(test_best),
        "reference": {k: clean(v) for k, v in ref.items()},
        "reference_test": {k: clean(v) for k, v in ref_test.items()},
        "checkpoint_selection": {
            "rule": ("highest (clean laps, strict safe overtakes, collision-free "
                     "episodes, lap-success rate, distance) on the validation seeds; "
                     "return is not used"),
            "best_key": list(best_key) if best_key is not None else None,
        },
    }
    # The acceptance criterion is explicitly about the held-out test episodes.
    summary["acceptance"] = {
        "criterion": ("strict gate: >= 6/10 clean laps and >= 3/10 certified strict "
                      "overtakes (safe lane change, blocker passed, target lane kept, "
                      "no collision, no off-track) on 10 independent held-out episodes; "
                      "run scripts/accept_overtake.py for the authoritative verdict"),
        "test_seed_base": args.test_seed_base,
        "test_episodes": args.test_episodes,
        "clean_laps": summary["test_safe_pass_final"]["clean_laps"],
        "strict_safe_overtakes":
            summary["test_safe_pass_final"]["strict_safe_overtake_episodes"],
        "safe_pass_observed": summary["test_safe_pass_final"]["safe_pass_episodes"] > 0,
    }
    save_config(os.path.join(run_dir, "summary.json"), summary)
    log("=== done ===")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    episodes_csv.close()
    evals_csv.close()


def _save(agent, path):
    if hasattr(agent, "state_dict"):
        np.savez_compressed(path, **agent.state_dict())


if __name__ == "__main__":
    main()
