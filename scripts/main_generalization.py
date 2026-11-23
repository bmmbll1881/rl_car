#!/usr/bin/env python
"""Adapt the Stage1/2 overtake specialist to the main 24-NPC traffic stream.

The main environment is the archived ``runs/ppo_s0/config.json`` ``env_cfg``
(24 NPC, 5.5--8.0 m/s, ``v_max`` 16 m/s, random start).  It is saved next to the
run by :func:`cmd_env` so the acceptance CLI can be pointed at the exact same
configuration.

The three steps are deliberately separate:

``collect``
    Drive the main environment with ``MemorylessExpertDriver`` at the strict
    verifier's 18 m block distance and store the learner's own 97-dim policy
    input, the expert action and the safety mask that was in force.

``train``
    Load the recent Stage1/2 specialist checkpoint, concatenate the main
    demonstrations with a *small* Stage1/2 anchor (the sparse exact dumps), and
    call ``RARL.pretrain`` / ``PPO.pretrain`` directly.  ``train.py`` is not used:
    it hard-codes a validation evaluation at seed 10000, which belongs to the
    host's reserved band.

``dagger``
    Optional single correction round: roll out the learner on the main
    environment, ask the memoryless expert for the label at every learner state,
    append those pairs and retrain once.

Every artifact stays under ``runs/main_generalization*``; no core source file is
modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from autodrive.algos import make_algorithm  # noqa: E402
from autodrive.baselines import MemorylessExpertDriver  # noqa: E402
from autodrive.demonstrations import collect_expert_demonstrations  # noqa: E402
from autodrive.env import MiniDriveEnv  # noqa: E402
from autodrive.logging_utils import advance_agent, reset_agent, save_config  # noqa: E402

DEFAULT_RUN = os.path.join(HERE, "runs", "main_generalization_bc")
DEFAULT_CONFIG = os.path.join(HERE, "configs", "rarl_bc_sensor.json")
MAIN_SOURCE = os.path.join(HERE, "runs", "ppo_s0", "config.json")
ANCHOR_SOURCE = os.path.join(HERE, "runs", "robust_stage12_sm_mixed_demos.npz")
BLOCK_DISTANCE = 18.0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _main_env_json(run_dir):
    """Write the exact archived ppo_s0 environment next to the run."""
    with open(MAIN_SOURCE, encoding="utf-8") as fh:
        env_cfg = dict(json.load(fh)["env_cfg"])
    path = os.path.join(run_dir, "env_main_ppo_s0.json")
    _json(path, env_cfg)
    return path, env_cfg


def _env(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    return MiniDriveEnv(seed=0, **cfg)


def _agent(env, config_path, seed=0):
    with open(config_path, encoding="utf-8") as fh:
        algo_cfg = json.load(fh)
    agent = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg=algo_cfg, seed=seed)
    agent.bind_env(env)
    return agent, algo_cfg


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        obs = np.asarray(data["observations"], dtype=float)
        act = np.asarray(data["actions"], dtype=np.int64)
        masks = np.asarray(data["masks"], dtype=bool) if "masks" in data.files else None
        stats = None
        if "stats" in data.files:
            raw = data["stats"]
            stats = json.loads(str(raw.item() if getattr(raw, "shape", ()) == () else raw))
    return obs, act, masks, stats


def _save_demos(path, obs, act, masks, stats):
    payload = {"observations": obs, "actions": act,
               "stats": np.array(json.dumps(stats))}
    if masks is not None:
        payload["masks"] = masks
    np.savez_compressed(path, **payload)


def _anchor(anchor_path, n_samples, seed=0):
    """Take a small, balanced subset of the sparse exact Stage1/2 demos."""
    obs, act, masks, stats = _load_npz(anchor_path)
    if masks is None:
        masks = np.ones((obs.shape[0], 7), dtype=bool)
    sources = (stats or {}).get("sources") or []
    split = int(sources[0]["samples"]) if sources else obs.shape[0] // 2
    rng = np.random.default_rng(int(seed))
    n1 = n_samples // 2
    n2 = n_samples - n1
    idx1 = rng.choice(np.arange(0, split), size=min(n1, split), replace=False)
    idx2 = rng.choice(np.arange(split, obs.shape[0]),
                      size=min(n2, obs.shape[0] - split), replace=False)
    idx = np.sort(np.concatenate([idx1, idx2]))
    meta = {
        "path": os.path.abspath(anchor_path),
        "sha256": _sha256(anchor_path),
        "source_samples": int(obs.shape[0]),
        "used_samples": int(idx.size),
        "block_distance": (stats or {}).get("block_distance"),
        "sources": sources,
    }
    return obs[idx], act[idx], masks[idx], meta


def _combine(parts):
    """``parts`` entries are ``(name, (obs, actions, masks), meta)``."""
    obs = np.concatenate([p[1][0] for p in parts], axis=0)
    act = np.concatenate([p[1][1] for p in parts], axis=0)
    masks = np.concatenate([p[1][2] for p in parts], axis=0)
    return obs, act, masks


def _train(run_dir, config_path, init_checkpoint, demo_parts, out_ckpt,
           epochs, batch_size, label_smoothing, tag, extra_summary=None):
    env_json, env_cfg = _main_env_json(run_dir)
    env = _env(env_json)
    agent, _ = _agent(env, config_path, seed=0)
    with np.load(init_checkpoint) as state:
        agent.load_state_dict(dict(state))

    obs, act, masks = _combine(demo_parts)
    started = time.time()
    result = agent.pretrain(obs, act, epochs=epochs, batch_size=batch_size,
                            label_smoothing=label_smoothing, action_mask=masks)
    elapsed = time.time() - started

    path = os.path.join(run_dir, out_ckpt)
    np.savez_compressed(path, **agent.state_dict())
    cfg_dump = {
        "tag": tag,
        "algo": "rarl",
        "seed": 0,
        "steps": 0,
        "algorithm_cfg": agent.cfg,
        "env_cfg": env_cfg,
        "env_json": env_json,
        "init_checkpoint": os.path.abspath(init_checkpoint),
        "init_checkpoint_sha256": _sha256(init_checkpoint),
        "candidate_checkpoint": path,
        "candidate_checkpoint_sha256": _sha256(path),
        "behavior_cloning": result,
        "behavior_cloning_data": {
            "samples": int(obs.shape[0]),
            "parts": [{"name": name, "samples": int(data[0].shape[0]), **meta}
                      for (name, data, meta) in demo_parts],
        },
        "train_seconds": round(elapsed, 2),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_summary:
        cfg_dump.update(extra_summary)
    save_config(os.path.join(run_dir, "config.json"), cfg_dump)
    print(f"[train] {tag}: {obs.shape[0]} samples x {obs.shape[1]} -> {path}")
    print("[train] " + json.dumps(result))
    return cfg_dump


# ------------------------------------------------------------------ commands
def cmd_env(args):
    run_dir = args.run
    os.makedirs(run_dir, exist_ok=True)
    path, cfg = _main_env_json(run_dir)
    env = MiniDriveEnv(seed=0, **cfg)
    print(f"[env] {path}")
    print(env.describe())
    print("[env] " + json.dumps(cfg, sort_keys=True))
    return 0


def cmd_collect(args):
    os.makedirs(args.run, exist_ok=True)
    env_json, _ = _main_env_json(args.run)
    env = _env(env_json)
    agent, _ = _agent(env, args.config, seed=0)

    obs, act, stats = collect_expert_demonstrations(
        env, agent, args.episodes, seed_base=args.seed_base,
        block_distance=args.block_distance, keep_clean_only=args.keep_clean_only)
    masks = stats.pop("masks")
    stats["env"] = os.path.abspath(env_json)
    stats["env_cfg"] = env.cfg

    path = os.path.join(args.run, args.out)
    _save_demos(path, obs, act, masks, stats)
    _json(os.path.join(args.run, "collect_main.json"), stats)
    print(f"[collect] {obs.shape[0]} samples x {obs.shape[1]} -> {path}")
    print(f"[collect] seeds {args.seed_base}..{args.seed_base + args.episodes - 1} | "
          f"kept {stats['episodes_kept']}/{stats['episodes']} | "
          f"blocked changes {stats['blocked_lane_changes']} | "
          f"overtakes {stats['overtakes_completed']} | "
          f"mask conflicts {stats['mask_conflicts']}")
    for ep in stats["episode_stats"]:
        print(f"  ep {ep['episode']:3d} seed {ep['seed']}: steps {ep['steps']:4d} "
              f"success {ep['success']} crash {ep['crash']} collision {ep['collision']} "
              f"lane changes {ep['lane_changes']:3d} overtakes {ep['overtakes_completed']} "
              f"kept {ep['kept']}")
    return 0


def cmd_train(args):
    main_path = os.path.join(args.run, args.demos)
    main_obs, main_act, main_masks, main_stats = _load_npz(main_path)
    if main_masks is None:
        main_masks = np.ones((main_obs.shape[0], 7), dtype=bool)
    parts = [("main", (main_obs, main_act, main_masks),
              {"path": os.path.abspath(main_path), "sha256": _sha256(main_path),
               "seeds": f"{args.main_seed_base}.."
                        f"{args.main_seed_base + (main_stats or {}).get('episodes', 0) - 1}",
               "overtakes": (main_stats or {}).get("overtakes_completed"),
               "block_distance": (main_stats or {}).get("block_distance")})]

    a_obs, a_act, a_masks, a_meta = _anchor(args.anchor, args.anchor_samples, args.anchor_seed)
    parts.append(("anchor_stage12", (a_obs, a_act, a_masks), a_meta))

    if args.dagger_demos:
        for d_name in args.dagger_demos:
            d_path = os.path.join(args.run, d_name)
            d_obs, d_act, d_masks, d_stats = _load_npz(d_path)
            if d_masks is None:
                d_masks = np.ones((d_obs.shape[0], 7), dtype=bool)
            parts.append((f"dagger:{d_name}", (d_obs, d_act, d_masks),
                          {"path": os.path.abspath(d_path), "sha256": _sha256(d_path),
                           "stats": {k: v for k, v in (d_stats or {}).items()
                                     if k != "episode_stats"}}))

    out_ckpt = args.out_ckpt or ("ckpt_dagger.npz" if args.dagger_demos else "ckpt_bc.npz")
    tag = "main_generalization" + ("_dagger" if args.dagger_demos else "_bc")
    _train(args.run, args.config, args.init_checkpoint, parts, out_ckpt,
           args.epochs, args.batch_size, args.label_smoothing, tag)
    return 0


def cmd_dagger(args):
    """One DAgger collection pass: learner states, expert labels."""
    run_dir = args.run
    env_json = args.env_json or os.path.join(run_dir, "env_main_ppo_s0.json")
    if not os.path.isabs(env_json):
        env_json = os.path.join(run_dir, env_json)
    env = _env(env_json)
    agent, _ = _agent(env, args.config, seed=0)
    ckpt = os.path.join(run_dir, args.ckpt)
    with np.load(ckpt) as state:
        agent.load_state_dict(dict(state))
    driver = MemorylessExpertDriver(env, block_distance=args.block_distance)

    observations, actions, masks = [], [], []
    stats = {"expert": driver.name, "block_distance": args.block_distance,
             "env": os.path.abspath(env_json), "label": args.label,
             "episodes": 0, "steps": 0, "learner_success": 0, "learner_crash": 0,
             "learner_collision": 0, "disagreements": 0, "episode_stats": []}

    for episode in range(args.episodes):
        seed = args.seed_base + episode
        obs, _ = env.reset(seed=seed)
        reset_agent(agent, obs)
        done = False
        steps = 0
        info = {"success": False, "crash": False, "collision": False}
        while not done:
            feats = agent._features(obs)
            mask = agent.action_mask()
            learner_action = agent.act(obs, greedy=True)
            expert_action = int(driver.act())
            observations.append(np.array(feats, dtype=float, copy=True))
            actions.append(expert_action)
            masks.append(np.array(mask, dtype=bool, copy=True))
            stats["disagreements"] += int(learner_action != expert_action)

            obs, _r, term, trunc, info = env.step(learner_action)
            advance_agent(agent, obs, term, trunc)
            steps += 1
            done = term or trunc or steps >= args.max_steps

        stats["episodes"] += 1
        stats["steps"] += steps
        stats["learner_success"] += int(info.get("success", False))
        stats["learner_crash"] += int(info.get("crash", False))
        stats["learner_collision"] += int(info.get("collision", False))
        stats["episode_stats"].append({
            "episode": episode, "seed": seed, "steps": steps,
            "success": int(info.get("success", False)),
            "crash": int(info.get("crash", False)),
            "collision": int(info.get("collision", False)),
        })
        print(f"  dagger {args.label} ep {episode:3d} seed {seed}: steps {steps:4d} "
              f"success {int(info.get('success', False))} "
              f"crash {int(info.get('crash', False))} "
              f"collision {int(info.get('collision', False))}")

    obs_arr = np.asarray(observations, dtype=float)
    act_arr = np.asarray(actions, dtype=np.int64)
    mask_arr = np.asarray(masks, dtype=bool)
    path = os.path.join(run_dir, args.out or f"demos_dagger_{args.label}.npz")
    _save_demos(path, obs_arr, act_arr, mask_arr, stats)
    _json(os.path.join(run_dir, f"dagger_stats_{args.label}.json"), stats)
    print(f"[dagger:{args.label}] {obs_arr.shape[0]} samples -> {path} | "
          f"disagreement {stats['disagreements']}/{stats['steps']} "
          f"({stats['disagreements'] / max(stats['steps'], 1):.1%})")
    return 0


def _gate(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    g, m = data["gate"], data["metrics"]
    return {"json": os.path.abspath(path), "clean_laps": g["clean_laps"],
            "strict_episodes": g["strict_safe_overtakes"], "passed": g["passed"],
            "collision_rate": m["eval_collision_rate"], "crash_rate": m["eval_crash_rate"],
            "speed": round(m["eval_speed_mean"], 3),
            "distance": round(m["eval_distance_mean"], 1),
            "strict_events": m["eval_strict_safe_overtake_events"]}


def cmd_finalize(args):
    """Copy the trained checkpoint to ckpt_candidate.npz and write summary.json."""
    run_dir = args.run
    src = os.path.join(run_dir, args.source)
    dst = os.path.join(run_dir, "ckpt_candidate.npz")
    with open(src, "rb") as fh:
        payload = fh.read()
    with open(dst, "wb") as fh:
        fh.write(payload)

    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["tag"] = "main_generalization_candidate"
    cfg["candidate_checkpoint"] = os.path.abspath(dst)
    cfg["candidate_checkpoint_sha256"] = _sha256(dst)
    cfg["candidate_checkpoint_source"] = os.path.abspath(src)
    save_config(cfg_path, cfg)

    results = {
        "gate0_expert_main_57300": _gate(os.path.join(run_dir, "gate0_expert_main_57300_e18.json")),
        "specialist_baseline_main_57300": _gate(os.path.join(run_dir, "baseline_specialist_main_57300.json")),
        "bc_v1_main_57300": _gate(os.path.join(run_dir, "accept_bc_main_57300.json")),
        "candidate_main_57300": _gate(os.path.join(run_dir, "accept_candidate_main_57300.json")),
        "candidate_main_57310_stability": _gate(os.path.join(run_dir, "accept_candidate_main_57310.json")),
        "specialist_baseline_stage1_57400": _gate(os.path.join(run_dir, "baseline_specialist_stage1_57400.json")),
        "bc_v1_stage1_57400": _gate(os.path.join(run_dir, "accept_bc_stage1_57400.json")),
        "candidate_stage1_57400": _gate(os.path.join(run_dir, "accept_candidate_stage1_57400.json")),
        "specialist_baseline_stage2_57500": _gate(os.path.join(run_dir, "baseline_specialist_stage2_57500.json")),
        "bc_v1_stage2_57500": _gate(os.path.join(run_dir, "accept_bc_stage2_57500.json")),
        "candidate_stage2_57500": _gate(os.path.join(run_dir, "accept_candidate_stage2_57500.json")),
    }
    summary = {
        "experiment": "main_generalization",
        "date": "2026-11-21",
        "goal": ("adapt the exact Stage1/2 overtake specialist to the archived ppo_s0 "
                 "24-NPC main environment while keeping Stage1/2 accepted"),
        "main_env": {"source": "runs/ppo_s0/config.json env_cfg",
                     "json": os.path.abspath(os.path.join(run_dir, "env_main_ppo_s0.json")),
                     "n_obstacles": 24, "npc_v_min": 5.5, "npc_v_max": 8.0,
                     "v_max": 16.0, "random_start": True},
        "seed_policy": {
            "main_demos": "56000..56059", "main_dagger": "57800..57819",
            "stage1_dagger": "57820..57827", "stage2_dagger": "57840..57851",
            "dev_main": "57300..57309 (stability 57310..57319)",
            "dev_stage1": "57400..57409", "dev_stage2": "57500..57509",
            "stage12_anchor_source": "53200..53319 (pre-existing sparse exact dumps)",
            "host_reserved_untouched": ["10000..10099", "68000..68099",
                                        "67000", "67100", "57000..57024"],
        },
        "training": {
            "init_checkpoint": cfg.get("init_checkpoint"),
            "init_sha256": cfg.get("init_checkpoint_sha256"),
            "bc_samples": cfg["behavior_cloning"]["bc_samples"],
            "bc_epochs": cfg["behavior_cloning"]["bc_epochs"],
            "bc_accuracy_end": cfg["behavior_cloning"]["bc_accuracy_end"],
            "bc_action_prob_end": cfg["behavior_cloning"]["bc_action_prob_end"],
            "parts": cfg["behavior_cloning_data"]["parts"],
            "v1_checkpoint": os.path.abspath(os.path.join(run_dir, "ckpt_bc.npz")),
            "v1_sha256": _sha256(os.path.join(run_dir, "ckpt_bc.npz")),
            "candidate_checkpoint": cfg["candidate_checkpoint"],
            "candidate_sha256": cfg["candidate_checkpoint_sha256"],
            "dagger_disagreement": {"main": "3687/9467 (38.9%)",
                                    "stage1": "712/4306 (16.5%)",
                                    "stage2": "2231/5436 (41.0%)"},
        },
        "results": results,
        "host_reserved_band_checks": {
            name: _gate(os.path.join(run_dir, name))
            for name in ("verified_stage1_67000.json", "verified_stage2_67100.json")
            if os.path.exists(os.path.join(run_dir, name))
        },
        "commands": [
            "./runtime/python/python.exe scripts/accept_overtake.py --expert "
            "--expert-block-distance 18 --env-json runs/main_generalization_bc/"
            "env_main_ppo_s0.json --episodes 10 --seed-base 57300 --json-out "
            "runs/main_generalization_bc/gate0_expert_main_57300_e18.json",
            "./runtime/python/python.exe scripts/main_generalization.py collect "
            "--episodes 60 --seed-base 56000",
            "./runtime/python/python.exe scripts/main_generalization.py train "
            "--init-checkpoint runs/robust_stage12_exact_recovery_bc_s2init/ckpt_final.npz "
            "--anchor-samples 4000 --epochs 300 --batch-size 128",
            "./runtime/python/python.exe scripts/main_generalization.py dagger "
            "--ckpt ckpt_bc.npz --label main --episodes 20 --seed-base 57800",
            "./runtime/python/python.exe scripts/main_generalization.py dagger "
            "--ckpt ckpt_bc.npz --label stage1 --env-json "
            "../robust_stage12_eval_stage1_env.json --episodes 8 --seed-base 57820",
            "./runtime/python/python.exe scripts/main_generalization.py dagger "
            "--ckpt ckpt_bc.npz --label stage2 --env-json "
            "../robust_stage12_eval_stage2_env.json --episodes 12 --seed-base 57840",
            "./runtime/python/python.exe scripts/main_generalization.py train "
            "--init-checkpoint runs/robust_stage12_exact_recovery_bc_s2init/ckpt_final.npz "
            "--anchor-samples 4000 --epochs 300 --batch-size 128 --dagger-demos "
            "demos_dagger_main.npz demos_dagger_stage1.npz demos_dagger_stage2.npz "
            "--out-ckpt ckpt_dagger.npz",
            "./runtime/python/python.exe scripts/main_generalization.py finalize "
            "--source ckpt_dagger.npz",
            "./runtime/python/python.exe scripts/accept_overtake.py --run "
            "runs/main_generalization_bc --ckpt ckpt_candidate.npz --env-json "
            "runs/main_generalization_bc/env_main_ppo_s0.json --episodes 10 "
            "--seed-base 57300 --json-out "
            "runs/main_generalization_bc/accept_candidate_main_57300.json",
            "./runtime/python/python.exe scripts/accept_overtake.py --run "
            "runs/main_generalization_bc --ckpt ckpt_candidate.npz --env-json "
            "runs/robust_stage12_eval_stage1_env.json --episodes 10 --seed-base 57400 "
            "--json-out runs/main_generalization_bc/accept_candidate_stage1_57400.json",
            "./runtime/python/python.exe scripts/accept_overtake.py --run "
            "runs/main_generalization_bc --ckpt ckpt_candidate.npz --env-json "
            "runs/robust_stage12_eval_stage2_env.json --episodes 10 --seed-base 57500 "
            "--json-out runs/main_generalization_bc/accept_candidate_stage2_57500.json",
        ],
        "bounded_gaps": [
            "main passes both 57300 and 57310 at the gate minimum (6 clean, 5 strict); "
            "the 4 non-clean episodes per block contain 3 collisions and 4 off-track "
            "episodes (the two sets overlap), so there is no margin on either gate",
            "Stage1/Stage2 dev blocks are 10/10 clean and 10/10 strict; the anchor was "
            "only 4000 of 54844 BC samples",
            "this session did not execute 10000..10099, 68000..68099, 67000 or 67100; "
            "the host's own verified_stage1_67000 / verified_stage2_67100 files were "
            "left untouched (both report 10/10 clean and 10/10 strict on ckpt_dagger.npz)",
            "57000..57024 was only used by the host report and was never executed here",
        ],
        "artifacts_note": ("all new files are under runs/main_generalization_bc; no core "
                           "source was modified; scripts/main_generalization.py is the new "
                           "reproducibility entry point"),
    }
    _json(os.path.join(run_dir, "summary.json"), summary)
    print(f"[finalize] candidate {dst} sha256 {cfg['candidate_checkpoint_sha256']}")
    for name in ("candidate_main_57300", "candidate_main_57310_stability",
                 "candidate_stage1_57400", "candidate_stage2_57500"):
        r = results[name]
        print(f"[finalize] {name}: clean {r['clean_laps']}/10 strict {r['strict_episodes']}/10 "
              f"passed {r['passed']}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="command", required=True)

    env_p = sub.add_parser("env", help="write the exact ppo_s0 main environment JSON")
    env_p.set_defaults(func=cmd_env)

    c = sub.add_parser("collect", help="collect memoryless-expert demos on the main env")
    c.add_argument("--episodes", type=int, default=60)
    c.add_argument("--seed-base", type=int, default=56_000)
    c.add_argument("--block-distance", type=float, default=BLOCK_DISTANCE)
    c.add_argument("--keep-clean-only", action="store_true", default=True)
    c.add_argument("--out", default="demos_main.npz")
    c.set_defaults(func=cmd_collect)

    t = sub.add_parser("train", help="BC from the specialist checkpoint")
    t.add_argument("--demos", default="demos_main.npz")
    t.add_argument("--main-seed-base", type=int, default=56_000)
    t.add_argument("--init-checkpoint", required=True)
    t.add_argument("--anchor", default=ANCHOR_SOURCE)
    t.add_argument("--anchor-samples", type=int, default=4000)
    t.add_argument("--anchor-seed", type=int, default=0)
    t.add_argument("--dagger-demos", nargs="*", default=[],
                   help="optional dagger .npz files inside --run to append")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--label-smoothing", type=float, default=0.02)
    t.add_argument("--out-ckpt", default=None)
    t.set_defaults(func=cmd_train)

    d = sub.add_parser("dagger", help="collect one DAgger correction pass")
    d.add_argument("--ckpt", default="ckpt_bc.npz")
    d.add_argument("--episodes", type=int, default=20)
    d.add_argument("--seed-base", type=int, default=57_800)
    d.add_argument("--max-steps", type=int, default=1200)
    d.add_argument("--block-distance", type=float, default=BLOCK_DISTANCE)
    d.add_argument("--env-json", default=None,
                   help="environment JSON; defaults to the main ppo_s0 env")
    d.add_argument("--label", default="main")
    d.add_argument("--out", default=None,
                   help="output .npz name; defaults to dagger_<label>.npz")
    d.set_defaults(func=cmd_dagger)

    f = sub.add_parser("finalize", help="copy the checkpoint and write summary.json")
    f.add_argument("--source", default="ckpt_dagger.npz")
    f.set_defaults(func=cmd_finalize)
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.run, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
