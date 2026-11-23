#!/usr/bin/env python
"""Collect scripted-expert demonstrations for behaviour cloning.

The expert is ``MemorylessExpertDriver`` by default: every action is a pure
function of the current simulator state, so the labels are Markov in the policy
input (``--legacy-expert`` selects the old stateful ``PurePursuitDriver``).  The
observations written are the *learner's own* policy input
(``RARL._features`` when the algorithm has it), so the file can be fed straight
to ``PPO.pretrain`` / ``RARL.pretrain``.  The safety mask that was in force at
each step is written too (``masks``), and ``--keep-clean-only`` drops episodes
where the expert crashed so the data describes the target behaviour only.

Examples
--------
    python scripts/collect_bc_demos.py --algo rarl \
        --config configs/rarl_bc_lowlr.json \
        --env-json configs/env_overtake_stage1.json \
        --episodes 100 --keep-clean-only --out runs/bc_demos.npz

    python scripts/collect_bc_demos.py --algo ppo --episodes 2 --out runs/bc_ppo.npz

``--require-overtake`` exits non-zero when the expert completed no overtake, so a
caller does not accidentally treat ordinary lane-keeping data as overtake data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from autodrive.algos import REGISTRY, make_algorithm  # noqa: E402
from autodrive.demonstrations import collect_expert_demonstrations  # noqa: E402
from autodrive.env import MiniDriveEnv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Collect expert demonstrations for BC")
    p.add_argument("--algo", default="rarl", choices=sorted(REGISTRY))
    p.add_argument("--config", default=None, help="algorithm hyper-parameter JSON")
    p.add_argument("--env-json", default=None, help="environment override JSON")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=30_000)
    p.add_argument("--max-steps", type=int, default=None, help="per-episode step cap")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "bc_demos.npz"))
    p.add_argument("--require-overtake", action="store_true",
                   help="fail if the expert completed no overtake")
    p.add_argument("--legacy-expert", action="store_true",
                   help="use the stateful PurePursuitDriver instead of the default "
                        "memoryless expert (labels are then not Markov in the input)")
    p.add_argument("--keep-clean-only", action="store_true",
                   help="drop samples from episodes where the expert crashed or collided")
    args = p.parse_args()

    env_over = {}
    if args.env_json:
        with open(args.env_json, encoding="utf-8") as fh:
            env_over = json.load(fh)
    env = MiniDriveEnv(seed=0, **env_over)

    algo_cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            algo_cfg = json.load(fh)
    agent = make_algorithm(args.algo, env.obs_dim, env.n_actions, cfg=algo_cfg, seed=0)
    if hasattr(agent, "bind_env"):
        agent.bind_env(env)

    obs, actions, stats = collect_expert_demonstrations(
        env, agent, args.episodes, seed_base=args.seed_base, max_steps=args.max_steps,
        memoryless=not args.legacy_expert, keep_clean_only=args.keep_clean_only)
    masks = stats.pop("masks")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, observations=obs, actions=actions, masks=masks,
                        stats=np.array(json.dumps(stats)))
    print(f"[demos] {obs.shape[0]} samples x {obs.shape[1]} features -> {args.out}")
    print(f"[demos] expert {stats['expert']} | mask used on "
          f"{stats['masked_samples']} samples, {stats['mask_conflicts']} label conflicts")
    print(json.dumps({k: v for k, v in stats.items() if k != "episode_stats"}, indent=2))
    for ep in stats["episode_stats"]:
        print(f"  ep {ep['episode']} seed {ep['seed']}: steps {ep['steps']} "
              f"lane changes {ep['lane_changes']} "
              f"(blocked {ep['blocked_lane_changes']}) "
              f"overtakes {ep['overtakes_completed']} "
              f"success {ep['success']} crash {ep['crash']} collision {ep['collision']} "
              f"kept {ep['kept']}")

    if args.require_overtake and stats["overtakes_completed"] == 0:
        print("[demos] ERROR: the expert completed no overtake; refusing to label "
              "this data as an overtake demonstration", file=sys.stderr)
        return 2
    if stats["overtakes_completed"] == 0:
        print("[demos] WARNING: no overtake was demonstrated in these episodes",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
