#!/usr/bin/env python
"""Diagnose the RARL risk/oversight stack on held-out seeds.

Reports, for the greedy policy:
  * how often the TTC shield would fire and how often it overrides the raw
    greedy action;
  * how often an overtake opportunity is present and what the policy does in
    it (steer towards the target lane / hold the lane / brake);
  * lane changes, completed lane changes in the planned direction, and the
    episode-level success / collision rates.

Examples
--------
    python scripts/diagnose_risk.py --run runs/rarl_overtake_smoke
    python scripts/diagnose_risk.py --run runs/rarl_overtake_smoke \
        --config configs/rarl_overtake.json --episodes 10 --seed-base 20000

``--config`` defaults to the algorithm config stored in the run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from autodrive.algos import make_algorithm
from autodrive.env import MiniDriveEnv
from autodrive.logging_utils import (OvertakeTracker, advance_agent, policy_lane_snapshot,
                                     reset_agent)


def _raw_greedy(agent) -> int | None:
    """Un-masked argmax of the policy (what the shield would have to override)."""
    obs = getattr(agent, "_obs", None)
    if obs is None or not hasattr(agent, "policy"):
        return None
    return int(np.argmax(agent.policy.actor.forward(obs)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="ckpt_best.npz")
    p.add_argument("--config", default=None,
                   help="algorithm config JSON (default: the one stored in the run)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=20_000)
    args = p.parse_args()

    with open(os.path.join(args.run, "config.json"), encoding="utf-8") as fh:
        run_cfg = json.load(fh)
    algo_cfg = run_cfg.get("algorithm_cfg") or {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            algo_cfg = json.load(fh)
    env = MiniDriveEnv(seed=0, **run_cfg["env_cfg"])
    agent = make_algorithm(run_cfg["algo"], env.obs_dim, env.n_actions,
                           cfg=algo_cfg, seed=run_cfg["seed"])
    if hasattr(agent, "bind_env"):
        agent.bind_env(env)
    agent.load_state_dict(dict(np.load(os.path.join(args.run, args.ckpt))))

    cfg = getattr(agent, "cfg", {})
    threshold = float(cfg.get("shield_hazard_threshold", 0.5))
    mask_threshold = float(cfg.get("action_risk_mask_threshold", 0.25))
    gate = getattr(agent, "use_shield", False) or getattr(agent, "use_overtake_lock", False)

    tracker = OvertakeTracker()
    front = rear = overrides = masked_steps = 0
    action_counts = np.zeros(env.n_actions, dtype=int)
    risk_sum = chosen_risk_sum = 0.0
    max_front = max_rear = 0.0
    successes = collisions = finished = 0
    # did a *safe* overtake actually happen?  A pass counts only when the
    # blocker that forced the lane change is behind the ego, and "safe" requires
    # that episode to end without a collision.
    pass_episodes = safe_pass_episodes = 0
    # catch-up evidence: is the ego actually closing on a blocked lane?
    block_distance = float(cfg.get("lane_block_distance", 18.0))
    speed_sum = blocked = closing = 0
    lead_gap_sum = 0.0
    blocked_episodes = opportunity_episodes = 0

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed_base + episode)
        reset_agent(agent, obs)
        tracker.start_episode(env)
        opp0 = tracker.opportunity_steps
        passes0 = tracker.passes
        ep_blocked = False
        done = False
        while not done:
            lead_gap, lead_speed = env.obstacles.lane_lead(env.lane_of(), env.prev_s)
            if lead_gap < block_distance:
                blocked += 1
                lead_gap_sum += float(lead_gap)
                ep_blocked = True
                if env.v > lead_speed + 0.5:
                    closing += 1
            action = agent.act(obs, greedy=True)
            lane = policy_lane_snapshot(agent)
            # read the mask the policy actually used; never recompute it here,
            # because _action_mask() also advances the overtake cooldown
            mask = np.asarray(getattr(agent.policy, "_last_mask",
                                      np.ones(env.n_actions, dtype=bool)), dtype=bool)
            raw = _raw_greedy(agent)
            if raw is not None and gate:
                overrides += int(not bool(mask[raw]))
            masked_steps += int(np.sum(mask) < len(mask))

            f = float(getattr(agent, "last_risk", {}).get("ttc_front_hazard", 0.0))
            r = float(getattr(agent, "last_risk", {}).get("ttc_rear_hazard", 0.0))
            max_front, max_rear = max(max_front, f), max(max_rear, r)
            front += int(f >= threshold)
            rear += int(r >= threshold)

            risk = np.asarray(getattr(agent, "last_action_risk", np.zeros(env.n_actions)))
            if risk.size == env.n_actions:
                risk_sum += float(np.max(risk))
                chosen_risk_sum += float(risk[action])
            action_counts[action] += 1

            obs, _, term, trunc, info = env.step(action)
            speed_sum += float(info["speed"])
            # settle the agent first: it sets ``_lane_bonus_taken`` for the
            # verified lane change this step, which the tracker must see
            advance_agent(agent, obs, term, trunc)
            tracker.observe(env, action, lane=lane,
                            bonus=float(getattr(agent, "_lane_bonus_taken", 0.0)))
            done = term or trunc
        successes += int(info["success"])
        collisions += int(info["collision"])
        blocked_episodes += int(ep_blocked)
        opportunity_episodes += int(tracker.opportunity_steps > opp0)
        ep_passes = int(tracker.passes - passes0)
        if ep_passes > 0:
            pass_episodes += 1
            if not info["collision"] and not info["crash"]:
                safe_pass_episodes += 1
        finished += 1

    summary = tracker.summary()
    steps = max(summary["steps"], 1)
    print(json.dumps({
        "run": args.run,
        "episodes": finished,
        "success_rate": successes / max(finished, 1),
        "collision_rate": collisions / max(finished, 1),
        "steps": summary["steps"],
        "mean_speed": round(speed_sum / steps, 3),
        "blocked_steps": blocked,
        "blocked_frac": round(blocked / steps, 4),
        "closing_steps": closing,
        "closing_frac_of_blocked": round(closing / max(blocked, 1), 4),
        "mean_lead_gap_when_blocked": round(lead_gap_sum / max(blocked, 1), 2),
        "episodes_blocked": blocked_episodes,
        "episodes_with_opportunity": opportunity_episodes,
        "overtake_opportunity_steps": summary["overtake_opportunity_steps"],
        "overtake_toward_frac": round(summary["overtake_toward_frac"], 4),
        "overtake_hold_frac": round(summary["overtake_hold_frac"], 4),
        "overtake_brake_frac": round(summary["overtake_brake_frac"], 4),
        "overtake_other_frac": round(summary["overtake_other_frac"], 4),
        "lane_changes": summary["lane_changes"],
        "lane_changes_toward": summary["lane_changes_toward"],
        "lane_bonus_events": summary["lane_bonus_events"],
        "overtake_window_steps": summary["overtake_window_steps"],
        "overtake_passes": summary["overtake_passes"],
        "episodes_with_pass": pass_episodes,
        "episodes_with_safe_pass": safe_pass_episodes,
        "safe_overtake_observed": bool(safe_pass_episodes > 0),
        "front_hazard_steps": front,
        "rear_hazard_steps": rear,
        "greedy_override_steps": overrides,
        "greedy_override_frac": round(overrides / steps, 4),
        "mask_limited_steps": masked_steps,
        "mask_limited_frac": round(masked_steps / steps, 4),
        "action_risk_mask_threshold": mask_threshold,
        "mean_max_action_risk": round(risk_sum / steps, 4),
        "mean_chosen_action_risk": round(chosen_risk_sum / steps, 4),
        "actions": action_counts.tolist(),
        "max_front_hazard": round(max_front, 4),
        "max_rear_hazard": round(max_rear, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
