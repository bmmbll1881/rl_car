"""CSV logging + evaluation protocol shared by every algorithm.

Design notes
------------
* One directory per run (``runs/<run_id>/``) holding ``config.json``,
  ``episodes.csv`` (training), ``evals.csv`` (greedy evaluation) and the
  checkpoints -- so a run is self-describing and can be diffed against another.
* Evaluation always uses a *fixed* set of seeds and the greedy policy, so
  numbers from different runs/algorithms are comparable.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

from .acceptance import StrictOvertakeMonitor
from .env import MiniDriveEnv


class CSVLogger:
    def __init__(self, path: str, fieldnames):
        self.path = path
        self.fieldnames = list(fieldnames)
        self._rows = 0
        new = not os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames, extrasaction="ignore")
        if new:
            self._writer.writeheader()

    def log(self, **row):
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._rows += 1
        if self._rows % 10 == 0:
            self._fh.flush()

    def close(self):
        self._fh.flush()
        self._fh.close()


def reset_agent(agent, obs) -> None:
    """Tell a stateful agent that a new episode starts (no-op for stateless ones)."""
    fn = getattr(agent, "reset_episode", None)
    if callable(fn):
        fn(obs)


def advance_agent(agent, next_obs, terminated: bool = False, truncated: bool = False) -> None:
    """Let a stateful agent absorb one transition during a non-learning rollout."""
    fn = getattr(agent, "observe_transition", None)
    if callable(fn):
        fn(next_obs, terminated, truncated)


def policy_lane_snapshot(agent) -> dict:
    """The lane planner view that was in force *before* the last action."""
    lane = getattr(agent, "_lane_taken", None)
    if lane:
        return dict(lane)
    return dict(getattr(agent, "last_lane", None) or {})


class OvertakeTracker:
    """Overtake statistics over one or more rollouts.

    Everything is counted against the policy's *own* lane planner
    (``lane_aware``), so "opportunity" means "the agent saw a safe adjacent lane
    with more room", not a purely geometric gap.  A rollout calls
    :meth:`start_episode` at every reset and :meth:`observe` after every step.
    """

    COUNTERS = ("steps", "opportunity_steps", "toward", "hold", "brake", "other",
                "lane_changes", "lane_changes_toward", "bonus_events",
                "window_steps", "passes")

    def __init__(self):
        for k in self.COUNTERS:
            setattr(self, k, 0)
        self._prev_lane = None
        self._passed_prev = False
        self._episode_start = {k: 0 for k in self.COUNTERS}

    def start_episode(self, env) -> None:
        self._prev_lane = int(env.lane_of())
        self._passed_prev = False
        self._episode_start = {k: getattr(self, k) for k in self.COUNTERS}

    def observe(self, env, action, lane=None, bonus: float = 0.0) -> None:
        """Record one step.

        ``bonus`` must be the reward the agent actually paid for a *verified*
        lane change on this step, so ``bonus_events`` is a completion count, not
        a "the policy steered" count.
        """
        lane = lane or {}
        action = int(action)
        self.steps += 1
        direction = int(lane.get("direction", 0))
        if lane.get("overtake_available"):
            self.opportunity_steps += 1
            if action in (0, 1, 3, 4):
                steer_dir = 1 if action in (0, 1) else -1
                if steer_dir == direction:
                    self.toward += 1
                else:
                    self.other += 1
            elif action in (2, 5):
                self.hold += 1
            elif action == 6:
                self.brake += 1
            else:
                self.other += 1

        prev = lane.get("ego_lane", self._prev_lane)
        current = int(env.lane_of())
        if prev is not None and current != int(prev):
            self.lane_changes += 1
            moved = 1 if current > int(prev) else -1
            if direction != 0 and moved == direction:
                self.lane_changes_toward += 1
        if float(bonus) > 0.0:
            self.bonus_events += 1
        if lane.get("overtake_window"):
            self.window_steps += 1
        passed = bool(lane.get("overtake_passed"))
        if passed and not self._passed_prev:
            self.passes += 1
        self._passed_prev = passed
        self._prev_lane = current

    @staticmethod
    def _ratios(counts: dict) -> dict:
        opp = max(int(counts["opportunity_steps"]), 1)
        return {
            "overtake_opportunity_steps": int(counts["opportunity_steps"]),
            "overtake_toward_frac": counts["toward"] / opp,
            "overtake_hold_frac": counts["hold"] / opp,
            "overtake_brake_frac": counts["brake"] / opp,
            "overtake_other_frac": counts["other"] / opp,
            "lane_changes": int(counts["lane_changes"]),
            "lane_changes_toward": int(counts["lane_changes_toward"]),
            "lane_bonus_events": int(counts["bonus_events"]),
            "overtake_window_steps": int(counts["window_steps"]),
            "overtake_passes": int(counts["passes"]),
        }

    def episode_summary(self) -> dict:
        counts = {k: getattr(self, k) - self._episode_start[k] for k in self.COUNTERS}
        return self._ratios(counts)

    def summary(self) -> dict:
        out = self._ratios({k: getattr(self, k) for k in self.COUNTERS})
        out["steps"] = int(self.steps)
        return out


def evaluate_agent(env: MiniDriveEnv, agent, episodes: int = 5, base_seed: int = 10_000,
                   on_step=None, block_distance: float = 18.0,
                   pass_margin: float = 3.0, strict_hold_steps: int = 3) -> dict:
    """Run ``episodes`` greedy episodes on fixed seeds and aggregate metrics.

    Besides the environment return, every episode is scanned by
    :class:`autodrive.acceptance.StrictOvertakeMonitor`, which certifies a strict
    overtake from the simulator state alone (safe lane change, blocking car
    passed, target-lane keeping, no collision, no off-track).  Those fields are
    what :func:`selection_key` ranks checkpoints on.
    """
    rows = []
    tracker = OvertakeTracker()
    monitor = StrictOvertakeMonitor({
        "block_distance": float(block_distance),
        "pass_margin": float(pass_margin),
        "min_hold_steps": int(strict_hold_steps),
    })
    for i in range(episodes):
        seed = base_seed + i
        obs, info = env.reset(seed=seed)
        reset_agent(agent, obs)
        tracker.start_episode(env)
        monitor.reset_episode(env, episode=i, seed=seed)
        ret, done, steps, speeds, lats = 0.0, False, 0, [], []
        min_npc = float("inf")
        while not done:
            monitor.before_step(env)
            a = agent.act(obs, greedy=True)
            lane = policy_lane_snapshot(agent)
            obs, r, term, trunc, info = env.step(a)
            monitor.after_step(env, info)
            ret += r
            steps += 1
            speeds.append(info["speed"])
            lats.append(abs(info["lateral"]))
            min_npc = min(min_npc, info.get("npc_min_distance", float("inf")))
            # settle the agent first: it sets ``_lane_bonus_taken`` for the
            # verified lane change this step, which the tracker must see
            advance_agent(agent, obs, term, trunc)
            bonus = float(getattr(agent, "_lane_bonus_taken", 0.0))
            tracker.observe(env, a, lane=lane, bonus=bonus)
            if on_step is not None:
                on_step(env, a, info)
            done = term or trunc
        overtake = tracker.episode_summary()
        clean = not info["crash"] and not info.get("collision", False)
        passes = int(overtake["overtake_passes"])
        strict = monitor.finish_episode(info)
        rows.append(
            {
                "seed": seed,
                "return": ret,
                "steps": steps,
                "success": int(info["success"]),
                "crash": int(info["crash"]),
                "collision": int(info.get("collision", False)),
                "distance": info["progress"],
                "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                "lateral_rms": float(np.sqrt(np.mean(np.square(lats)))) if lats else 0.0,
                "min_npc_distance": min_npc if np.isfinite(min_npc) else float("nan"),
                "lap_time_s": steps * env.dt,
                "overtake": overtake,
                # A safe pass is an actual blocker pass (not a lane change) in an
                # episode that never collided and never left the road.  The
                # event count is zeroed for a dirty episode so a pass followed
                # by a crash can never be reported as safe.
                "clean": bool(clean),
                "clean_lap": bool(int(info["success"]) and clean),
                "safe_pass": bool(passes > 0 and clean),
                "safe_pass_events": passes if clean else 0,
                # strict certification, computed from the simulator only
                "strict_events": int(strict["strict_events"]),
                "strict_safe_overtake": bool(strict["strict_safe_overtake"]),
                "strict_events_detail": strict["events"],
                "strict_blocked_lane_changes": int(strict["blocked_lane_changes"]),
            }
        )
    lap_times = [r["lap_time_s"] for r in rows if r["success"]]
    overtake = tracker.summary()
    agg = {
        "eval_return_mean": float(np.mean([r["return"] for r in rows])),
        "eval_return_min": float(np.min([r["return"] for r in rows])),
        "eval_success_rate": float(np.mean([r["success"] for r in rows])),
        "eval_crash_rate": float(np.mean([r["crash"] for r in rows])),
        "eval_collision_rate": float(np.mean([r["collision"] for r in rows])),
        "eval_distance_mean": float(np.mean([r["distance"] for r in rows])),
        "eval_speed_mean": float(np.mean([r["mean_speed"] for r in rows])),
        "eval_lateral_rms": float(np.mean([r["lateral_rms"] for r in rows])),
        "eval_min_npc_distance": float(np.nanmean([r["min_npc_distance"] for r in rows]))
        if rows and not np.all(np.isnan([r["min_npc_distance"] for r in rows])) else float("nan"),
        "eval_lap_time_mean": float(np.mean(lap_times)) if lap_times else float("nan"),
        "eval_overtake_opportunity_steps": overtake["overtake_opportunity_steps"],
        "eval_overtake_toward_frac": overtake["overtake_toward_frac"],
        "eval_overtake_hold_frac": overtake["overtake_hold_frac"],
        "eval_overtake_brake_frac": overtake["overtake_brake_frac"],
        "eval_lane_changes": overtake["lane_changes"],
        "eval_lane_changes_toward": overtake["lane_changes_toward"],
        "eval_lane_bonus_events": overtake["lane_bonus_events"],
        "eval_overtake_window_steps": overtake["overtake_window_steps"],
        "eval_overtake_passes": overtake["overtake_passes"],
        # safe-pass accounting: passes in clean episodes only, plus the episode
        # count the acceptance criterion is written against
        "eval_clean_episodes": int(sum(1 for r in rows if r["clean"])),
        "eval_clean_laps": int(sum(1 for r in rows if r["clean_lap"])),
        "eval_clean_lap_rate": float(np.mean([r["clean_lap"] for r in rows])) if rows else 0.0,
        "eval_safe_pass_episodes": int(sum(1 for r in rows if r["safe_pass"])),
        "eval_safe_pass_events": int(sum(int(r["safe_pass_events"]) for r in rows)),
        "eval_safe_pass_rate": float(np.mean([r["safe_pass"] for r in rows])) if rows else 0.0,
        # strict certification (simulator-only verifier)
        "eval_strict_safe_overtake_episodes":
            int(sum(1 for r in rows if r["strict_safe_overtake"])),
        "eval_strict_safe_overtake_events": int(sum(int(r["strict_events"]) for r in rows)),
        "eval_strict_safe_overtake_rate":
            float(np.mean([r["strict_safe_overtake"] for r in rows])) if rows else 0.0,
    }
    agg["_rows"] = rows
    return agg


def selection_key(result: dict) -> tuple:
    """Rank checkpoints by what the task asks for, **not** by return.

    Priority: clean laps, then certified strict overtakes, then collision-free
    episodes, then lap-success rate and distance.  A policy that stands still
    can have a high (less negative) return on a truncating episode, which is why
    return is deliberately not part of this key.

    ``result`` is either a dict returned by :func:`evaluate_agent` (it then uses
    its ``_rows``) or a dict that already carries the aggregate fields.
    """
    rows = result.get("_rows") or []
    if rows:
        n = max(len(rows), 1)
        clean_laps = sum(1 for r in rows if r.get("clean_lap"))
        strict = sum(1 for r in rows if r.get("strict_safe_overtake"))
        clean = sum(1 for r in rows if r.get("clean"))
        success = sum(int(r.get("success", 0)) for r in rows)
        distance = float(np.mean([float(r.get("distance", 0.0)) for r in rows]))
        return (int(clean_laps), int(strict), int(clean), success / n, distance)
    return (int(result.get("eval_clean_laps", 0)),
            int(result.get("eval_strict_safe_overtake_episodes", 0)),
            int(result.get("eval_clean_episodes", 0)),
            float(result.get("eval_success_rate", 0.0)),
            float(result.get("eval_distance_mean", 0.0)))


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_config(path: str, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
