"""Strict, independent verification of an *overtake* (acceptance protocol).

Why this exists
---------------
Two weaker signals were being used as "an overtake happened":

* ``overtake_passes`` counted by the learner's own lane planner, and
* a lane change followed by "the blocker is now behind me".

Neither is a full description of the manoeuvre, and both are computed from the
policy's own state, so a policy bug can make them agree with each other.  This
module is a verifier that watches the **simulator**, not the agent, and accepts
a strict overtake only when every one of these holds:

1. **safe lane change** -- the ego crosses into an adjacent lane while its own
   lane is blocked (lead closer than ``block_distance``) and the destination is
   ``safe_to_enter`` at the moment of the crossing (front/rear gaps above their
   minima and no car alongside);
2. **blocking car passed** -- the NPC that blocked the source lane ends up more
   than ``pass_margin`` metres *behind* the ego;
3. **target lane keeping** -- the ego stays in the destination lane for at least
   ``min_hold_steps`` steps and does not leave it before the pass completes;
4. **no collision / no off-track** in the whole episode.

The verifier owns a :class:`autodrive.sensors.SensorFusion`, so it reads the same
interpretable state the policy sees, but it never reads the policy.  It is used
by :func:`autodrive.logging_utils.evaluate_agent` (for checkpoint selection and
reporting) and by ``scripts/accept_overtake.py`` (the acceptance gate).
"""
from __future__ import annotations

from dataclasses import dataclass

from .sensors import SensorFusion


@dataclass
class StrictOvertakeEvent:
    """One candidate manoeuvre, with the reason it was accepted or rejected."""

    episode: int
    seed: int
    start_step: int
    src_lane: int
    dst_lane: int
    blocker_id: int | None
    passed_step: int | None = None
    hold_steps: int = 0
    valid: bool = False
    reason: str = "pending"

    def as_dict(self) -> dict:
        return {
            "episode": int(self.episode),
            "seed": int(self.seed),
            "start_step": int(self.start_step),
            "src_lane": int(self.src_lane),
            "dst_lane": int(self.dst_lane),
            "blocker_id": None if self.blocker_id is None else int(self.blocker_id),
            "passed_step": None if self.passed_step is None else int(self.passed_step),
            "hold_steps": int(self.hold_steps),
            "valid": bool(self.valid),
            "reason": str(self.reason),
        }


DEFAULT_CONFIG = {
    "block_distance": 18.0,     # [m] source lane must be blocked for this to be an overtake
    "pass_margin": 3.0,         # [m] blocker must be this far behind the ego
    "min_hold_steps": 3,        # steps in the destination lane before the pass counts
    "max_maneuver_steps": 250,  # abandon a manoeuvre that never completes
    "lane_advantage_margin": 5.0,
    "min_side_gap": 7.0,
    "range": 40.0,
}


class StrictOvertakeMonitor:
    """Watch a rollout and certify strict overtakes.

    Usage per control step::

        monitor.before_step(env)
        obs, reward, term, trunc, info = env.step(action)
        monitor.after_step(env, info)

    and ``monitor.finish_episode(info)`` at the end.
    """

    def __init__(self, cfg=None):
        c = dict(DEFAULT_CONFIG)
        c.update(cfg or {})
        self.cfg = c
        self.block_distance = float(c["block_distance"])
        self.pass_margin = float(c["pass_margin"])
        self.min_hold_steps = int(c["min_hold_steps"])
        self.max_maneuver_steps = int(c["max_maneuver_steps"])
        self.sensor = SensorFusion({
            "range": float(c["range"]),
            "block_distance": self.block_distance,
            "lane_advantage_margin": float(c["lane_advantage_margin"]),
            "min_side_gap": float(c["min_side_gap"]),
        })
        self.events: list[StrictOvertakeEvent] = []
        self._pre = None
        self._active: StrictOvertakeEvent | None = None
        self._blocked_changes = 0
        self._rejected = 0
        self._episode = 0
        self._seed = 0
        self._dirty = False
        self._episode_events: list[StrictOvertakeEvent] = []

    # ------------------------------------------------------------------ episode
    def reset_episode(self, env, episode: int = 0, seed: int = 0) -> None:
        self._episode = int(episode)
        self._seed = int(seed)
        self._active = None
        self._pre = None
        self._dirty = False
        self._episode_events = []
        self._blocked_changes = 0
        self._rejected = 0

    # --------------------------------------------------------------------- step
    def before_step(self, env) -> None:
        """Snapshot the state the action was chosen in."""
        self._pre = self.sensor.scan(env)

    def _close(self, event: StrictOvertakeEvent, reason: str, valid: bool) -> None:
        event.reason = reason
        event.valid = bool(valid)
        self._episode_events.append(event)
        if not valid:
            self._rejected += 1

    def after_step(self, env, info=None) -> None:
        info = info or {}
        if info.get("crash") or info.get("collision"):
            self._dirty = True
        pre = self._pre
        if pre is None:
            return
        post = self.sensor.scan(env)
        lane_pre, lane_post = int(pre.ego_lane), int(post.ego_lane)

        # ---- advance / close the active manoeuvre --------------------------
        active = self._active
        if active is not None:
            offset = None
            if active.blocker_id is not None:
                offset = post.npc_offsets.get(int(active.blocker_id))
            active.hold_steps = int(env.steps) - int(active.start_step)
            if offset is not None and offset < -self.pass_margin:
                # the blocker is behind the ego: the pass happened
                active.passed_step = int(env.steps)
                if active.hold_steps >= self.min_hold_steps and lane_post == active.dst_lane:
                    self._close(active, "passed", True)
                elif lane_post != active.dst_lane:
                    self._close(active, "left_target_lane", False)
                else:
                    self._close(active, "target_lane_hold_too_short", False)
                self._active = None
            elif lane_post != active.dst_lane:
                self._close(active, "left_target_lane", False)
                self._active = None
            elif active.hold_steps > self.max_maneuver_steps:
                self._close(active, "timeout", False)
                self._active = None

        # ---- start a manoeuvre on a lane change ----------------------------
        if lane_post != lane_pre and self._active is None:
            src, dst = lane_pre, lane_post
            adjacent = abs(dst - src) == 1
            src_reading = pre.reading(src)
            dst_reading = pre.reading(dst)
            src_blocked = bool(src_reading.exists
                               and src_reading.front_gap < self.block_distance)
            dst_safe = bool(dst_reading.exists and dst_reading.safe_to_enter)
            blocker = int(pre.front_ids[src]) if 0 <= src < int(len(pre.front_ids)) else -1
            if adjacent and src_blocked and dst_safe and blocker >= 0:
                self._blocked_changes += 1
                self._active = StrictOvertakeEvent(
                    episode=self._episode, seed=self._seed,
                    start_step=int(env.steps), src_lane=int(src), dst_lane=int(dst),
                    blocker_id=int(blocker))
            elif adjacent and src_blocked:
                # a change made while blocked but into a lane that was not safe
                self._blocked_changes += 1
                self._rejected += 1
                self._episode_events.append(StrictOvertakeEvent(
                    episode=self._episode, seed=self._seed,
                    start_step=int(env.steps), src_lane=int(src), dst_lane=int(dst),
                    blocker_id=int(blocker) if blocker >= 0 else None,
                    reason="unsafe_destination", valid=False))

    def finish_episode(self, info=None) -> dict:
        """Close the episode and return its certified result."""
        if self._active is not None:
            self._close(self._active, "episode_end_without_pass", False)
            self._active = None
        valid = [e for e in self._episode_events if e.valid]
        for e in valid:
            self.events.append(e)
        result = {
            "episode": int(self._episode),
            "seed": int(self._seed),
            "clean": not self._dirty,
            "blocked_lane_changes": int(self._blocked_changes),
            "strict_events": len(valid),
            "strict_safe_overtake": bool(valid) and not self._dirty,
            "rejected": int(self._rejected),
            "events": [e.as_dict() for e in self._episode_events],
        }
        return result

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        return {
            "events": len(self.events),
            "events_detail": [e.as_dict() for e in self.events],
            "rejected": int(self._rejected),
            "blocked_lane_changes": int(self._blocked_changes),
        }


def gate(clean_laps: int, strict_overtakes: int, episodes: int,
         min_clean_laps: int, min_strict_overtakes: int) -> dict:
    """Evaluate the two-part acceptance gate and report both parts honestly."""
    clean_ok = int(clean_laps) >= int(min_clean_laps)
    strict_ok = int(strict_overtakes) >= int(min_strict_overtakes)
    return {
        "episodes": int(episodes),
        "clean_laps": int(clean_laps),
        "min_clean_laps": int(min_clean_laps),
        "clean_lap_rate": float(clean_laps) / max(int(episodes), 1),
        "clean_lap_pass": bool(clean_ok),
        "strict_safe_overtakes": int(strict_overtakes),
        "min_strict_safe_overtakes": int(min_strict_overtakes),
        "strict_safe_overtake_rate": float(strict_overtakes) / max(int(episodes), 1),
        "strict_safe_overtake_pass": bool(strict_ok),
        "passed": bool(clean_ok and strict_ok),
    }
