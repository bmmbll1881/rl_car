"""Pure-pursuit reference driver with lane selection.

Used for two purposes:
  1. environment sanity check -- if this controller cannot complete a lap, the
     environment itself is mis-parameterised (too tight, too slow, too crowded);
  2. an upper reference for the learning curves.

The driver steers at a look-ahead point, plans its speed from the curvature ahead
(comfort lateral acceleration, deliberately below the car's 8 m/s^2 limit) and
picks a **lane**: it stays in its lane while the lane is clear, otherwise it
switches to the lane with the largest free distance.  A learned policy can
legitimately beat it, which is what the experiments show.

:class:`MemorylessExpertDriver` implements the same policy without the three
hidden variables (latched lane, slewed lateral offset, cooldown), so its action
is a function of the current state.  It is the default demonstration source:
behaviour cloning cannot reproduce a label that depends on history the learner
cannot observe.
"""
from __future__ import annotations

import numpy as np

from .sensors import SensorFusion
from .track import wrap_angle


def _smoothed_curvature(track) -> np.ndarray:
    """Absolute curvature smoothed over a 9-point window (raw differences are noisy at spline joints)."""
    k = np.abs(track.curvature())
    ker = np.ones(9) / 9.0
    return np.convolve(np.concatenate([k, k[:8]]), ker, mode="same")[: track.n_points]


def _steer_command(env, offset: float, lookahead: float) -> tuple[int, float]:
    """Discrete steering action towards the look-ahead point, plus its raw angle."""
    track = env.track
    s = env.prev_s
    target_s = (s + lookahead) % track.length
    j = min(int(np.searchsorted(track.s, target_s)), track.n_points - 1)
    tx, ty = track.points[j] + track.normals[j] * offset
    alpha = wrap_angle(np.arctan2(ty - env.y, tx - env.x) - env.theta)
    dist = max(np.hypot(tx - env.x, ty - env.y), 1e-6)
    delta = np.arctan2(2.0 * env.cfg["wheelbase"] * np.sin(alpha), dist)
    table = np.array([0.45, 0.22, 0.0, -0.22, -0.45]) * env.cfg["steer_max"]
    return int(np.argmin(np.abs(table - delta))), float(delta)


def _speed_action(driver, gap: float, v_lead: float, delta: float,
                  offset: float, k_steer: int) -> int:
    """Brake / accelerate / keep steering, from curvature-ahead speed planning."""
    env = driver.env
    ahead = (env.idx + np.arange(0, 14)) % driver.track.n_points
    k_plan = max(float(driver.kappa[ahead].max()), 1e-3)
    v_safe = min(driver.v_target, np.sqrt(driver.a_lat_max / k_plan))
    if gap < driver.brake_range:
        return 6                                        # brake: too close
    if gap < driver.look_range:
        v_safe = min(v_safe, v_lead + 1.2 + 0.35 * (gap - driver.brake_range))
    if env.v > v_safe * 1.02:
        return 6                                        # brake
    # the accelerate action cannot steer, so only use it when already aligned
    if (env.v < v_safe * 0.92 and abs(delta) < 0.08 * env.cfg["steer_max"]
            and abs(float(env._lateral) - offset) < 1.2):
        return 5
    return k_steer


class PurePursuitDriver:
    """Steer at a look-ahead point, plan speed from curvature, change lanes for traffic."""

    name = "expert"

    def __init__(self, env, lookahead: float = 10.0, a_lat_max: float = 5.0, v_target: float = 11.0,
                 look_range: float = 34.0, brake_range: float = 12.0, lane_switch_penalty: float = 8.0,
                 side_clearance: float = 14.0, slew: float = 0.45):
        self.env = env
        self.track = env.track
        self.lookahead = lookahead
        self.a_lat_max = a_lat_max
        self.v_target = v_target
        self.look_range = look_range
        self.brake_range = brake_range
        self.lane_switch_penalty = lane_switch_penalty
        self.side_clearance = side_clearance
        self.slew = slew
        self.kappa = _smoothed_curvature(self.track)
        self.target_lane = None
        self.current_offset = None
        self._cooldown = 0

    def reset(self):
        self.target_lane = None
        self.current_offset = None
        self._cooldown = 0

    # ------------------------------------------------------------------ traffic
    def _lane_is_clear_beside(self, lane: int) -> bool:
        """No vehicle in ``lane`` within +-side_clearance metres of the ego."""
        ob = self.env.obstacles
        if not getattr(ob, "n", 0):
            return True
        idx = np.where(ob.lane == lane)[0]
        if idx.size == 0:
            return True
        gaps = (ob.s[idx] - self.env.prev_s) % self.track.length
        gaps = np.minimum(gaps, self.track.length - gaps)
        return bool(np.min(gaps) > self.side_clearance)

    def _choose_lane(self):
        """Keep the current lane unless it is blocked; otherwise take the best free lane."""
        env = self.env
        ob = env.obstacles
        cur = env.lane_of()
        if self.target_lane is None:
            self.target_lane = cur
        if not getattr(ob, "n", 0):
            return cur, (self.look_range * 2, float("inf"))
        if self._cooldown > 0:
            self._cooldown -= 1
            gap, v = ob.lane_lead(self.target_lane, env.prev_s)
            return self.target_lane, (gap, v)

        gap_cur, v_cur = ob.lane_lead(cur, env.prev_s)
        if gap_cur > self.look_range:
            self.target_lane = cur
            return cur, (gap_cur, v_cur)

        best_lane, best_score, best_info = cur, gap_cur, (gap_cur, v_cur)
        for lane in range(env.n_lanes):
            if lane == cur or not self._lane_is_clear_beside(lane):
                continue                       # do not cut into a car that is beside me
            gap, v = ob.lane_lead(lane, env.prev_s)
            score = gap - self.lane_switch_penalty * abs(lane - cur)
            if score > best_score:
                best_lane, best_score, best_info = lane, score, (gap, v)
        if best_lane != self.target_lane:
            self._cooldown = 8                 # 0.8 s before the next lane decision
        self.target_lane = best_lane
        return best_lane, best_info

    # -------------------------------------------------------------------- policy
    def act(self, obs=None, greedy: bool = True) -> int:
        env, track = self.env, self.track
        s = env.prev_s

        lane, (gap, v_lead) = self._choose_lane()
        offset_goal = float(env.lane_centers[lane])
        # smooth lane changes: move the lateral target gradually instead of teleporting
        if self.current_offset is None:
            self.current_offset = float(env._lateral)
        delta_off = np.clip(offset_goal - self.current_offset, -self.slew, self.slew)
        self.current_offset = float(self.current_offset + delta_off)
        offset = self.current_offset

        k_steer, delta = _steer_command(env, offset, self.lookahead)
        return _speed_action(self, gap, v_lead, delta, offset, k_steer)


class MemorylessExpertDriver:
    """The same lane-picking pure-pursuit policy, but with **no hidden state**.

    :class:`PurePursuitDriver` keeps three variables that are not part of the
    observation the learner sees:

    * ``target_lane``  -- the lane it latched onto, retained across steps,
    * ``current_offset`` -- a slewed lateral target, which makes the steering
      depend on where the car *was*,
    * ``_cooldown`` -- a timer that suppresses lane decisions.

    The labels it produces are therefore not a function of the policy input: two
    states that look identical to the network can carry different labels.  A
    behaviour-cloning model cannot learn such a mapping, and the resulting error
    is usually blamed on the network.

    This driver recomputes everything from the current simulator state each
    step, so ``act`` is a pure function of the environment:

    * the target lane is ``SensorFusion``'s current candidate
      (``target_available`` -> ``best_lane``, otherwise the ego lane), which is
      itself derived from gaps/velocities only;
    * the lateral target is that lane's centre, with no slew;
    * no cooldown: a lane is chosen or not chosen on the evidence at hand.

    Callers that need the old behaviour keep using :class:`PurePursuitDriver`;
    this class only exists so demonstrations are Markov in the policy input.
    """

    name = "expert-memoryless"

    def __init__(self, env, lookahead: float = 10.0, a_lat_max: float = 5.0,
                 v_target: float = 11.0, look_range: float = 34.0,
                 brake_range: float = 12.0, block_distance: float = 18.0,
                 lane_advantage_margin: float = 5.0, side_clearance: float = 7.0):
        self.env = env
        self.track = env.track
        self.lookahead = float(lookahead)
        self.a_lat_max = float(a_lat_max)
        self.v_target = float(v_target)
        self.look_range = float(look_range)
        self.brake_range = float(brake_range)
        self.sensor = SensorFusion({
            "block_distance": float(block_distance),
            "lane_advantage_margin": float(lane_advantage_margin),
            "min_side_gap": float(side_clearance),
        })
        self.kappa = _smoothed_curvature(self.track)

    def reset(self):                     # interface compatibility; nothing hidden
        pass

    def _steering(self, offset: float) -> tuple[int, float]:
        return _steer_command(self.env, offset, self.lookahead)

    def act(self, obs=None, greedy: bool = True) -> int:
        env = self.env
        frame = self.sensor.scan(env)

        # memoryless lane choice: the sensor's current candidate, no latching
        lane = int(frame.best_lane) if frame.target_available else int(frame.ego_lane)
        lane = int(np.clip(lane, 0, env.n_lanes - 1))
        offset = float(env.lane_centers[lane])
        reading = frame.reading(lane)
        gap = float(reading.front_gap)
        v_lead = float(reading.front_speed)

        k_steer, delta = self._steering(offset)

        # speed planning: curvature ahead, capped by the vehicle in front of the
        # *chosen* lane (same rule as PurePursuitDriver, no state involved)
        return _speed_action(self, gap, v_lead, delta, offset, k_steer)
