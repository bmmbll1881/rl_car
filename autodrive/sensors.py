"""SensorFusion / LaneRadar: an interpretable, camera-free lane observation layer.

Why this exists
---------------
The RARL policy used to reconstruct "which lane is free" inside the algorithm
(``_lane_gaps`` / ``_lane_features`` / ``_lane_clearance``), and each of those
helpers walked the NPC bookkeeping again.  That made the tactic hard to test and
impossible to reuse: a reward gate, the safety shield and the observation vector
could silently disagree about the same gap.

:class:`SensorFusion` is the single place that turns the environment's
**explicit state** -- ego arc position / speed, each NPC's arc position, lane id
and speed, and the track's lane geometry -- into per-lane range and closing-speed
readings.  No camera images, no pixel features, no hidden simulation internals:
every number can be recomputed by hand from ``env.track`` and ``env.obstacles``.

What a scan reports
-------------------
For every lane (three rows: the left neighbour, the ego lane, the right
neighbour), :class:`LaneReading` holds

* ``front_gap`` / ``rear_gap``        arc distance to the nearest NPC ahead /
                                      behind the ego (centre-to-centre, metres;
                                      ``inf`` when the lane is empty),
* ``front_rel_speed`` / ``rear_rel_speed``
                                      closing speed: ``ego_v - front_v`` and
                                      ``rear_v - ego_v`` (positive = approaching),
* ``front_ttc`` / ``rear_ttc``        time to contact after subtracting the
                                      vehicle extent (``inf`` when not closing),
* ``safe_window``                    seconds this lane stays safe to be in: the
                                      smallest time until either side closes to
                                      its minimum gap, saturated at
                                      ``ttc_horizon``,
* ``safe_to_enter``                  no car alongside and both gaps above their
                                      minimum.

around the rows, :class:`SensorFrame` adds the ego-lane scalars the task needs:
``ego_lead_gap``, ``ego_lead_speed``, ``ego_lead_speed_delta`` (ego minus lead),
``ego_lead_ttc`` and ``safe_to_accelerate``, plus the planner's candidate
(``best_lane`` / ``direction`` / ``target_gap`` / ``target_safe_window`` /
``target_available``) and ``npc_offsets``, the signed arc offset of every NPC
(ahead positive, behind negative) used by the overtake state machine to check
that the blocking car has actually been passed.

Feature vector
--------------
:meth:`SensorFusion.features` flattens a frame into ``SENSOR_DIM`` numbers in
[0, 1] (or [-1, 1] for the speed terms), so it can be appended to an RL policy
input.  The layout is, for rows ``[left, ego, right]``::

    front_gap, front_rel_speed, rear_gap, rear_rel_speed,
    front_ttc, rear_ttc, safe_window                      (7 per row)

followed by::

    safe_to_accelerate, direction, target_safe_window, target_available

A lane that does not exist on this road (two-lane track, road edge) encodes its
whole row as ``-1.0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SENSOR_CONFIG = {
    "range": 40.0,            # [m] gaps saturate here, then divide by it
    "v_scale": 16.0,          # [m/s] relative speeds are divided by this
    "ttc_horizon": 6.0,       # [s] TTC / safe-window saturation
    "vehicle_extent": 4.3,    # [m] combined half lengths (ego + NPC) for contact TTC
    "min_front_gap": 6.0,     # [m] centre-to-centre room needed ahead in a lane
    "min_rear_gap": 5.0,      # [m] centre-to-centre room needed behind
    "min_side_gap": 7.0,      # [m] a car this close alongside makes a lane unsafe
    "accel_min_gap": 12.0,    # [m] lead gap needed to call acceleration safe
    "accel_max_closing": 2.0,  # [m/s] faster closing than this is not "safe"
    "accel_min_ttc": 1.0,     # [s] contact TTC needed to call acceleration safe
    "lane_advantage_margin": 5.0,  # [m] a neighbour must beat the ego lane by this
    "block_distance": 18.0,   # [m] ego lane counts as blocked below this gap
}

ROW_FEATURES = 7
ROW_LANES = 3                     # left neighbour, ego lane, right neighbour
SCALAR_FEATURES = 4
SENSOR_DIM = ROW_LANES * ROW_FEATURES + SCALAR_FEATURES      # == 25


@dataclass(frozen=True)
class LaneReading:
    """One lane's range/velocity picture, as seen from the ego."""

    lane: int
    exists: bool
    front_gap: float = float("inf")
    front_speed: float = 0.0
    rear_gap: float = float("inf")
    rear_speed: float = 0.0
    front_rel_speed: float = 0.0     # ego_v - front_v  (>0: closing on the lead)
    rear_rel_speed: float = 0.0      # rear_v - ego_v   (>0: follower closing in)
    front_ttc: float = float("inf")
    rear_ttc: float = float("inf")
    safe_window: float = float("inf")
    safe_to_enter: bool = False


@dataclass(frozen=True)
class SensorFrame:
    """Fused lane-radar view of the current environment state."""

    ego_lane: int
    ego_speed: float
    rows: tuple[LaneReading, ...]        # [left, ego, right] feature window
    front_gaps: np.ndarray = field(repr=False)
    rear_gaps: np.ndarray = field(repr=False)
    side_gaps: np.ndarray = field(repr=False)
    front_ids: np.ndarray = field(repr=False)
    readings: tuple[LaneReading, ...] = ()  # every lane, in absolute order
    # Signed arc offset per NPC id (metres ahead of the ego, negative == behind).
    # The overtake state machine uses it to verify that the car which forced the
    # manoeuvre really ended up behind the ego after the lane change.
    npc_offsets: dict = field(default_factory=dict, repr=False)
    ego_lead_gap: float = float("inf")
    ego_lead_speed: float = 0.0
    ego_lead_speed_delta: float = 0.0
    ego_lead_ttc: float = float("inf")
    safe_to_accelerate: bool = False
    best_lane: int = 0
    direction: int = 0
    target_gap: float = 0.0
    target_safe_window: float = 0.0
    target_available: bool = False

    def reading(self, lane: int) -> LaneReading:
        """Reading for an absolute lane index (``-1`` / ``n_lanes`` returns an empty one)."""
        for row in (self.readings or self.rows):
            if row.lane == int(lane):
                return row
        return LaneReading(lane=int(lane), exists=False)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


class SensorFusion:
    """Lane-resolved range/closing-speed sensor fed by interpretable state.

    ``scan(env)`` is read-only and cannot change the simulation, so it is safe to
    call from the observation path, the reward gate and tests.
    """

    DIM = SENSOR_DIM

    def __init__(self, cfg=None):
        c = dict(DEFAULT_SENSOR_CONFIG)
        for k, v in (cfg or {}).items():
            if k not in c:
                raise ValueError(f"unknown sensor config key '{k}'")
            c[k] = v
        self.cfg = c
        self.range = float(c["range"])
        self.v_scale = max(float(c["v_scale"]), 1e-6)
        self.horizon = max(float(c["ttc_horizon"]), 1e-6)
        self.extent = float(c["vehicle_extent"])
        self.last_frame: SensorFrame | None = None

    # ------------------------------------------------------------------ scan
    def scan(self, env) -> SensorFrame:
        """Read the current environment state into a :class:`SensorFrame`."""
        track = env.track
        ob = env.obstacles
        L = float(track.length)
        ego_s = float(env.prev_s)
        ego_v = float(env.v)
        n_lanes = int(len(env.lane_centers))
        ego_lane = int(env.lane_of())

        front_gaps = np.full(n_lanes, np.inf)
        rear_gaps = np.full(n_lanes, np.inf)
        side_gaps = np.full(n_lanes, np.inf)
        front_ids = np.full(n_lanes, -1, dtype=int)
        front_speeds = np.zeros(n_lanes)
        rear_speeds = np.zeros(n_lanes)
        npc_offsets: dict[int, float] = {}

        if int(ob.n) > 0:
            s = np.asarray(ob.s, dtype=float)
            ahead = (s - ego_s) % L
            behind = (ego_s - s) % L
            circ = np.minimum(ahead, behind)
            signed = ahead.copy()
            signed[signed > L / 2.0] -= L
            npc_offsets = {int(i): float(signed[i]) for i in range(int(ob.n))}
            lanes = np.asarray(ob.lane, dtype=int)
            speeds = np.asarray(ob.v, dtype=float)
            for lane in range(n_lanes):
                sel = np.where(lanes == lane)[0]
                if sel.size == 0:
                    continue
                i = int(sel[int(np.argmin(ahead[sel]))])
                j = int(sel[int(np.argmin(behind[sel]))])
                front_gaps[lane] = float(ahead[i])
                front_speeds[lane] = float(speeds[i])
                front_ids[lane] = i
                rear_gaps[lane] = float(behind[j])
                rear_speeds[lane] = float(speeds[j])
                side_gaps[lane] = float(circ[sel].min())

        readings = tuple(
            self._reading(lane, True, front_gaps[lane], front_speeds[lane],
                          rear_gaps[lane], rear_speeds[lane], side_gaps[lane], ego_v)
            for lane in range(n_lanes)
        )
        rows = []
        for lane in (ego_lane + 1, ego_lane, ego_lane - 1):
            if 0 <= lane < n_lanes:
                rows.append(readings[lane])
            else:
                rows.append(LaneReading(lane=int(lane), exists=False))
        rows = tuple(rows)

        ego_reading = readings[ego_lane]
        safe_accel = self._safe_to_accelerate(ego_reading)

        best_lane, target_gap = ego_lane, ego_reading.front_gap
        margin = float(self.cfg["lane_advantage_margin"])
        for lane in (ego_lane - 1, ego_lane + 1):
            if not 0 <= lane < n_lanes:
                continue
            cand = readings[lane]
            if not cand.safe_to_enter:
                continue
            if cand.front_gap < ego_reading.front_gap + margin:
                continue
            if cand.front_gap > target_gap:
                best_lane, target_gap = lane, cand.front_gap

        available = bool(ego_reading.front_gap < float(self.cfg["block_distance"])
                         and best_lane != ego_lane)
        if available:
            target = readings[best_lane]
            target_gap = target.front_gap
        else:
            target = ego_reading
            target_gap = ego_reading.front_gap
        frame = SensorFrame(
            ego_lane=ego_lane,
            ego_speed=ego_v,
            rows=rows,
            readings=readings,
            front_gaps=front_gaps,
            rear_gaps=rear_gaps,
            side_gaps=side_gaps,
            front_ids=front_ids,
            npc_offsets=npc_offsets,
            ego_lead_gap=ego_reading.front_gap,
            ego_lead_speed=ego_reading.front_speed,
            ego_lead_speed_delta=ego_reading.front_rel_speed,
            ego_lead_ttc=ego_reading.front_ttc,
            safe_to_accelerate=bool(safe_accel),
            best_lane=int(best_lane),
            direction=int(np.sign(best_lane - ego_lane)) if available else 0,
            target_gap=float(target_gap) if np.isfinite(target_gap) else float(self.range),
            target_safe_window=float(target.safe_window) if available else 0.0,
            target_available=available,
        )
        self.last_frame = frame
        return frame

    # -------------------------------------------------------------- readings
    def _reading(self, lane, exists, front_gap, front_speed, rear_gap, rear_speed,
                 side_gap, ego_v) -> LaneReading:
        if not exists:
            return LaneReading(lane=lane, exists=False)
        front_rel = (ego_v - front_speed) if np.isfinite(front_gap) else 0.0
        rear_rel = (rear_speed - ego_v) if np.isfinite(rear_gap) else 0.0
        front_ttc = self._ttc(front_gap, front_rel)
        rear_ttc = self._ttc(rear_gap, rear_rel)
        window = min(
            self._window(front_gap, front_rel, float(self.cfg["min_front_gap"])),
            self._window(rear_gap, rear_rel, float(self.cfg["min_rear_gap"])),
        )
        safe = bool(front_gap >= float(self.cfg["min_front_gap"])
                    and rear_gap >= float(self.cfg["min_rear_gap"])
                    and side_gap >= float(self.cfg["min_side_gap"]))
        return LaneReading(
            lane=int(lane), exists=True,
            front_gap=float(front_gap), front_speed=float(front_speed),
            rear_gap=float(rear_gap), rear_speed=float(rear_speed),
            front_rel_speed=float(front_rel), rear_rel_speed=float(rear_rel),
            front_ttc=float(front_ttc), rear_ttc=float(rear_ttc),
            safe_window=float(window), safe_to_enter=safe,
        )

    def _ttc(self, gap: float, closing: float) -> float:
        if not np.isfinite(gap) or closing <= 1e-6:
            return float("inf")
        contact = max(float(gap) - self.extent, 0.0)
        return float(contact / closing)

    def _window(self, gap: float, closing: float, min_gap: float) -> float:
        if not np.isfinite(gap):
            return float(self.horizon)
        if closing <= 1e-6:
            return float(self.horizon)
        return float(min(max((float(gap) - min_gap) / closing, 0.0), self.horizon))

    def _safe_to_accelerate(self, ego: LaneReading) -> bool:
        if not ego.exists:
            return True
        return bool(
            ego.front_gap >= float(self.cfg["accel_min_gap"])
            and ego.front_rel_speed <= float(self.cfg["accel_max_closing"])
            and ego.front_ttc >= float(self.cfg["accel_min_ttc"])
        )

    # -------------------------------------------------------------- features
    def _scaled(self, value: float, scale: float) -> float:
        if not np.isfinite(value):
            return 1.0
        return _clip(float(value) / max(float(scale), 1e-6), 0.0, 1.0)

    def _row(self, r: LaneReading) -> list[float]:
        if not r.exists:
            return [-1.0] * ROW_FEATURES
        return [
            self._scaled(r.front_gap, self.range),
            _clip(r.front_rel_speed / self.v_scale, -1.0, 1.0),
            self._scaled(r.rear_gap, self.range),
            _clip(r.rear_rel_speed / self.v_scale, -1.0, 1.0),
            self._scaled(r.front_ttc, self.horizon),
            self._scaled(r.rear_ttc, self.horizon),
            self._scaled(r.safe_window, self.horizon),
        ]

    def features_from(self, frame: SensorFrame) -> np.ndarray:
        """Flatten ``frame`` into the fixed-width, bounded policy feature vector."""
        values: list[float] = []
        for row in frame.rows:
            values.extend(self._row(row))
        values.extend([
            1.0 if frame.safe_to_accelerate else 0.0,
            float(frame.direction),
            self._scaled(frame.target_safe_window, self.horizon) if frame.target_available else 0.0,
            1.0 if frame.target_available else 0.0,
        ])
        return np.asarray(values, dtype=float)

    def features(self, env) -> np.ndarray:
        """``scan`` + ``features_from`` in one call (caches ``last_frame``)."""
        return self.features_from(self.scan(env))


#: Alias kept because "lane radar" is the name used in the design notes.
LaneRadar = SensorFusion
