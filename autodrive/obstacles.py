"""Dynamic obstacles: dense, individually-paced traffic that never traps the ego.

Design
------
* **Every vehicle has its own speed** (``v_range`` per car, plus a slow
  per-car modulation with a different phase), so the stream is *not* synchronised:
  cars catch up, spread out, form and dissolve platoons — and three cars can end
  up abreast purely from the dynamics.
* **Three-abreast is allowed** — ``npc_three_abreast`` blocks are placed on
  purpose at random spots, and emergent ones are fine too.  A wall of traffic is
  realistic; what must not happen is the ego being stuck behind one forever.
* **The wall always gives way** — every step the set checks whether all lanes are
  occupied within ``block_range`` *ahead of the ego*; if so, the lead vehicle of
  the lane with the most room is accelerated (a "traffic flow" nudge) until that
  lane opens.  So a 3-lane block is a transient picture, not a dead end: within a
  couple of seconds the ego always has a route again.
* Otherwise the layout keeps the "at most 2 of 3 lanes in a 12 m window" rule, so
  the road reads as dense but normally has a free lane.

The set is deterministic given the environment seed and steps with the ego's dt.
"""
from __future__ import annotations

import numpy as np


class ObstacleSet:
    """A dense fleet of lane-bound NPC vehicles with independent speeds."""

    def __init__(self, track, n: int = 24, rng=None, v_range=(4.5, 9.0), lane_centers=(0.0,),
                 lane_width: float = 3.2, drift: float = 0.22, n_side_by_side: int = 2,
                 n_three_abreast: int = 2, block_range: float = 32.0,
                 length: float = 4.4, width: float = 1.9, ego_length: float = 4.2, ego_width: float = 2.0):
        self.track = track
        self.n = int(n)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.v_range = v_range
        self.lane_centers = np.asarray(lane_centers, dtype=float)
        self.lane_width = float(lane_width)
        self.drift = float(drift)
        self.n_side_by_side = int(n_side_by_side)
        self.n_three_abreast = int(n_three_abreast)
        self.block_range = float(block_range)
        self.length = length
        self.width = width
        self.ego_length = ego_length
        self.ego_width = ego_width
        self.lateral_max = float(np.max(np.abs(self.lane_centers))) + self.lane_width / 2.0 - self.width / 2.0 - 0.15
        # placement parameters
        self.window = 12.0         # [m] corridor window for the "leave a lane free" rule
        self.same_lane_gap = 18.0  # [m] minimum bumper-to-bumper distance within a lane

        self.s = np.zeros(self.n)
        self.lane = np.zeros(self.n, dtype=int)
        self.lat = np.zeros(self.n)
        self.lat_base = np.zeros(self.n)
        self.lat_amp = np.zeros(self.n)
        self.lat_omega = np.zeros(self.n)
        self.lat_phase = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self.v_nominal = np.zeros(self.n)
        self.boost = np.zeros(self.n)
        self.t = 0.0
        self.pos = np.zeros((self.n, 2))
        self.heading = np.zeros(self.n)
        self._idx = np.zeros(self.n, dtype=int)

    # ------------------------------------------------------------------ helpers
    def _arc(self, a: float, b: float) -> float:
        d = abs(a - b) % self.track.length
        return min(d, self.track.length - d)

    def _pick_lane(self, s: float, placed, n_lanes: int, forbid=(), allow_wall: bool = False):
        """A lane that keeps same-lane spacing (and, unless ``allow_wall``, a free lane)."""
        ok = []
        for lane in range(n_lanes):
            if lane in forbid:
                continue
            if any(l2 == lane and self._arc(s, s2) < self.same_lane_gap for s2, l2 in placed):
                continue
            if not allow_wall:
                occupied = {l2 for s2, l2 in placed if self._arc(s, s2) < self.window}
                if len(occupied | {lane}) > n_lanes - 1:
                    continue
            ok.append(lane)
        if not ok:
            return None
        return int(ok[self.rng.integers(len(ok))])

    # ------------------------------------------------------------------ spawning
    def reset(self, ego_s: float = 0.0, min_gap: float = 25.0):
        """Place the fleet: dense, some two-abreast pairs, a few three-abreast walls."""
        L = self.track.length
        if self.n == 0:
            self.lane = np.zeros(0, dtype=int)
            self.pos = np.zeros((0, 2))
            return
        n_lanes = len(self.lane_centers)
        span = max(L - 2.0 * min_gap, 1.0)
        placed = []

        # ---- 1. deliberate three-abreast walls -----------------------------
        n_walls = min(self.n_three_abreast, max(self.n // n_lanes, 0))
        for w in range(n_walls):
            s_w = (ego_s + min_gap + (w + 1) * span / (n_walls + 1.0)) % L
            lanes = self.rng.permutation(n_lanes)
            for j in range(n_lanes):
                placed.append((float((s_w + j * 2.2) % L), int(lanes[j])))

        # ---- 2. two-abreast pairs -----------------------------------------
        pairs_left = min(self.n_side_by_side, max(self.n // 2, 0))
        stride = span / max(self.n, 1)
        s = ego_s + min_gap
        while pairs_left > 0 and len(placed) < self.n:
            s_k = (s + self.rng.uniform(0.2, 0.8) * stride) % L
            lane = self._pick_lane(s_k, placed, n_lanes)
            if lane is None:
                s += stride
                continue
            placed.append((s_k, lane))
            partner = self._pick_lane((s_k + 1.5) % L, placed, n_lanes, forbid=(lane,))
            if partner is not None:
                placed.append((float((s_k + 1.5) % L), partner))
                pairs_left -= 1
            s += stride

        # ---- 3. fill the rest along the lap --------------------------------
        stride = span / max(self.n - len(placed), 1)
        s = ego_s + min_gap
        guard = 0
        while len(placed) < self.n and guard < 4 * self.n:
            guard += 1
            s_k = (s + self.rng.uniform(-0.2, 0.2) * stride) % L
            lane = None
            for _ in range(10):
                lane = self._pick_lane(s_k, placed, n_lanes)
                if lane is not None:
                    break
                s_k = (s_k + 5.0) % L
            if lane is None:
                continue
            placed.append((s_k, lane))
            s += stride

        self.n = len(placed)
        self.s = np.array([p[0] for p in placed])
        self.lane = np.array([p[1] for p in placed], dtype=int)
        self.lat = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self.v_nominal = np.zeros(self.n)
        self.boost = np.zeros(self.n)
        self.pos = np.zeros((self.n, 2))
        self.heading = np.zeros(self.n)
        self._idx = np.zeros(self.n, dtype=int)

        # ---- 4. individual speeds (NOT synchronised) -----------------------
        self.v_nominal = self.rng.uniform(*self.v_range, size=self.n)
        self.v = self.v_nominal.copy()
        self.lat_base = self.lane_centers[self.lane].copy()
        self.lat_amp = self.rng.uniform(0.0, self.drift, size=self.n)
        self.lat_omega = self.rng.uniform(0.15, 0.45, size=self.n)
        self.lat_phase = self.rng.uniform(0.0, 2 * np.pi, size=self.n)
        self.t = 0.0
        self.lat = self.lat_base.copy()
        self._update_positions()

    # ------------------------------------------------------------------ dynamics
    def step(self, dt: float, ego_s: float | None = None, adversary=None):
        """Advance the fleet.

        ``adversary`` (optional) is ``(index, steer_cmd, throttle_cmd)`` with both
        commands in [-1, 1]: that vehicle is driven externally (by a learned
        adversarial policy) instead of following its lane.
        """
        if self.n == 0:
            return
        L = self.track.length
        self.t += dt
        # every car has its own pace and its own phase -> the stream is not synced
        desired = self.v_nominal * (1.0 + 0.10 * np.sin(0.6 * self.t + self.lat_phase))
        self.boost *= 0.985
        desired = desired + self.boost

        # --- car following inside each lane (prevents NPC/NPC overlap) -------
        for i in range(self.n):
            others = np.where(self.lane == self.lane[i])[0]
            others = others[others != i]
            if others.size == 0:
                continue
            gaps = (self.s[others] - self.s[i]) % L
            j = int(np.argmin(gaps))
            gap = float(gaps[j])
            if gap < self.same_lane_gap:
                desired[i] = min(desired[i], self.v[others[j]] * (0.4 + 0.6 * gap / self.same_lane_gap))

        self.v = np.clip(self.v + np.clip(desired - self.v, -5.0 * dt, 3.0 * dt), 1.0, 13.0)
        self.s = (self.s + self.v * dt) % L
        self.lat = self.lat_base + self.lat_amp * np.sin(self.lat_omega * self.t + self.lat_phase)
        self._update_positions()

        # --- externally driven adversarial vehicle ---------------------------
        if adversary is not None and self.n > 0:
            idx, steer_cmd, throttle_cmd = adversary
            idx = int(idx) % self.n
            v = float(np.clip(self.v[idx] + 6.0 * float(throttle_cmd) * dt, 0.5, 13.0))
            lat = float(np.clip(self.lat[idx] + 2.6 * float(steer_cmd) * dt,
                                -self.lateral_max, self.lateral_max))
            self.v[idx] = v
            self.s[idx] = (self.s[idx] + v * dt) % L
            self.lat[idx] = lat
            self._update_positions()

        if ego_s is not None:
            self._release_blockage(ego_s, dt)

    def place_lead(self, ego_s: float, lane: int, gap: float,
                   speed: float | None = None, exclude=()) -> int:
        """Move one vehicle into ``lane``, ``gap`` metres ahead of the ego.

        Used by the overtake curriculum: the ego is guaranteed to meet a slower
        car in its own lane within the first seconds of the episode, so "close
        the gap, then decide" is actually experienced during training.  Only
        vehicles that can be moved without breaking the same-lane spacing rule
        are considered; the one already closest to the target is taken.

        Returns the moved index, or ``-1`` when the insertion is impossible
        (no traffic, or every candidate would overlap its lane neighbours).
        """
        return self._place_at((float(ego_s) + float(gap)) % self.track.length,
                              lane, speed, exclude)

    def place_follow(self, ego_s: float, lane: int, gap: float,
                     speed: float | None = None, exclude=()) -> int:
        """Move one vehicle into ``lane``, ``gap`` metres *behind* the ego.

        The reverse of :meth:`place_lead`; the stage-2 curriculum uses it to
        put a fast car in the target lane so the state machine has to weigh the
        rear safe window instead of always seeing an empty lane.
        """
        return self._place_at((float(ego_s) - float(gap)) % self.track.length,
                              lane, speed, exclude)

    def _place_at(self, target: float, lane: int, speed: float | None = None,
                  exclude=()) -> int:
        if self.n == 0:
            return -1
        L = self.track.length
        lane = int(lane)
        skip = {int(i) for i in exclude}
        best, best_dist = -1, np.inf
        for i in range(self.n):
            if i in skip:
                continue
            d = self._arc(self.s[i], target)
            if d >= best_dist:
                continue
            same = np.where((self.lane == lane) & (np.arange(self.n) != i))[0]
            if same.size:
                ds = np.abs(self.s[same] - target) % L
                if float(np.minimum(ds, L - ds).min()) < self.same_lane_gap:
                    continue
            best, best_dist = i, d
        if best < 0:
            return -1
        self.s[best] = target
        self.lane[best] = lane
        self.lat_base[best] = self.lane_centers[lane]
        self.lat[best] = self.lat_base[best]
        self.lat_amp[best] = 0.0            # the planted car keeps its lane
        self.v_nominal[best] = float(self.v_range[0] if speed is None else speed)
        self.v[best] = self.v_nominal[best]
        self.boost[best] = 0.0
        self._update_positions()
        return int(best)

    def place_adversary(self, ego_s: float, index: int = 0, gap: float = 14.0) -> int:
        """Drop one vehicle just ahead of the ego and hand it to the adversary policy."""
        if self.n == 0:
            return -1
        i = int(index) % self.n
        lane = len(self.lane_centers) // 2
        self.s[i] = (ego_s + gap) % self.track.length
        self.lane[i] = lane
        self.lat_base[i] = self.lane_centers[lane]
        self.lat[i] = self.lat_base[i]
        self.v[i] = 6.0
        self.v_nominal[i] = 6.0
        self._update_positions()
        return i

    def _release_blockage(self, ego_s: float, dt: float):
        """If every lane is occupied within ``block_range`` ahead, open the best one."""
        leads = [self.lane_lead(lane, ego_s) for lane in range(len(self.lane_centers))]
        if not all(gap < self.block_range for gap, _ in leads):
            return
        # open the lane that needs the least effort: accelerate its lead car
        lane = int(np.argmax([gap for gap, _ in leads]))
        idx = np.where(self.lane == lane)[0]
        if idx.size == 0:
            return
        gaps = (self.s[idx] - ego_s) % self.track.length
        j = int(idx[np.argmin(gaps)])
        self.boost[j] = max(float(self.boost[j]), 3.0)

    def _update_positions(self):
        s = np.asarray(self.s)
        idx = np.searchsorted(self.track.s, s) % self.track.n_points
        self._idx = idx
        pts = self.track.points[idx]
        nrm = self.track.normals[idx]
        tan = self.track.tangents[idx]
        self.pos = pts + nrm * self.lat[:, None]
        self.heading = np.arctan2(tan[:, 1], tan[:, 0])

    # ------------------------------------------------------------------- queries
    def nearest(self, xy, k: int = 3, max_range: float = 20.0):
        """Return ``(dist, bearing, v_los)`` for the ``k`` closest NPCs."""
        if self.n == 0:
            return np.full(k, max_range), np.zeros(k), np.zeros(k)
        d = self.pos - np.asarray(xy, dtype=float).reshape(1, 2)
        dist = np.linalg.norm(d, axis=1)
        order = np.argsort(dist)
        kk = min(k, self.n)
        out_d = np.full(k, max_range)
        out_b = np.zeros(k)
        out_v = np.zeros(k)
        sel = order[:kk]
        out_d[:kk] = dist[sel]
        bearing = np.arctan2(d[sel, 1], d[sel, 0])
        out_b[:kk] = bearing
        out_v[:kk] = self.v[sel] * np.cos(self.heading[sel] - bearing)
        return out_d, out_b, out_v

    def lane_lead(self, lane_id: int, ego_s: float):
        """``(gap, speed)`` of the first NPC of that lane ahead of ``ego_s``."""
        L = self.track.length
        idx = np.where(self.lane == lane_id)[0]
        if idx.size == 0:
            return L, float("inf")
        gaps = (self.s[idx] - ego_s) % L
        j = int(np.argmin(gaps))
        return float(gaps[j]), float(self.v[idx[j]])

    def lane_free_distance(self, lane_id: int, ego_s: float) -> float:
        return self.lane_lead(lane_id, ego_s)[0]

    def is_blocked(self, ego_s: float, block_range: float | None = None) -> bool:
        """True when every lane has a vehicle within ``block_range`` ahead."""
        rng = self.block_range if block_range is None else block_range
        return all(self.lane_lead(lane, ego_s)[0] < rng for lane in range(len(self.lane_centers)))

    def max_lanes_occupied(self, window: float | None = None) -> int:
        """Worst-case number of lanes occupied inside any static corridor window."""
        if self.n == 0:
            return 0
        w = self.window if window is None else window
        L = self.track.length
        worst = 0
        for s0 in np.arange(0.0, L, 2.0):
            occ = {int(l) for ss, l in zip(self.s, self.lane) if self._arc(s0, ss) < w}
            worst = max(worst, len(occ))
        return worst

    def wall_windows(self, window: float | None = None):
        """Arc positions where all lanes are occupied within ``window``."""
        if self.n == 0:
            return []
        w = self.window if window is None else window
        L = self.track.length
        n_lanes = len(self.lane_centers)
        out = []
        for s0 in np.arange(0.0, L, 2.0):
            occ = {int(l) for ss, l in zip(self.s, self.lane) if self._arc(s0, ss) < w}
            if len(occ) >= n_lanes:
                out.append(float(s0))
        return out

    def side_by_side_pairs(self, max_ds: float = 4.0):
        """Indices of pairs driving two-abreast (different lanes, nearly same s)."""
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.lane[i] != self.lane[j] and self._arc(self.s[i], self.s[j]) <= max_ds:
                    pairs.append((i, j))
        return pairs

    def ray_hits(self, origin, headings, max_range: float):
        if self.n == 0:
            return np.full(len(headings), np.inf)
        o = np.asarray(origin, dtype=float).reshape(2)
        h = np.asarray(headings, dtype=float).reshape(-1)
        d = np.stack([np.cos(h), np.sin(h)], axis=1)
        r = 0.5 * float(np.hypot(self.length, self.width)) * 0.85
        c = self.pos
        oc = c[None, :, :] - o[None, None, :]
        proj = np.einsum("rnk,rk->rn", oc, d)
        perp2 = np.einsum("rnk,rnk->rn", oc, oc) - proj * proj
        hit = (perp2 <= r * r) & (proj > 0) & (proj <= max_range)
        t = proj - np.sqrt(np.maximum(r * r - perp2, 0.0))
        t = np.where(hit, np.maximum(t, 0.0), np.inf)
        return t.min(axis=1)

    def collision(self, xy, heading: float = 0.0) -> bool:
        if self.n == 0:
            return False
        d = np.asarray(xy, dtype=float).reshape(2) - self.pos
        ca, sa = np.cos(-self.heading), np.sin(-self.heading)
        along = d[:, 0] * ca - d[:, 1] * sa
        lat = d[:, 0] * sa + d[:, 1] * ca
        half_along = 0.5 * (self.ego_length + self.length) * 0.92
        half_lat = 0.5 * (self.ego_width + self.width) * 0.88
        return bool(np.any((np.abs(along) < half_along) & (np.abs(lat) < half_lat)))

    def min_distance(self, xy) -> float:
        if self.n == 0:
            return float("inf")
        return float(np.min(np.linalg.norm(self.pos - np.asarray(xy, dtype=float).reshape(1, 2), axis=1)))

    def describe(self) -> str:
        return (f"{self.n} NPC vehicles in {len(self.lane_centers)} lanes "
                f"({self.lane_width:.1f} m wide), speeds {self.v_range[0]:.1f}-{self.v_range[1]:.1f} m/s "
                f"(independent), {self.n_side_by_side} two-abreast pairs, "
                f"{self.n_three_abreast} three-abreast walls that clear when the ego arrives")
