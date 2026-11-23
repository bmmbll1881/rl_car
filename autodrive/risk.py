"""Risk-aware observation layer, ported from `jingyunWu-23/risk`.

Two pieces of that repository are reproduced here, adapted to MiniDrive's state
(no CARLA, no torch):

``TimeVaryingRiskField``  (``rarl/risk/risk_field.py``)
    *road risk*      = w_road / max(road_half_width - |lateral|, margin)^2
                     + w_center * (1 - exp(-decay * |lateral - nearest_lane_centre|))
    *vehicle risk*   = sum over belief modes of p * (inside ellipse ? w_v
                     : w_v * exp(-decay * excess_ratio)), where the ellipse axes are
                     scaled by sqrt(-2 ln(1 - p)) and widened by the uncertainty.

``GaussianMixtureBelief``  (``rarl/belief/gm_bayes.py``)
    Per surrounding vehicle: a 4-D constant-velocity state ``[x, y, vx, vy]`` with
    ``num_modes`` lateral-manoeuvre hypotheses; IMM mixing -> CV predict ->
    Kalman update -> mode re-weighting.  The posterior uncertainty (trace of the
    covariance) feeds the vehicle risk.

Both are pure numpy and deterministic.  The mode axis is vectorised (the
reference implementation loops over modes and inverts a 4x4 matrix per mode),
which is what makes this fast enough to run inside the training loop.

``TTCCollisionRisk`` is an *optional* extension (``hazard_mode="ttc"`` in
``autodrive/algos/rarl.py``).  Instead of a static ellipse around every traffic
car it looks at the time-to-collision along the ego heading, so a car one lane
over is ignored until an action actually moves the ego towards it.  It is off by
default: the legacy 48-dim policy input must stay loadable.
"""
from __future__ import annotations

import math

import numpy as np


DEFAULT_RISK_CONFIG = {
    # road field (rarl/risk/risk_field.py)
    "road_weight": 1.0,
    "center_weight": 0.1,
    "center_decay": 0.8,
    "boundary_safety_margin": 0.4,
    # vehicle field
    "vehicle_weight": 2.0,
    "uncertainty_weight": 1.0,
    "vehicle_decay": 1.0,
    # belief (rarl/belief/gm_bayes.py)
    "num_modes": 7,
    "process_noise": 0.15,
    "measurement_noise": 0.5,
    "min_mode_probability": 0.001,
}


class TimeVaryingRiskField:
    """Road + multimodal vehicle risk (see module docstring)."""

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT_RISK_CONFIG)
        self.cfg.update(cfg or {})

    # ------------------------------------------------------------------- road
    def road_risk(self, lateral: float, half_width: float, lane_centers) -> float:
        margin = float(self.cfg["boundary_safety_margin"])
        boundary_gap = float(half_width) - abs(float(lateral))
        boundary = float(self.cfg["road_weight"]) / max(boundary_gap, margin) ** 2
        centers = np.asarray(lane_centers, dtype=float)
        nearest = float(centers[int(np.argmin(np.abs(centers - lateral)))])
        center = float(self.cfg["center_weight"]) * (
            1.0 - math.exp(-float(self.cfg["center_decay"]) * abs(float(lateral) - nearest))
        )
        return float(boundary + center)

    # ---------------------------------------------------------------- vehicle
    def vehicle_risk(self, ego_xy, sv_modes, length: float = 4.4, width: float = 1.9) -> float:
        """``sv_modes``: iterable of lists of ``(mean_xy, cov2, probability)``."""
        ego = np.asarray(ego_xy, dtype=float).reshape(2)
        risk = 0.0
        w_u = float(self.cfg["uncertainty_weight"])
        w_v = float(self.cfg["vehicle_weight"])
        for modes in sv_modes:
            for mean, cov2, prob in modes:
                p = min(max(float(prob), 1e-6), 0.999)
                uncertainty = max(0.0, float(np.trace(np.asarray(cov2, dtype=float))))
                # an improbable mode has to be *closer* to matter (same as the repo)
                scale = math.sqrt(max(1e-6, -2.0 * math.log(max(1e-6, 1.0 - p))))
                axis_x = max((length + w_u * uncertainty) * scale, 1e-3)
                axis_y = max((width + 0.5 * w_u * uncertainty) * scale, 1e-3)
                delta = ego - np.asarray(mean, dtype=float).reshape(2)
                ellipse = (delta[0] / axis_x) ** 2 + (delta[1] / axis_y) ** 2
                if ellipse <= 1.0:
                    mode_risk = w_v
                else:
                    mode_risk = w_v * math.exp(-float(self.cfg["vehicle_decay"]) * (math.sqrt(ellipse) - 1.0))
                risk += p * mode_risk
        return float(risk)

    def total(self, ego_xy, lateral, half_width, lane_centers, sv_modes, length=4.4, width=1.9) -> dict:
        road = self.road_risk(lateral, half_width, lane_centers)
        vehicle = self.vehicle_risk(ego_xy, sv_modes, length, width)
        total = road + vehicle
        return {
            "road_risk": road,
            "vehicle_risk": vehicle,
            "total_risk": total,
            "risk_margin": 1.0 / (1.0 + total),
        }

    def features(self, *args, **kwargs) -> np.ndarray:
        """The 4 risk features, squashed into [0, 1] so they match the env observation.

        The raw field diverges near the road edge (1/gap^2 -> ~100) and would
        otherwise dominate a network whose other inputs are O(1).  ``x/(1+x)`` is
        monotone, so the ordering — and therefore the risk-aware behaviour — is
        preserved.
        """
        d = self.total(*args, **kwargs)
        road, veh, total = d["road_risk"], d["vehicle_risk"], d["total_risk"]
        return np.array([road / (1.0 + road), veh / (1.0 + veh),
                         total / (1.0 + total), d["risk_margin"]], dtype=float)


DEFAULT_TTC_CONFIG = {
    # time-to-collision gate: below ``ttc_safe_time`` the hazard ramps up, above
    # ``ttc_horizon`` a candidate is ignored
    "ttc_safe_time": 2.0,
    "ttc_horizon": 3.0,
    # a vehicle only matters when its lateral clearance is inside this band
    "ttc_clearance_margin": 0.3,
    "ttc_uncertainty_weight": 0.5,
    # vehicle footprint (combined half extents), so gaps are bumper-to-bumper
    "ttc_vehicle_length": 4.4,
    "ttc_vehicle_width": 1.9,
    "ttc_proximity_range": 12.0,
    "ttc_action_scale": 2.5,
    # kinematic bicycle model used to roll out action candidates
    "steer_max": 0.45,
    "wheelbase": 2.6,
    "accel_max": 5.0,
    "brake_max": 10.0,
    "drag": 0.25,
    "v_max": 16.0,
    "a_lat_max": 9.0,
}


class TTCCollisionRisk:
    """Longitudinal collision hazard from multimodal traffic predictions.

    The same formula serves two purposes:

    * :meth:`estimate` answers "how dangerous is the *current* state?" for the
      observation/reward (front and rear vehicles separately);
    * :meth:`action_risks` answers "how dangerous is each candidate action?" by
      rolling the ego forward with the kinematic bicycle model and checking the
      predicted poses against the predicted positions of the traffic modes.

    A mode is ``(mean, cov2, probability)`` where ``mean`` is either ``[x, y]``
    or ``[x, y, vx, vy]``; the 4-vector form is what the belief exposes through
    :meth:`GaussianMixtureBelief.modes_with_velocity`.
    """

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT_TTC_CONFIG)
        self.cfg.update(cfg or {})

    # ------------------------------------------------------------- primitives
    @staticmethod
    def _split_mode(mean, cov2, prob):
        m = np.asarray(mean, dtype=float).reshape(-1)
        xy = m[:2]
        vxy = m[2:4] if m.size >= 4 else np.zeros(2)
        cov = np.asarray(cov2, dtype=float).reshape(2, 2) if cov2 is not None else np.zeros((2, 2))
        return xy, vxy, cov, float(prob)

    def _clearance_margin(self, cov) -> float:
        """Lateral band an overlap has to beat (footprint + margin + uncertainty).

        Split out of :meth:`_lateral_clearance` so the per-action rollout can
        compute it once per traffic mode instead of once per predicted pose.
        """
        c = np.asarray(cov, dtype=float).reshape(2, 2)
        uncertainty = math.sqrt(max(0.0, float(c[0, 0]) + float(c[1, 1])))
        margin = float(self.cfg["ttc_clearance_margin"])
        margin += float(self.cfg["ttc_uncertainty_weight"]) * uncertainty
        return float(self.cfg["ttc_vehicle_width"]) + margin

    def _lateral_clearance(self, rel_y: float, cov) -> float:
        """Signed gap between the two footprints (negative == overlapping)."""
        return abs(float(rel_y)) - self._clearance_margin(cov)

    def _hazard_from_ttc(self, ttc: float) -> float:
        safe = max(float(self.cfg["ttc_safe_time"]), 1e-6)
        horizon = max(float(self.cfg["ttc_horizon"]), safe)
        if not np.isfinite(ttc):
            return 0.0
        if ttc <= 0.0:
            return 1.0
        if ttc >= horizon:
            return 0.0
        return float(np.clip((safe - ttc) / safe, 0.0, 1.0))

    # -------------------------------------------------------------- read-out
    def estimate(self, ego_xy, ego_heading: float, ego_speed: float, sv_modes) -> dict:
        """Front/rear TTC hazard for the current ego state.

        ``sv_modes`` is the per-vehicle list produced by
        :meth:`GaussianMixtureBelief.modes_with_velocity` (or the same structure
        with 2-D means).
        """
        ego = np.asarray(ego_xy, dtype=float).reshape(2)
        h = float(ego_heading)
        c, s = math.cos(h), math.sin(h)
        ego_vel = float(ego_speed) * np.array([c, s])
        length = float(self.cfg["ttc_vehicle_length"])
        proximity_range = max(float(self.cfg["ttc_proximity_range"]), 1e-6)

        front_hazard = 0.0
        rear_hazard = 0.0
        proximity = 0.0
        ttc_min = float("inf")

        for modes in sv_modes:
            for mean, cov2, prob in modes:
                xy, vxy, cov, p = self._split_mode(mean, cov2, prob)
                if p <= 0.0:
                    continue
                delta = xy - ego
                rel = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
                if self._lateral_clearance(rel[1], cov) > 0.0:
                    continue
                rel_v = np.array([c * (vxy[0] - ego_vel[0]) + s * (vxy[1] - ego_vel[1]),
                                  -s * (vxy[0] - ego_vel[0]) + c * (vxy[1] - ego_vel[1])])

                if rel[0] >= 0.0:
                    gap = float(rel[0]) - length
                    closing = -float(rel_v[0])
                    proximity = max(proximity, p * max(0.0, 1.0 - max(gap, 0.0) / proximity_range))
                    if gap <= 0.0:
                        front_hazard = max(front_hazard, p)
                        ttc_min = 0.0
                    elif closing > 1e-6:
                        ttc = gap / closing
                        if ttc < float(self.cfg["ttc_horizon"]):
                            ttc_min = min(ttc_min, ttc)
                        front_hazard = max(front_hazard, p * self._hazard_from_ttc(ttc))
                else:
                    gap = -float(rel[0]) - length
                    closing = float(rel_v[0])
                    if gap <= 0.0:
                        rear_hazard = max(rear_hazard, p)
                    elif closing > 1e-6:
                        rear_hazard = max(rear_hazard, p * self._hazard_from_ttc(gap / closing))

        return {
            "ttc_hazard": float(max(front_hazard, rear_hazard)),
            "ttc_front_hazard": float(front_hazard),
            "ttc_rear_hazard": float(rear_hazard),
            "ttc_proximity": float(proximity),
            "ttc_min": float(ttc_min),
        }

    # ------------------------------------------------- action-conditioned risk
    def _pose_risk(self, ego_xy, ego_heading: float, sv_xy, margin: float, prob: float) -> float:
        """Hazard of one predicted ego pose against one traffic mode.

        ``margin`` is :meth:`_clearance_margin` for that mode (constant over the
        rollout), so the caller can hoist it out of the step loop.
        """
        h = float(ego_heading)
        c, s = math.cos(h), math.sin(h)
        delta = np.asarray(sv_xy, dtype=float) - np.asarray(ego_xy, dtype=float)
        rel_y = -s * delta[0] + c * delta[1]
        if abs(float(rel_y)) - margin > 0.0:
            return 0.0
        rel_x = c * delta[0] + s * delta[1]
        lon_clear = abs(float(rel_x)) - float(self.cfg["ttc_vehicle_length"])
        if lon_clear <= 0.0:
            return float(prob)
        return float(prob * math.exp(-lon_clear / max(float(self.cfg["ttc_action_scale"]), 1e-6)))

    def action_risks(self, ego_state, action_table, sv_modes,
                     horizon: float = 1.0, dt: float = 0.1) -> dict:
        """Per-action collision risk over a short ego rollout.

        ``ego_state`` is ``(x, y, heading, speed)``; ``action_table`` rows are
        ``(steering fraction, throttle fraction)`` exactly like
        :data:`autodrive.env.ACTION_TABLE`.  A straight action in a free lane is
        returned as exactly ``0.0`` because the lateral gate never opens.
        """
        x, y, heading, speed = (float(v) for v in ego_state)
        actions = np.asarray(action_table, dtype=float)
        steps = max(1, int(round(float(horizon) / max(float(dt), 1e-6))))

        # One row per traffic mode.  Keeping the modes in arrays lets the
        # per-pose check below run as a single vector operation instead of one
        # Python call per mode (same arithmetic, ~3x faster training steps).
        traf_xy, traf_vxy, traf_margin, traf_p = [], [], [], []
        for modes in sv_modes:
            for mean, cov2, prob in modes:
                xy, vxy, cov, p = self._split_mode(mean, cov2, prob)
                if p > 0.0:
                    traf_xy.append(xy)
                    traf_vxy.append(vxy)
                    traf_margin.append(self._clearance_margin(cov))
                    traf_p.append(p)

        risks = np.zeros(len(actions), dtype=float)
        if not traf_xy:
            return {"risk": risks}
        traf_xy = np.asarray(traf_xy, dtype=float)          # (M, 2)
        traf_vxy = np.asarray(traf_vxy, dtype=float)        # (M, 2)
        traf_margin = np.asarray(traf_margin, dtype=float)  # (M,)
        traf_p = np.asarray(traf_p, dtype=float)            # (M,)
        length = float(self.cfg["ttc_vehicle_length"])
        action_scale = max(float(self.cfg["ttc_action_scale"]), 1e-6)

        steer_max = float(self.cfg["steer_max"])
        wheelbase = max(float(self.cfg["wheelbase"]), 1e-6)
        accel_max = float(self.cfg["accel_max"])
        brake_max = float(self.cfg["brake_max"])
        drag = float(self.cfg["drag"])
        v_max = float(self.cfg["v_max"])
        a_lat_max = float(self.cfg["a_lat_max"])

        for ai, (steer_frac, thr_frac) in enumerate(actions):
            ex, ey, eh, ev = x, y, heading, speed
            delta = float(steer_frac) * steer_max
            accel = float(thr_frac) * (accel_max if thr_frac >= 0.0 else brake_max)
            worst = 0.0
            for k in range(steps):
                ev = float(np.clip(ev + (accel - drag * ev) * dt, 0.0, v_max))
                omega_cmd = ev / wheelbase * math.tan(delta)
                omega_max = a_lat_max / max(ev, 1.0)
                omega = float(np.clip(omega_cmd, -omega_max, omega_max))
                eh_mid = eh + 0.5 * omega * dt
                ex += ev * math.cos(eh_mid) * dt
                ey += ev * math.sin(eh_mid) * dt
                eh += omega * dt

                c, s = math.cos(eh), math.sin(eh)
                d = traf_xy + traf_vxy * ((k + 1) * float(dt)) - np.array([ex, ey])
                rel_x = c * d[:, 0] + s * d[:, 1]
                rel_y = -s * d[:, 0] + c * d[:, 1]
                lon_clear = np.abs(rel_x) - length
                risk = np.where(lon_clear <= 0.0, traf_p,
                                traf_p * np.exp(-np.maximum(lon_clear, 0.0) / action_scale))
                risk = np.where(np.abs(rel_y) - traf_margin > 0.0, 0.0, risk)
                worst = max(worst, float(risk.max()))
                if worst >= 1.0:
                    break
            risks[ai] = worst
        return {"risk": risks}


class GaussianMixtureBelief:
    """Multimodal constant-velocity belief over surrounding vehicles."""

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT_RISK_CONFIG)
        self.cfg.update(cfg or {})
        self.num_modes = int(self.cfg["num_modes"])
        self.q = np.eye(4, dtype=float) * float(self.cfg["process_noise"])
        self.r = np.eye(4, dtype=float) * float(self.cfg["measurement_noise"])
        self.min_p = float(self.cfg["min_mode_probability"])
        # lateral-manoeuvre hypotheses (m/s), same set as the reference repo
        self.mode_vlat = np.array([0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5], dtype=float)[: self.num_modes]
        self.transition = self._transition_matrix(self.num_modes)
        self._beliefs = {}
        self._order = []

    def _transition_matrix(self, n: int) -> np.ndarray:
        t = np.full((n, n), 0.05 / max(1, n - 1), dtype=float)
        np.fill_diagonal(t, 0.95)
        return t / t.sum(axis=1, keepdims=True)

    def reset(self):
        self._beliefs.clear()
        self._order = []

    @property
    def empty(self) -> bool:
        return not self._beliefs

    # ------------------------------------------------------------------ update
    def update(self, measurements: dict, dt: float):
        """``measurements``: ``{veh_id: (xy, vxy, lateral_unit, speed)}``."""
        alive = set(measurements)
        for veh_id in list(self._beliefs):
            if veh_id not in alive:
                del self._beliefs[veh_id]
        self._order = list(measurements)

        M = self.num_modes
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

        for veh_id, (xy, vxy, lat_unit, speed) in measurements.items():
            xy = np.asarray(xy, dtype=float).reshape(2)
            vxy = np.asarray(vxy, dtype=float).reshape(2)
            lat_unit = np.asarray(lat_unit, dtype=float).reshape(2)
            z = np.array([xy[0], xy[1], vxy[0], vxy[1]], dtype=float)

            bel = self._beliefs.get(veh_id)
            if bel is None:
                mean = np.tile(z, (M, 1))
                mean[:, 2:] += self.mode_vlat[:, None] * lat_unit[None, :]
                cov = np.tile(np.eye(4) * float(self.cfg["measurement_noise"]), (M, 1, 1))
                self._beliefs[veh_id] = {
                    "mean": mean, "cov": cov, "prob": np.full(M, 1.0 / M),
                    "lat": lat_unit, "speed": float(speed),
                }
                continue

            mean, cov, prob = bel["mean"], bel["cov"], bel["prob"]

            # ---- IMM mixing --------------------------------------------------
            mixed_p = prob @ self.transition
            w = prob[:, None] * self.transition
            w = w / np.maximum(w.sum(axis=0, keepdims=True), self.min_p)
            mixed_mean = w.T @ mean
            delta = mean[:, None, :] - mixed_mean[None, :, :]
            mixed_cov = (np.einsum("ik,ilm->klm", w, cov)
                         + np.einsum("ik,ikp,ikq->kpq", w, delta, delta))

            # ---- predict -----------------------------------------------------
            mean = mixed_mean @ F.T
            mean[:, 2:] += 0.1 * self.mode_vlat[:, None] * dt * lat_unit[None, :]
            cov = F @ mixed_cov @ F.T + self.q

            # ---- measurement update (vectorised Kalman) ----------------------
            S = cov + self.r
            Sinv = np.linalg.inv(S)
            K = cov @ Sinv
            innov = z - mean
            mean = mean + np.einsum("kij,kj->ki", K, innov)
            cov = cov - np.einsum("kij,kjl->kil", K, cov)
            lik = np.exp(-0.5 * np.einsum("ki,ki->k", innov, np.einsum("kij,kj->ki", Sinv, innov)))
            maneuver = np.exp(-0.5 * np.abs(z[3] - mean[:, 3]))
            prob = np.maximum(mixed_p * lik * maneuver, self.min_p)
            prob = prob / prob.sum()

            bel["mean"], bel["cov"], bel["prob"] = mean, cov, prob
            bel["lat"], bel["speed"] = lat_unit, float(speed)

    # ---------------------------------------------------------------- read-out
    def modes_for(self, veh_ids):
        """``[(mean_xy, cov2, prob)]`` per vehicle, ordered as ``veh_ids``."""
        out = []
        for vid in veh_ids:
            bel = self._beliefs.get(vid)
            if bel is None:
                out.append([])
                continue
            out.append([(bel["mean"][k, :2], bel["cov"][k, :2, :2], float(bel["prob"][k]))
                        for k in range(self.num_modes)])
        return out

    def modes_with_velocity(self, veh_ids):
        """Same as :meth:`modes_for` but with the full ``[x, y, vx, vy]`` mean.

        ``TTCCollisionRisk`` needs the velocity to form a time-to-collision;
        ``TimeVaryingRiskField.vehicle_risk`` keeps using the 2-D form.
        """
        out = []
        for vid in veh_ids:
            bel = self._beliefs.get(vid)
            if bel is None:
                out.append([])
                continue
            out.append([(bel["mean"][k].copy(), bel["cov"][k, :2, :2], float(bel["prob"][k]))
                        for k in range(self.num_modes)])
        return out

    def features(self, veh_ids, ego_xy, max_vehicles: int = 3,
                 pos_scale: float = 30.0, uncer_scale: float = 2.0, v_scale: float = 14.0) -> np.ndarray:
        """``[rel_x, rel_y, vx, vy, uncertainty]`` per vehicle, zero padded and scaled."""
        ego = np.asarray(ego_xy, dtype=float).reshape(2)
        rows = []
        for vid in veh_ids[:max_vehicles]:
            bel = self._beliefs.get(vid)
            if bel is None:
                rows.append(np.zeros(5))
                continue
            p = bel["prob"][:, None]
            mean = (p * bel["mean"]).sum(axis=0)
            unc = float((bel["prob"] * np.trace(bel["cov"], axis1=1, axis2=2)).sum())
            rows.append(np.clip(np.concatenate([(mean[:2] - ego) / pos_scale,
                                                mean[2:] / v_scale,
                                                [unc / uncer_scale]]), -3.0, 3.0))
        while len(rows) < max_vehicles:
            rows.append(np.zeros(5))
        return np.concatenate(rows) if rows else np.zeros(5 * max_vehicles)
