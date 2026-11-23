"""RARL — the risk-aware half of `jingyunWu-23/risk`, without the adversary.

Why this shape
--------------
The reference repository has two halves: a **risk-aware** observation/reward
stack (`rarl/risk/risk_field.py` + `rarl/belief/gm_bayes.py`) and an
**adversarial** training loop (`a/`: an adversary vehicle is trained against the
ego).  This project first ported both; the adversarial vehicle was then removed
by request, because the task is no longer "survive an opponent" but
**"complete the lap as fast as possible without crashing into traffic"**.

So the core method is *risk-aware PPO*:

1. the ego observes, on top of the environment observation, the 4 risk features
   (road / vehicle / total / margin) and the Gaussian-mixture belief features of
   the closest vehicles — both ported from the reference repo;
2. the environment reward (time cost + crash penalty) is additionally penalised
   by ``risk_weight * total_risk``, which is exactly how the reference shapes its
   reward.

The adversary vehicle, the alternating ``adv/ego`` phases and the second policy
are gone; the traffic itself (24 rule-based NPCs) is unchanged.

Optional extensions (all **off by default**)
--------------------------------------------
``hazard_mode="ttc"``
    Replace the static ellipse used for shaping with
    :class:`autodrive.risk.TTCCollisionRisk` (longitudinal time-to-collision).
    ``ttc_features`` appends 2 features (hazard, proximity); it defaults to
    ``True`` inside TTC mode, so the policy input grows by 2.  The legacy
    ``"field"`` mode keeps the 48-dim layout.
``action_risk``
    Append the per-action risk vector (``n_actions`` dims) and optionally mask
    actions whose predicted risk exceeds ``action_risk_mask_threshold``.
``safety_shield``
    Hard action mask: a high front TTC hazard removes every non-braking action,
    a high rear hazard removes braking.
``lane_aware``
    Append 9 lane features (front/rear gaps for left/ego/right, the chosen
    direction, the best gap and an availability flag) and pay
    ``lane_change_bonus`` once, when a lane change completes **and passes
    verification** (planned direction, clear destination, no immediate repeat).
    ``safe_progress_weight`` then pays for verified safe forward motion, boosted
    for ``overtake_window_steps`` after a completion until the car that blocked
    the old lane is behind the ego.
``overtake_lock``
    While an open adjacent lane exists, force the initiating hard-steer action
    until the lane change completes, then enter a cooldown.  The safety shield
    can veto that steer, in which case the lock is abandoned and braking takes
    over; ``overtake_lock_steps`` bounds how long a stuck lock may persist.
``overtake_state_machine`` (default **off**)
    Route ``lane_aware`` and ``overtake_lock`` through
    :class:`autodrive.overtake.OvertakeController`: FOLLOW -> PREPARE_LEFT /
    PREPARE_RIGHT -> COMPLETE, with ABORT when the target lane's safe window /
    TTC drop, the lead clears, or the manoeuvre times out.  The state machine
    output replaces the per-step target choice; the observation width is
    unchanged, so a checkpoint from the same feature flags still loads.  The
    positive reward is paid only for a *safe* COMPLETE and for verified safe
    forward progress inside the pass window it opens -- never for taking an
    action or merely crossing a lane line.

``lane_clearance_features`` (default **off**) appends 6 features describing the
front/rear NPC gap in the ego lane and in the two adjacent lanes.  It is kept
separate from ``lane_aware`` so the old 48-dim policy input (and every older
RARL checkpoint) still loads.

``sensor_fusion`` (default **off**)
    Append :class:`autodrive.sensors.SensorFusion`'s 25-dim lane-radar block:
    per lane front/rear gap, front/rear closing speed, contact TTC and safe
    window, plus the ego-lead speed difference, closing TTC, safe-to-accelerate
    flag and the planner's target lane / safe window.  The sensor is built from
    the environment's explicit state (arc positions, lane ids, speeds), never
    from camera images.

    The sensor is also the **single source** of lane geometry for
    ``lane_aware`` / ``lane_clearance_features`` / ``overtake_lock`` -- those
    used to re-walk the NPC bookkeeping on their own.  Turning the flag off
    therefore leaves the observation width unchanged.
"""
from __future__ import annotations

import numpy as np

from ..env import ACTION_TABLE
from ..overtake import OvertakeController, OvertakePhase
from ..risk import GaussianMixtureBelief, TTCCollisionRisk, TimeVaryingRiskField
from ..sensors import SENSOR_DIM, SensorFusion
from ..track import wrap_angle
from .base import Algorithm
from .ppo import PPO, DEFAULT_CONFIG as PPO_DEFAULT_CONFIG

DEFAULT_CONFIG = {
    # the learner is exactly the project's PPO, just with a wider observation
    "ppo": dict(PPO_DEFAULT_CONFIG),
    # risk field / belief (see autodrive/risk.py)
    "risk": {},
    "risk_weight": 1.0,          # penalty per unit of **vehicle** risk (the traffic)
    "road_risk_weight": 0.05,    # tiny nudge per unit of road risk (keep off the kerb)
    "risk_vehicles": 4,       # risk is evaluated on the closest N vehicles
    "belief_vehicles": 3,     # belief features fed to the policy
    "belief_track": 6,        # belief is *updated* only for the closest N vehicles
    # lane-level front/rear clearance; off by default so the policy input stays
    # 48-dim and older checkpoints keep loading
    "lane_clearance_features": False,
    "lane_clearance_range": 30.0,   # [m] gaps saturate here, then scale to [0, 1]
    # --- lane radar / sensor fusion (opt-in observation block) --------------
    # The sensor itself is always used internally for lane tactics; this flag
    # only appends its 25-dim feature vector to the policy input.
    "sensor_fusion": False,
    "sensor_range": 40.0,           # [m] gap saturation
    "sensor_v_scale": 16.0,         # [m/s] relative-speed normalisation
    "sensor_ttc_horizon": 6.0,      # [s] TTC / safe-window saturation
    "sensor_vehicle_extent": 4.3,   # [m] combined half lengths for contact TTC
    "sensor_min_front_gap": 6.0,    # [m] room needed ahead to call a lane usable
    "sensor_min_rear_gap": 5.0,     # [m] room needed behind
    "sensor_accel_min_gap": 12.0,   # [m] lead gap needed for safe acceleration
    "sensor_accel_max_closing": 2.0,  # [m/s] faster closing is not safe
    "sensor_accel_min_ttc": 1.0,    # [s] contact TTC needed for safe acceleration
    # --- TTC hazard mode (opt-in) -------------------------------------------
    "hazard_mode": "field",       # "field" (legacy) | "ttc"
    "ttc_features": True,         # append [hazard, proximity] in TTC mode
    "action_risk": False,         # append per-action risk + optional mask
    "action_risk_horizon": 1.0,   # [s] ego rollout used for the action risks
    "action_risk_mask_threshold": 0.25,
    "action_risk_weight": 0.05,   # reward penalty on the chosen action's risk
    "dense_risk_weight": 0.0,     # residual weight on ellipse risk in TTC mode
    "dense_risk_end": 0.0,
    "dense_risk_decay_steps": 0,
    # --- safety shield / lane tactics (opt-in) ------------------------------
    "safety_shield": False,
    "shield_hazard_threshold": 0.5,
    "lane_aware": False,
    "lane_block_distance": 18.0,  # [m] ego lane is "blocked" below this gap
    "lane_advantage_margin": 5.0,  # [m] adjacent lane must be this much better
    "overtake_side_clearance": 7.0,  # [m] reject a lane with a car beside the ego
    "overtake_lock": False,
    "overtake_lock_pause_in_window": False,  # do not chain a new lock during the pass window
    "overtake_lock_min_speed": 0.0,  # [m/s] never force a swerve at a crawl
    "overtake_cooldown_steps": 20,
    "overtake_lock_steps": 0,      # 0 = no limit; otherwise abandon a stuck lock
    # --- sensor-driven overtake state machine (opt-in) ----------------------
    # When on, lane_aware / overtake_lock consume OvertakeController output
    # instead of the per-step lane planner.  No observation dims change.
    "overtake_state_machine": False,
    "overtake_opportunity_gap": 18.0,       # [m] blocked lead closer than this
    "overtake_min_closing": 0.5,            # [m/s] ego must be faster by this
    "overtake_release_gap": 30.0,           # [m] lead no longer worth passing
    "overtake_entry_min_front": 12.0,       # [m] target-lane room ahead
    "overtake_entry_min_rear": 8.0,         # [m] target-lane room behind
    "overtake_entry_min_window": 2.0,       # [s] target lane safe window to start
    "overtake_entry_min_front_ttc": 1.5,    # [s]
    "overtake_entry_min_rear_ttc": 2.0,     # [s]
    "overtake_abort_min_window": 1.0,       # [s] abort once below this
    "overtake_abort_min_front_ttc": 0.8,    # [s]
    "overtake_abort_min_rear_ttc": 1.0,     # [s]
    "overtake_complete_min_window": 1.5,    # [s] still safe when completing
    "overtake_complete_min_front_ttc": 1.0,  # [s]
    "overtake_complete_min_rear_ttc": 1.0,  # [s]
    "overtake_complete_hold_steps": 3,      # steps in the target lane
    "overtake_complete_hold_after": 3,      # steps reported as COMPLETE
    "overtake_prepare_timeout_steps": 60,
    "overtake_abort_hold_steps": 20,
    "overtake_preferred_side": "auto",      # auto | left | right
    "overtake_complete_bonus": 3.0,         # one-off, safe COMPLETE only
    "lane_change_bonus": 1.0,      # paid once per *verified* planned lane change
    # When true the bonus is deferred until the car that blocked the old lane is
    # actually behind the ego ("the overtake", not just "the lane change").  It
    # closes the degenerate optimum of crawling 10 m, crossing, collecting the
    # bonus and stopping.
    "lane_change_bonus_requires_pass": False,
    "lane_change_min_progress": 0.0,    # [m] arc progress required between two bonuses
    "lane_change_min_front_gap": 0.0,   # [m] destination must still be clear ahead
    # --- safe forward progress (opt-in; 0 = off) ----------------------------
    # The environment reward is +0.02 per metre and -0.1 per step, while the TTC
    # penalty is up to -1.0 per step, so standing still used to be the local
    # optimum.  These two terms pay for *verified safe* forward motion only:
    # positive arc progress, no front hazard above ``safe_progress_max_hazard``,
    # and no action the risk mask would call dangerous.
    "safe_progress_weight": 0.0,        # reward per metre of safe progress
    "safe_speed_target": 0.0,           # [m/s] progress above this speed pays nothing
    "safe_progress_max_hazard": 0.2,    # front TTC hazard that still counts as "safe"
    # Geometric fallback for the gate above: the TTC hazard is a *probability*
    # weighted by the belief's mode weight (~1/7 per mode), so a single vehicle
    # saturates near 0.14 and the configured shield threshold of 0.5 is never
    # reached.  A centre-to-centre gap in the ego lane is unambiguous, so it is
    # what actually keeps the progress reward from paying for tailgating.
    "safe_progress_min_gap": 0.0,       # [m] no progress reward closer than this
    "overtake_window_steps": 0,         # steps of boosted progress after a completion
    "overtake_pass_margin": 3.0,        # [m] blocker behind the ego == overtake done
    "overtake_after_pass_steps": 0,     # steps of progress reward *after* the pass
    "overtake_progress_multiplier": 1.0,  # progress weight inside the window
    # When true the positive progress term is only paid inside the window opened
    # by a verified lane change, i.e. for progress *after* an overtake.  The
    # one-off lane-change bonus is already completion-only, and nothing pays for
    # taking an action.
    "progress_reward_only_in_overtake_window": False,
}

# lane rows: [left adjacent, ego, right adjacent] x [front gap, rear gap]
LANE_CLEARANCE_ROWS = 3
LANE_CLEARANCE_PER_ROW = 2
LANE_CLEARANCE_DIM = LANE_CLEARANCE_ROWS * LANE_CLEARANCE_PER_ROW

# lane-aware block: 6 gaps + direction + best gap + availability flag
LANE_AWARE_DIM = 9


class RARL(Algorithm):
    """Risk-aware PPO (see module docstring)."""

    name = "rarl"

    def __init__(self, obs_dim, n_actions, cfg=None, seed=0):
        c = dict(DEFAULT_CONFIG)
        for k, v in (cfg or {}).items():
            if k == "ppo" and isinstance(v, dict):
                merged = dict(c["ppo"])
                merged.update(v)
                c["ppo"] = merged
            else:
                c[k] = v
        super().__init__(obs_dim, n_actions, c, seed)
        self.cfg = c

        self.belief = GaussianMixtureBelief(c["risk"])
        self.risk_field = TimeVaryingRiskField(c["risk"])
        self.ttc_risk = TTCCollisionRisk(c["risk"])
        self.n_belief = int(c["belief_vehicles"])
        self.n_risk = int(c["risk_vehicles"])
        self.n_track = int(c["belief_track"])
        self.use_lane_clearance = bool(c["lane_clearance_features"])
        self.lane_clearance_range = float(c["lane_clearance_range"])
        self.n_lane_features = LANE_CLEARANCE_DIM if self.use_lane_clearance else 0

        # One sensor, one reading of the traffic.  ``lane_aware``,
        # ``lane_clearance_features`` and ``overtake_lock`` all consume this
        # frame, so the observation, the reward gate and the shield cannot
        # disagree about the same gap.
        self.sensor = SensorFusion({
            "range": c["sensor_range"],
            "v_scale": c["sensor_v_scale"],
            "ttc_horizon": c["sensor_ttc_horizon"],
            "vehicle_extent": c["sensor_vehicle_extent"],
            "min_front_gap": c["sensor_min_front_gap"],
            "min_rear_gap": c["sensor_min_rear_gap"],
            "min_side_gap": c["overtake_side_clearance"],
            "accel_min_gap": c["sensor_accel_min_gap"],
            "accel_max_closing": c["sensor_accel_max_closing"],
            "accel_min_ttc": c["sensor_accel_min_ttc"],
            "lane_advantage_margin": c["lane_advantage_margin"],
            "block_distance": c["lane_block_distance"],
        })
        self.use_sensor = bool(c["sensor_fusion"])
        self.n_sensor = SENSOR_DIM if self.use_sensor else 0
        self.last_sensor = None

        self.hazard_mode = str(c["hazard_mode"]).lower()
        if self.hazard_mode not in ("field", "ttc"):
            raise ValueError(f"unknown hazard_mode '{c['hazard_mode']}' (field | ttc)")
        self.use_ttc = self.hazard_mode == "ttc"
        self.use_ttc_features = self.use_ttc and bool(c["ttc_features"])
        self.use_action_risk = bool(c["action_risk"])
        self.use_shield = bool(c["safety_shield"])
        self.use_lane_aware = bool(c["lane_aware"])
        self.use_overtake_lock = bool(c["overtake_lock"])
        self.use_overtake_sm = bool(c["overtake_state_machine"])
        if self.use_overtake_sm and not self.use_lane_aware:
            raise ValueError("overtake_state_machine requires lane_aware=True: the "
                             "state machine replaces the lane planner's target choice")
        self.overtake = None
        if self.use_overtake_sm:
            self.overtake = OvertakeController({
                "opportunity_gap": c["overtake_opportunity_gap"],
                "min_closing": c["overtake_min_closing"],
                "release_gap": c["overtake_release_gap"],
                "entry_min_front": c["overtake_entry_min_front"],
                "entry_min_rear": c["overtake_entry_min_rear"],
                "entry_min_window": c["overtake_entry_min_window"],
                "entry_min_front_ttc": c["overtake_entry_min_front_ttc"],
                "entry_min_rear_ttc": c["overtake_entry_min_rear_ttc"],
                "abort_min_window": c["overtake_abort_min_window"],
                "abort_min_front_ttc": c["overtake_abort_min_front_ttc"],
                "abort_min_rear_ttc": c["overtake_abort_min_rear_ttc"],
                "complete_min_window": c["overtake_complete_min_window"],
                "complete_min_front_ttc": c["overtake_complete_min_front_ttc"],
                "complete_min_rear_ttc": c["overtake_complete_min_rear_ttc"],
                "complete_hold_steps": c["overtake_complete_hold_steps"],
                "complete_hold_after": c["overtake_complete_hold_after"],
                "prepare_timeout_steps": c["overtake_prepare_timeout_steps"],
                "abort_hold_steps": c["overtake_abort_hold_steps"],
                "pass_margin": c["overtake_pass_margin"],
                "preferred_side": c["overtake_preferred_side"],
            })
        self.n_ttc_features = 2 if self.use_ttc_features else 0
        self.n_action_risk = n_actions if self.use_action_risk else 0
        self.n_lane_aware = LANE_AWARE_DIM if self.use_lane_aware else 0

        # policy input: env observation + 4 risk features + optional TTC /
        # action-risk / sensor-fusion / lane-aware blocks + 5 belief features
        # per vehicle (+ 6 lane-clearance features when explicitly enabled)
        self.policy_in = (obs_dim + 4 + self.n_ttc_features + self.n_action_risk
                          + self.n_sensor + self.n_lane_aware + self.n_lane_features
                          + 5 * self.n_belief)
        ppo_cfg = dict(c["ppo"])
        if "bc_lr" in c:
            # a top-level ``bc_lr`` is the warm-start learning rate; forward it to
            # the PPO config where ``pretrain`` reads it (explicit ppo.bc_lr wins)
            ppo_cfg.setdefault("bc_lr", float(c["bc_lr"]))
        self.policy = PPO(self.policy_in, n_actions, cfg=ppo_cfg, seed=seed)

        self.env = None
        self.last_risk = {}
        self.last_action_risk = np.zeros(n_actions, dtype=float)
        self.last_lane = {}
        self._obs = None
        self._feature_obs = None
        self._feature_cache = None
        self._lane_taken = {}
        self._lane_action = None
        self._lane_bonus_taken = 0.0
        self._action_risk_taken = 0.0
        self._overtake_target = None
        self._overtake_cooldown = 0
        self._lock_step = 0
        self._last_bonus_s = -np.inf
        self._overtake_window = 0
        self._overtake_blocker = None
        self._overtake_passed = False
        self._lane_bonus_pending = 0.0
        self._progress_reward = 0.0
        self._episode_progress = 0.0
        self.overtake_completions = 0
        self._sm_command = None
        self._sm_key = None
        self._sm_bonus_ready = 0.0
        self._sm_complete_step = None
        self.overtake_aborts = 0

    # ------------------------------------------------------------------- wiring
    def bind_env(self, env):
        self.env = env
        self._feature_obs = None
        self._feature_cache = None

    # --------------------------------------------------------------- observation
    def _measurements(self, ids):
        env = self.env
        ob = env.obstacles
        meas = {}
        for i in ids:
            idx = int(ob._idx[i])
            meas[i] = (ob.pos[i], env.track.tangents[idx] * ob.v[i],
                       env.track.normals[idx], float(ob.v[i]))
        return meas

    def _nearest_ids(self, k):
        env = self.env
        ob = env.obstacles
        if ob.n == 0:
            return []
        d = np.linalg.norm(ob.pos - np.array([[env.x, env.y]]), axis=1)
        return [int(i) for i in np.argsort(d)[:k]]

    def _scan(self, frame=None):
        """The current :class:`SensorFrame` (scanned on demand, never mutated)."""
        if self.env is None:
            raise RuntimeError("RARL lane tactics need bind_env() before scanning traffic")
        return self.sensor.scan(self.env) if frame is None else frame

    # ---------------------------------------------------- overtake state machine
    def _sm_signature(self, frame):
        """Cheap fingerprint of the sensor facts that drive the state machine.

        One control step must advance the controller exactly once, no matter
        whether the observation, the reward gate or the action mask asks first.
        The fingerprint includes the control step and the lane readings, so a
        test that moves traffic without stepping the simulator is still seen.
        """
        rows = tuple(
            (int(r.lane), bool(r.exists),
             round(float(r.front_gap), 4), round(float(r.rear_gap), 4),
             round(float(r.front_ttc), 4), round(float(r.rear_ttc), 4),
             bool(r.safe_to_enter))
            for r in frame.rows
        )
        ids = tuple(int(x) for x in np.asarray(frame.front_ids).reshape(-1))
        steps = int(self.env.steps) if self.env is not None else -1
        return (steps, int(frame.ego_lane), round(float(frame.ego_speed), 4), ids, rows)

    def _overtake_step(self, frame=None):
        """Advance the overtake controller once for this control step."""
        if not self.use_overtake_sm or self.overtake is None:
            return None
        frame = self._scan(frame)
        key = self._sm_signature(frame)
        if key == self._sm_key and self._sm_command is not None:
            return self._sm_command
        cmd = self.overtake.step(frame)
        self._sm_key = key
        self._sm_command = cmd
        self.last_sensor = frame
        if cmd.event == "complete":
            # safe COMPLETE: the controller only gets here inside the target
            # lane, with the window/TTC gate open and the blocker behind the ego
            self._sm_bonus_ready += float(self.cfg["overtake_complete_bonus"])
            self._overtake_window = int(self.cfg["overtake_after_pass_steps"])
            self._overtake_passed = True
            self.overtake_completions += 1
            self._last_bonus_s = float(self.env.s_total) if self.env is not None else 0.0
            self._sm_complete_step = int(self.env.steps) if self.env is not None else None
        elif cmd.event == "abort":
            self.overtake_aborts += 1
            self._overtake_cooldown = int(self.cfg["overtake_abort_hold_steps"])
        return cmd

    def _sm_veto(self, reason: str) -> None:
        """The shield rejected the forced steer: abort instead of swerving."""
        if self.overtake is None:
            return
        self.overtake.abort(reason)
        self.overtake_aborts += 1
        self._sm_command = self.overtake.last_command
        self._overtake_target = None
        self._overtake_cooldown = int(self.cfg["overtake_abort_hold_steps"])

    def _sm_hold_action(self, lane: int) -> int:
        """Tactical action that keeps the ego in ``lane`` and lets it pass.

        The state machine owns the manoeuvre, so the lock keeps working after the
        lane index changes: correct the lateral error, then keep the speed up
        while the target lane is clear.  Nothing here pays reward; it only picks
        the action the mask will allow.
        """
        env = self.env
        centre = float(env.lane_centers[int(lane)])
        err = centre - float(env._lateral)
        heading_err = wrap_angle(float(env.theta) - float(env._track_heading))
        # predicted lateral error after a short look-ahead (heading contributes)
        aim = err - float(env.v) * np.sin(heading_err) * 0.5
        if aim > 1.0:
            return 0                                  # far right of centre -> hard left
        if aim > 0.35:
            return 1                                  # slightly right -> soft left
        if aim < -1.0:
            return 4                                  # far left of centre -> hard right
        if aim < -0.35:
            return 3                                  # slightly left -> soft right
        reading = self.last_sensor.reading(int(lane)) if self.last_sensor is not None else None
        if reading is not None and reading.front_gap >= float(self.cfg["sensor_accel_min_gap"]):
            return 5                                  # centred and clear -> accelerate
        return 2                                      # hold speed

    def _lane_gaps(self):
        """Per-lane ``(front, rear, side)`` arc distances to the nearest NPC.

        Thin wrapper over :class:`autodrive.sensors.SensorFusion`; the sensor is
        the only place that walks the NPC bookkeeping.  ``front`` / ``rear``
        follow the modulo-arc convention used everywhere in the simulator, and
        ``side`` is the circular distance to the nearest NPC in that lane
        regardless of direction (a car alongside makes a lane unsafe to enter).
        Empty lanes report ``inf``.
        """
        frame = self._scan()
        return frame.front_gaps, frame.rear_gaps, frame.side_gaps

    def _lane_lead_id(self, lane: int) -> int | None:
        """Id of the nearest NPC ahead of the ego in ``lane`` (``None`` if empty)."""
        idx = int(self._scan().front_ids[int(lane)])
        return idx if idx >= 0 else None

    def _blocker_is_passed(self) -> bool:
        """Whether the car that blocked the old lane is now behind the ego."""
        env = self.env
        ob = env.obstacles
        b = self._overtake_blocker
        if b is None or not 0 <= int(b) < ob.n:
            return True
        L = float(env.track.length)
        gap = (float(ob.s[int(b)]) - float(env.prev_s)) % L
        if gap > L / 2.0:                      # measured backwards -> behind
            gap -= L
        return gap < -float(self.cfg["overtake_pass_margin"])

    def _lane_clearance(self, frame=None) -> np.ndarray:
        """Front/rear NPC gap per lane, relative to the ego lane (6 features).

        Rows are ``[left adjacent, ego, right adjacent]`` — lane index ``+1 / 0 / -1``,
        because ``lane_centers`` grows towards the left of the driving direction —
        and each row is ``[front gap, rear gap]`` in arc length along the track.

        Gaps are clipped at ``lane_clearance_range`` and divided by it, so 1.0
        means "free for at least that range" (also what an empty lane reports).
        A lane that does not exist on this road (2-lane tracks, road edge) is
        encoded as -1.0; every other value is in [0, 1].

        The distances come straight from the sensor frame, so this block and the
        ``lane_aware`` block can never disagree about the same gap.
        """
        frame = self._scan(frame)
        scale = max(self.lane_clearance_range, 1e-6)
        out = np.full((LANE_CLEARANCE_ROWS, LANE_CLEARANCE_PER_ROW), -1.0)
        for row, reading in enumerate(frame.rows):
            if not reading.exists:
                continue
            out[row, 0] = min(reading.front_gap, scale) / scale
            out[row, 1] = min(reading.rear_gap, scale) / scale
        return out.reshape(-1)

    def _lane_gap_values(self, frame):
        """Normalised front/rear gap pairs for the three lanes, plus the scale used.

        Missing lanes contribute ``[0.0, 0.0]`` (the ``lane_aware`` convention,
        distinct from the ``-1.0`` used by ``_lane_clearance``).
        """
        scale = max(self.lane_clearance_range, 1e-6)
        values = []
        for reading in frame.rows:
            if reading.exists:
                values.extend([min(reading.front_gap, scale) / scale,
                               min(reading.rear_gap, scale) / scale])
            else:
                values.extend([0.0, 0.0])
        return values, scale

    def _lane_features(self, frame=None) -> np.ndarray:
        """9 features describing the lane-choice situation (``lane_aware``).

        ``[left front, left rear, ego front, ego rear, right front, right rear,
        direction, best gap, overtake available]``.  The last entry is exactly
        1.0 when a safe adjacent lane is open, 0.0 otherwise.

        The candidate (``best_lane`` / ``direction`` / ``available``) is decided
        by :class:`autodrive.sensors.SensorFusion` from the same readings, so the
        observation and ``overtake_lock`` act on one consistent picture.
        """
        env = self.env
        frame = self._scan(frame)
        if self.use_overtake_sm:
            return self._sm_lane_features(frame)
        values, scale = self._lane_gap_values(frame)

        ego_lane = int(frame.ego_lane)
        available = bool(frame.target_available)
        best_lane = int(frame.best_lane)
        best_gap = float(frame.target_gap)

        self.last_lane = {
            "front": frame.front_gaps.copy(),
            "rear": frame.rear_gaps.copy(),
            "lane_side_clearance": frame.side_gaps.copy(),
            "ego_lane": ego_lane,
            "best_lane": best_lane,
            "direction": int(frame.direction),
            "overtake_available": available,
            "current_gap": float(frame.ego_lead_gap),
            "best_gap": best_gap,
            "ego_lead_ttc": float(frame.ego_lead_ttc),
            "ego_lead_speed_delta": float(frame.ego_lead_speed_delta),
            "target_safe_window": float(frame.target_safe_window),
            "safe_to_accelerate": bool(frame.safe_to_accelerate),
            # the car that currently blocks the ego lane; the pass window opened
            # after a verified lane change closes once this one is behind us
            "blocker": int(frame.front_ids[ego_lane]) if available
            and int(frame.front_ids[ego_lane]) >= 0 else None,
            "overtake_window": bool(self._overtake_window > 0),
            "overtake_passed": bool(self._overtake_passed),
        }
        pause = bool(self.cfg["overtake_lock_pause_in_window"]) and self._overtake_window > 0
        rolling = float(env.v) >= float(self.cfg["overtake_lock_min_speed"])
        if self.use_overtake_lock and available and rolling and self._overtake_cooldown <= 0 \
                and self._overtake_target is None and not pause:
            self._overtake_target = best_lane
            self._lock_step = int(env.steps)

        return np.array(values + [float(frame.direction), min(best_gap, scale) / scale,
                                  1.0 if available else 0.0], dtype=float)

    def _sm_lane_features(self, frame) -> np.ndarray:
        """The 9 ``lane_aware`` features, with the target chosen by the state machine.

        The layout is unchanged (six gaps + direction + best gap + available
        flag), so every checkpoint built with the same feature flags still
        loads; only the *values* of the last three entries now come from
        :class:`OvertakeController` instead of the one-step planner.
        """
        values, scale = self._lane_gap_values(frame)

        cmd = self._overtake_step(frame)
        ego_lane = int(frame.ego_lane)
        target = int(cmd.target_lane) if cmd.target_lane is not None else None
        if target is not None and 0 <= target < len(frame.front_gaps):
            best_gap = float(frame.front_gaps[target])
        else:
            best_gap = float(frame.ego_lead_gap)
        # keep the legacy stats/checkpoint surfaces coherent with the decision
        self._overtake_target = target if cmd.lock else None
        self.last_lane = {
            "front": frame.front_gaps.copy(),
            "rear": frame.rear_gaps.copy(),
            "lane_side_clearance": frame.side_gaps.copy(),
            "ego_lane": ego_lane,
            "best_lane": int(target) if target is not None else ego_lane,
            "direction": int(cmd.direction),
            "overtake_available": bool(cmd.available),
            "current_gap": float(frame.ego_lead_gap),
            "best_gap": best_gap if np.isfinite(best_gap) else float(self.sensor.range),
            "ego_lead_ttc": float(frame.ego_lead_ttc),
            "ego_lead_speed_delta": float(frame.ego_lead_speed_delta),
            "target_safe_window": float(cmd.target_window),
            "safe_to_accelerate": bool(frame.safe_to_accelerate),
            "blocker": cmd.blocker_id,
            "overtake_window": bool(self._overtake_window > 0),
            "overtake_passed": bool(cmd.blocker_passed),
            "overtake_state": cmd.state,
            "overtake_reason": cmd.reason,
            "overtake_lock": bool(cmd.lock),
        }
        return np.array(values + [float(cmd.direction),
                                  min(best_gap, scale) / scale,
                                  1.0 if cmd.available else 0.0], dtype=float)

    def _ttc_features(self, hazard: dict) -> np.ndarray:
        h = float(hazard.get("ttc_hazard", 0.0))
        p = float(hazard.get("ttc_proximity", 0.0))
        return np.array([h / (1.0 + h), p / (1.0 + p)], dtype=float)

    def _features(self, obs) -> np.ndarray:
        env = self.env
        if env is None:
            raise RuntimeError("RARL._features needs bind_env() before it can read traffic")
        if self._feature_obs is obs and self._feature_cache is not None:
            return self._feature_cache

        frame = self._scan()
        self.last_sensor = frame

        if self.belief.empty:                     # first step of an episode
            self.belief.update(self._measurements(self._nearest_ids(self.n_track)), env.dt)

        ids_risk = self._nearest_ids(self.n_risk)
        field_modes = self.belief.modes_for(ids_risk)
        if self.use_ttc:
            hazard = self.ttc_risk.estimate(
                (env.x, env.y), env.theta, env.v,
                self.belief.modes_with_velocity(ids_risk))
        else:
            hazard = None

        feats = self.risk_field.features(
            (env.x, env.y), env._lateral, env.half_width, env.lane_centers, field_modes)
        self.last_risk = {"total_risk": float(feats[2]), "road_risk": float(feats[0]),
                          "vehicle_risk": float(feats[1])}
        if hazard is not None:
            self.last_risk.update(hazard)

        parts = [obs, feats]
        if self.use_ttc_features:
            parts.append(self._ttc_features(hazard))

        if self.use_action_risk:
            self.last_action_risk = self.ttc_risk.action_risks(
                (env.x, env.y, env.theta, env.v), ACTION_TABLE,
                self.belief.modes_with_velocity(ids_risk),
                horizon=float(self.cfg["action_risk_horizon"]), dt=env.dt)["risk"]
            parts.append(self.last_action_risk)

        if self.use_sensor:
            parts.append(self.sensor.features_from(frame))
        if self.use_lane_aware:
            parts.append(self._lane_features(frame))
        if self.use_lane_clearance:
            parts.append(self._lane_clearance(frame))

        belief_feats = self.belief.features(self._nearest_ids(self.n_belief), (env.x, env.y),
                                            self.n_belief)
        parts.append(belief_feats)

        out = np.concatenate(parts)
        self._feature_obs = obs
        self._feature_cache = out
        return out

    # ------------------------------------------------------------------ acting
    def _shield_blocks_action(self, a: int) -> bool:
        """Whether the safety shield forbids action ``a`` in the current state."""
        if not self.use_shield:
            return False
        threshold = float(self.cfg["shield_hazard_threshold"])
        if self.use_ttc:
            if a != 6 and float(self.last_risk.get("ttc_front_hazard", 0.0)) >= threshold:
                return True                       # no cruise / accelerate / steer-in
            if a == 6 and float(self.last_risk.get("ttc_rear_hazard", 0.0)) >= threshold:
                return True                       # do not brake with a tailgater
        if self.use_action_risk and self.last_action_risk.size == self.n_actions:
            return bool(float(self.last_action_risk[a])
                        >= float(self.cfg["action_risk_mask_threshold"]))
        return False

    def _action_mask(self) -> np.ndarray:
        """Boolean action mask; all-``True`` when no safety feature is enabled."""
        mask = np.ones(self.n_actions, dtype=bool)
        if self._overtake_cooldown > 0:
            self._overtake_cooldown -= 1

        # State-machine lock: the controller owns the commitment.  The shield
        # stays authoritative; a vetoed steer aborts the manoeuvre (ABORT) and
        # leaves the normal mask (braking) in charge.
        if self.use_overtake_sm and self.env is not None:
            cmd = self._overtake_step()
            if cmd is not None and cmd.lock and cmd.target_lane is not None:
                ego_lane = int(self.env.lane_of())
                if int(cmd.target_lane) != ego_lane:
                    forced = 0 if int(cmd.target_lane) > ego_lane else 4
                else:
                    forced = self._sm_hold_action(int(cmd.target_lane))
                if self._shield_blocks_action(forced):
                    self._sm_veto("shield")
                else:
                    mask[:] = False
                    mask[forced] = True
                    return mask

        # Tactical lane change: keep only the initiating hard steer until the
        # ego has actually reached the target lane.  The shield stays
        # authoritative: if it vetoes that steer (car right in front, action
        # predicted to collide) the lock is abandoned and braking takes over.
        if (not self.use_overtake_sm and self.use_overtake_lock
                and self._overtake_target is not None and self.env is not None):
            ego_lane = self.env.lane_of()
            limit = int(self.cfg["overtake_lock_steps"])
            too_slow = float(self.env.v) < float(self.cfg["overtake_lock_min_speed"])
            if too_slow:
                # a forced swerve at walking pace only arcs the car off the road;
                # release the lock and let the policy regain speed first
                self._overtake_target = None
                self._overtake_cooldown = int(self.cfg["overtake_cooldown_steps"])
            elif limit > 0 and int(self.env.steps) - int(self._lock_step) > limit:
                # a lock that never reaches the lane would pin the policy to one
                # steering action; drop it and let the normal mask decide
                self._overtake_target = None
                self._overtake_cooldown = int(self.cfg["overtake_cooldown_steps"])
            elif ego_lane == self._overtake_target:
                self._overtake_target = None
                self._overtake_cooldown = int(self.cfg["overtake_cooldown_steps"])
            else:
                forced = 0 if self._overtake_target > ego_lane else 4
                if self._shield_blocks_action(forced):
                    self._overtake_target = None
                    self._overtake_cooldown = int(self.cfg["overtake_cooldown_steps"])
                else:
                    mask[:] = False
                    mask[forced] = True
                    return mask

        if self.use_shield:
            for a in range(self.n_actions):
                if self._shield_blocks_action(a):
                    mask[a] = False

        if not mask.any():                         # never leave an empty distribution
            if self.use_action_risk and self.last_action_risk.size == self.n_actions:
                mask[int(np.argmin(self.last_action_risk))] = True
            else:
                mask[6] = True
        return mask

    def action_mask(self) -> np.ndarray:
        """The mask ``act`` would apply in the current state (public form).

        With the safety shield or the overtake lock off there is nothing to mask
        and every action is allowed, which keeps behaviour cloning on such a
        configuration identical to the legacy unmasked update.  Calling this once
        per control step is exactly what ``act`` does; the collector uses it to
        store the mask alongside each demonstration.
        """
        if self.use_shield or self.use_overtake_lock:
            return self._action_mask()
        return np.ones(self.n_actions, dtype=bool)

    def act(self, obs, greedy: bool = False) -> int:
        feats = self._features(obs)
        self._obs = feats
        self._lane_taken = dict(self.last_lane) if self.last_lane else {}
        self._lane_bonus_taken = 0.0

        mask = None
        if self.use_shield or self.use_overtake_lock:
            mask = self._action_mask()

        action = self.policy.act(feats, greedy=greedy, action_mask=mask)
        self._lane_action = int(action)
        if self.use_action_risk and self.last_action_risk.size == self.n_actions:
            self._action_risk_taken = float(self.last_action_risk[action])
        return action

    # ------------------------------------------------------------------ hooks
    def _reset_overtake_state(self) -> None:
        """Clear the per-episode lane-tactic state (lock, window, bonus history)."""
        self._overtake_target = None
        self._overtake_cooldown = 0
        self._lock_step = 0
        self._last_bonus_s = -np.inf
        self._overtake_window = 0
        self._overtake_blocker = None
        self._overtake_passed = False
        self._lane_bonus_pending = 0.0
        self._episode_progress = 0.0
        if self.overtake is not None:
            self.overtake.reset()
        self._sm_command = None
        self._sm_key = None
        self._sm_bonus_ready = 0.0
        self._sm_complete_step = None

    def reset_episode(self, obs):
        """Prime the belief from a fresh observation (evaluation hook)."""
        if self.env is not None:
            self.belief.reset()
            ids = self._nearest_ids(self.n_track)
            if ids:
                self.belief.update(self._measurements(ids), self.env.dt)
        self._feature_obs = obs
        self._feature_cache = None
        self.last_lane = {}
        self.last_sensor = None
        self._lane_taken = {}
        self._lane_action = None
        self._lane_bonus_taken = 0.0
        self._action_risk_taken = 0.0
        self._reset_overtake_state()

    def observe_transition(self, next_obs, terminated: bool = False, truncated: bool = False):
        """Advance the belief to ``next_obs`` and settle the lane-change reward."""
        if self.env is not None:
            if terminated or truncated:
                self.belief.reset()
            ids = self._nearest_ids(self.n_track)
            if ids:
                self.belief.update(self._measurements(ids), self.env.dt)
        # same detection as in training, so the reported completion count and the
        # reported reward agree
        self._lane_bonus_taken = self._advance_lane_state()
        self._lane_bonus_pending = 0.0
        self._feature_obs = next_obs
        self._feature_cache = None
        if terminated or truncated:
            self._reset_overtake_state()

    # ---------------------------------------------------------------- learning
    def _dense_weight(self) -> float:
        """Linearly decay ``dense_risk_weight`` -> ``dense_risk_end``."""
        start = float(self.cfg["dense_risk_weight"])
        end = float(self.cfg["dense_risk_end"])
        steps = int(self.cfg["dense_risk_decay_steps"])
        if steps <= 0:
            return start
        frac = min(max(float(self.total_steps) / float(steps), 0.0), 1.0)
        return start + (end - start) * frac

    def _lane_change_is_verified(self, lane: dict, previous: int, current: int) -> bool:
        """Whether the crossing just observed is a *safe, planned* lane change.

        Four independent checks; any one of them failing pays nothing:

        * the direction matches the planner's own choice (``_lane_features``
          already rejected lanes with a car alongside and lanes without a real
          gap advantage, so the target is the lane the agent *wanted*);
        * with ``overtake_lock`` on, the lane is the one the lock committed to;
        * at least ``lane_change_min_progress`` metres of arc progress happened
          since the last paid bonus, which makes weaving back and forth worthless
          (each crossing used to be worth a full bonus);
        * the destination is *still* clear after the crossing -- a car alongside
          or immediately ahead of the new lane means this was a cut-in, not an
          overtake.
        """
        moved = 1 if current > previous else -1
        if not lane.get("overtake_available") or int(lane.get("direction", 0)) != moved:
            return False
        if self.use_overtake_lock and self._overtake_target != current:
            # the lock is the planner's commitment; if it was abandoned (shield
            # veto or deadline) or points at another lane, this is not the
            # manoeuvre we planned
            return False
        if float(self.env.s_total) - float(self._last_bonus_s) \
                < float(self.cfg["lane_change_min_progress"]):
            return False
        front, _rear, side = self._lane_gaps()
        if float(side[current]) < float(self.cfg["overtake_side_clearance"]):
            return False
        if float(front[current]) < float(self.cfg["lane_change_min_front_gap"]):
            return False
        return True

    def _advance_lane_state(self) -> float:
        """Verify a just-completed lane change and keep the pass window current.

        Called exactly once per transition, from :meth:`update` (training) and
        :meth:`observe_transition` (evaluation), so training and evaluation see
        the same completion counter.  Returns the one-off bonus payable for this
        transition (``0.0`` when it was not a verified completion).
        """
        env = self.env
        if env is None:
            return 0.0
        if self.use_overtake_sm:
            # The controller observes the state the transition ended in, so a
            # COMPLETE is only ever reported after the lane change and the
            # actual pass, and only while the target window is still safe.
            cmd = self._overtake_step()
            bonus = float(self._sm_bonus_ready)
            self._sm_bonus_ready = 0.0
            if bonus > 0.0:
                self._lane_bonus_pending = bonus
            if self._overtake_window > 0 and int(env.steps) != self._sm_complete_step:
                self._overtake_window -= 1
            self._lane_taken = {}
            return bonus
        bonus = 0.0
        defer = bool(self.cfg["lane_change_bonus_requires_pass"])
        lane = self._lane_taken or {}
        if "ego_lane" in lane:
            previous = int(lane["ego_lane"])
            current = int(env.lane_of())
            if current != previous and self._lane_change_is_verified(lane, previous, current):
                self._last_bonus_s = float(env.s_total)
                self._overtake_blocker = lane.get("blocker")
                self._overtake_window = int(self.cfg["overtake_window_steps"])
                self._overtake_passed = False
                if not defer:
                    bonus = float(self.cfg["lane_change_bonus"])
                    self.overtake_completions += 1
        if bonus > 0.0:
            self._lane_bonus_pending = bonus        # consumed by shaped_reward()
        elif self._overtake_window > 0:
            if self._overtake_passed:
                # already past the blocker: this is the after-overtake progress
                # window, paid for by _safe_progress_reward
                self._overtake_window -= 1
            elif self._blocker_is_passed():
                self._overtake_passed = True
                self._overtake_blocker = None
                if defer:
                    # the lane change turned into an actual overtake: pay now,
                    # once, and only because the blocker is behind the ego
                    bonus = float(self.cfg["lane_change_bonus"])
                    self.overtake_completions += 1
                    self._last_bonus_s = float(env.s_total)
                    self._lane_bonus_pending = bonus
                self._overtake_window = int(self.cfg["overtake_after_pass_steps"])
            else:
                self._overtake_window -= 1
        # the pre-step planner snapshot is settled; a second call for the same
        # transition must not be able to pay the bonus twice
        self._lane_taken = {}
        return bonus

    def _progress_is_safe(self) -> bool:
        """Whether forward motion right now is verified as safe.

        Two independent gates, both cheap and both decidable from the current
        state:

        * the TTC front hazard / chosen-action risk stay under their thresholds
          (this is the same signal the shield uses);
        * the ego lane has at least ``safe_progress_min_gap`` metres of room to
          the lead car (centre-to-centre, from the simulator's NPC bookkeeping).
          This one matters because the hazard is mode-probability weighted and
          therefore saturates far below its threshold, while a gap does not.
        """
        if self.use_ttc:
            hazard = float(self.last_risk.get("ttc_front_hazard", 0.0))
        else:
            hazard = float(self.last_risk.get("vehicle_risk", 0.0))
        if hazard > float(self.cfg["safe_progress_max_hazard"]):
            return False
        if self.use_action_risk and self.last_action_risk.size == self.n_actions:
            if float(self._action_risk_taken) > float(self.cfg["action_risk_mask_threshold"]):
                return False
        gap = float(self.cfg["safe_progress_min_gap"])
        if gap > 0.0 and self.last_lane:
            front = np.asarray(self.last_lane.get("front", []), dtype=float)
            lane = int(self.last_lane.get("ego_lane", 0))
            if front.size > lane and float(front[lane]) < gap:
                return False
        return True

    def _safe_progress_reward(self, ds: float) -> float:
        """Reward per metre of *verified safe* forward progress.

        ``ds`` is the signed arc progress of the step that just finished, so the
        term can only be collected by actually moving forward along the track --
        not by steering, not by oscillating in a lane.  It is capped at
        ``safe_speed_target`` (no reward for reckless speed) and gated by the
        same TTC front hazard / action risk that the shield uses.  Inside the
        pass window opened by a verified lane change it is multiplied by
        ``overtake_progress_multiplier``: this is what turns "complete a lane
        change" into "complete it *and then* make safe forward progress".
        """
        weight = float(self.cfg["safe_progress_weight"])
        if weight <= 0.0 or float(ds) <= 0.0 or self.env is None:
            return 0.0
        if self.use_overtake_sm and self._overtake_window <= 0:
            # state-machine mode: positive progress is only paid after a safe
            # COMPLETE opened the pass window -- never for ordinary driving and
            # never for merely steering
            return 0.0
        if bool(self.cfg["progress_reward_only_in_overtake_window"]) and self._overtake_window <= 0:
            # no positive progress pay outside the window opened by a verified
            # lane change: the term can only reward driving *away* after a pass
            return 0.0
        target = float(self.cfg["safe_speed_target"])
        if target > 0.0:
            ds = min(float(ds), target * float(self.env.dt))
        if not self._progress_is_safe():
            return 0.0
        multiplier = 1.0
        if self._overtake_window > 0:
            multiplier = float(self.cfg["overtake_progress_multiplier"])
        return weight * float(ds) * multiplier

    def shaped_reward(self, reward: float, ds: float = 0.0) -> float:
        """Environment reward minus the risk term (and other opt-in shaping).

        The weight is on the **vehicle** risk (the traffic, i.e. what actually
        causes collisions); the road term only gets a small nudge so the car does
        not hug the kerb.  In TTC mode the longitudinal hazard replaces the
        vehicle term, with only a residual (optionally decaying) ellipse weight.
        """
        veh = float(self.last_risk.get("vehicle_risk", 0.0))
        road = float(self.last_risk.get("road_risk", 0.0))
        if self.use_ttc:
            risk_term = float(self.last_risk.get("ttc_hazard", 0.0)) + self._dense_weight() * veh
        else:
            risk_term = veh
        shaped = float(reward) - float(self.cfg["risk_weight"]) * risk_term \
            - float(self.cfg["road_risk_weight"]) * road
        if self.use_action_risk:
            shaped -= float(self.cfg["action_risk_weight"]) * float(self._action_risk_taken)
        self._lane_bonus_taken = 0.0
        if self.use_lane_aware:
            self._lane_bonus_taken = float(self._lane_bonus_pending)
            self._lane_bonus_pending = 0.0
            shaped += self._lane_bonus_taken
            step_progress = self._safe_progress_reward(ds)
            self._episode_progress += step_progress
            shaped += step_progress
        return float(shaped)

    def update(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        self.total_steps += 1
        # belief is refreshed on the new state; risk is evaluated there too
        if self._feature_obs is not next_obs:
            ids = self._nearest_ids(self.n_track)
            if ids:
                self.belief.update(self._measurements(ids), self.env.dt)
        # verify the lane change this transition completed *before* the planner
        # is refreshed for the new state
        self._advance_lane_state()
        next_feats = self._features(next_obs)
        if (info or {}).get("collision"):
            # a COMPLETE detected on the same transition as a collision is not a
            # safe overtake and must not pay the one-off bonus
            self._lane_bonus_pending = 0.0
            self._sm_bonus_ready = 0.0
        shaped = self.shaped_reward(reward, ds=float((info or {}).get("ds", 0.0) or 0.0))
        out = self.policy.update(self._obs, action, shaped, next_feats, terminated, truncated)
        if terminated or truncated:
            self.belief.reset()                    # new episode -> new traffic layout
            self._feature_obs = None
            self._feature_cache = None
            # stash the episode total before the state is cleared, so the log
            # shows how much the safe-progress term actually paid
            self._progress_reward = self._episode_progress
            self._reset_overtake_state()
        return out

    # ------------------------------------------------- behaviour cloning
    def pretrain(self, obs, actions, epochs=12, batch_size=256, label_smoothing=0.0,
                 action_mask=None):
        """Warm-start the actor on demonstrations recorded in the *policy* space.

        ``obs`` must already be the RARL policy input (env observation + risk +
        optional sensor / lane blocks + belief), i.e. exactly what
        ``RARL._features`` returns.  The projection itself is PPO's BC, so the
        input width check there is what guards against feeding raw 29-dim env
        observations to a 73/97-dim sensor-fusion policy.

        ``action_mask`` (``[n, n_actions]``) is the safety mask recorded with
        each demonstration; ``PPO.pretrain`` then applies the same masked logits
        the PPO update uses, so the supervision is consistent with the shield.
        """
        out = self.policy.pretrain(obs, actions, epochs=epochs, batch_size=batch_size,
                                   label_smoothing=label_smoothing,
                                   action_mask=action_mask)
        out["bc_policy_in"] = int(self.policy_in)
        return out

    # ------------------------------------------------------------- persistence
    @staticmethod
    def _checkpoint_policy_input(state) -> int | None:
        """Policy input width stored in a checkpoint (``None`` if it cannot be read)."""
        meta = state.get("pol_meta")
        if meta is not None:
            arr = np.asarray(meta, dtype=float).reshape(-1)
            if arr.size:
                return int(arr[0])
        w0 = state.get("pol_a_W0")
        if w0 is not None:
            return int(np.shape(w0)[0])
        return None

    def state_dict(self):
        st = {"pol_" + k: v for k, v in self.policy.state_dict().items()}
        st["rarl_meta"] = np.array([self.total_steps])
        return st

    def load_state_dict(self, state):
        ckpt_in = self._checkpoint_policy_input(state)
        if ckpt_in is not None and ckpt_in != self.policy_in:
            raise ValueError(
                f"RARL policy input mismatch: checkpoint has {ckpt_in} dims, this agent "
                f"builds {self.policy_in} (env obs {self.obs_dim} + risk 4 + ttc "
                f"{self.n_ttc_features} + action risk {self.n_action_risk} + sensor "
                f"fusion {self.n_sensor} + lane aware "
                f"{self.n_lane_aware} + belief {5 * self.n_belief} + lane clearance "
                f"{self.n_lane_features}). "
                f"sensor_fusion is {'on' if self.use_sensor else 'off'} and adds "
                f"{SENSOR_DIM} dims when on; lane_clearance_features is "
                f"{'on' if self.use_lane_clearance else 'off'} "
                f"and adds {LANE_CLEARANCE_DIM} dims when on; turn the extra blocks "
                f"{'off' if (self.use_sensor or self.use_lane_clearance) else 'on'} to "
                f"load this checkpoint, or retrain a model with the current observation."
            )
        self.policy.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith("pol_")})
        meta = np.asarray(state.get("rarl_meta", [0]), dtype=float)
        self.total_steps = int(meta[0])

    def stats(self):
        lane = self.last_lane or {}
        sens = self.last_sensor
        ttc_min = float(self.last_risk.get("ttc_min", 0.0))
        ego_ttc = float(sens.ego_lead_ttc) if sens is not None else float("inf")
        target_window = float(sens.target_safe_window) if sens is not None else 0.0
        sm_state = self._sm_command.state if self._sm_command is not None else "off"
        sm_code = {"follow": 0.0, "prepare_left": 1.0, "prepare_right": 1.0,
                   "abort": 2.0, "complete": 3.0}.get(sm_state, -1.0)
        return {
            "risk_total": float(self.last_risk.get("total_risk", 0.0)),
            "risk_road": float(self.last_risk.get("road_risk", 0.0)),
            "risk_vehicle": float(self.last_risk.get("vehicle_risk", 0.0)),
            "risk_ttc": float(self.last_risk.get("ttc_hazard", 0.0)),
            "risk_ttc_front": float(self.last_risk.get("ttc_front_hazard", 0.0)),
            "risk_ttc_rear": float(self.last_risk.get("ttc_rear_hazard", 0.0)),
            "ttc_min": ttc_min if np.isfinite(ttc_min) else 0.0,
            "action_risk": float(np.max(self.last_action_risk)) if self.use_action_risk else 0.0,
            "dense_weight": self._dense_weight(),
            "lane_overtake": float(bool(lane.get("overtake_available", False))),
            "lane_current_gap": float(lane.get("current_gap", 0.0)),
            "lane_best_gap": float(lane.get("best_gap", 0.0)),
            "overtake_lock": float(self._overtake_target is not None),
            "overtake_cooldown": float(self._overtake_cooldown),
            "overtake_completions": float(self.overtake_completions),
            "overtake_aborts": float(self.overtake_aborts),
            "overtake_state": sm_code,
            "overtake_state_machine": float(self.use_overtake_sm),
            "overtake_window": float(self._overtake_window > 0),
            "progress_reward": float(self._progress_reward),
            "sensor_safe_accel": float(bool(sens.safe_to_accelerate)) if sens else 0.0,
            "sensor_target_window": target_window if np.isfinite(target_window) else 0.0,
            "sensor_ego_ttc": ego_ttc if np.isfinite(ego_ttc) else 0.0,
            "sensor_lead_delta": float(lane.get("ego_lead_speed_delta", 0.0)),
        }
