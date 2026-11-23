"""MiniDrive: a small, fully deterministic 2D autonomous-driving environment.

Task
----
A kinematic car has to drive one complete lap of a closed circuit without leaving
the road **and without hitting the other vehicles** that drive along the track.
The agent sees 9 lidar ranges (the road boundaries), the two closest vehicles as
(range, bearing), its own speed / lateral offset / heading error, and picks one of
7 discrete actions per step.

Why this environment for a *test bed*?
  * no exotic dependencies (numpy only), so a training run is reproducible on
    any machine, including CI;
  * ~0.1 ms per step, so a 100 k-step experiment finishes in seconds -- which
    makes hyper-parameter and debugging loops practical;
  * the episode is dense-reward (progress) *and* has an unambiguous success
    criterion (lap completed), which is what a comparison needs;
  * ``n_obstacles`` turns the dynamic-traffic layer on/off without touching a
    single line of algorithm code, which makes it a controlled ablation.

The API mirrors Gymnasium (``reset() -> (obs, info)``, ``step() -> (obs, rew,
terminated, truncated, info)``) but the class is standalone, so the same
training loop can be pointed at ``gymnasium.make("CarRacing-v3")`` later.
"""
from __future__ import annotations

import numpy as np

from .obstacles import ObstacleSet
from .track import Track, wrap_angle

DEFAULT_CONFIG = {
    # road geometry / actuation
    "n_lanes": 3,           # number of parallel lanes
    "lane_width": 3.2,      # [m] -> total road width = 9.6 m
    "half_width": 4.8,      # [m] == n_lanes * lane_width / 2
    "wheelbase": 2.6,       # [m]
    "steer_max": 0.45,      # [rad] max steering angle  -> min radius 5.4 m
    "accel_max": 5.0,       # [m/s^2]   (was 4.0)
    "brake_max": 10.0,      # [m/s^2]   (was 8.0)
    "v_max": 16.0,          # [m/s] == 2 x npc_v_max, ~58 km/h
    "a_lat_max": 9.0,       # [m/s^2] friction-limited yaw rate (no tyre model)
    "drag": 0.25,           # 1/s, linear drag  (terminal speed = accel_max/drag = 20 m/s)
    "dt": 0.1,              # [s] control period
    "max_steps": 1200,      # 120 s hard time limit (traffic makes laps slower)
    # observation
    "lidar_rays": 9,
    "lidar_range": 20.0,    # [m]
    "lidar_fov": 140.0,     # [deg] total field of view
    "lidar_noise": 0.0,     # [m] 1-sigma range noise
    "n_npc_features": 4,    # closest vehicles reported as (range, bearing, closing speed)
    # reward: the task is *fastest crash-free lap*, so time dominates the return
    #   r = -w_step  (time cost, 1 per control step)
    #       + w_progress * ds  (tiny shaping so early exploration is not hopeless)
    #       - w_lateral * (e/W)^2
    #   crash: r_crash, lap complete: r_lap
    # a finished lap worth ~200 - 500 steps is still better than any crash (-200)
    "w_progress": 0.02,
    "w_lateral": 0.05,
    "w_step": 0.1,          # 1.0 per second of driving
    "r_crash": -150.0,      # left the road: must be worse than *any* timeout
    "r_collision": -150.0,  # hit another vehicle
    "r_lap": 20.0,
    # dynamic traffic (set n_obstacles = 0 for the empty-road ablation)
    "n_obstacles": 24,      # dense stream: ~1 vehicle every 19 m of corridor
    "npc_v_min": 5.5,       # [m/s] one cruise speed per episode for the whole stream
    "npc_v_max": 8.0,       # [m/s]
    "npc_drift": 0.22,      # [m] lateral wander inside the lane
    "npc_side_by_side": 2,  # how many two-abreast pairs to place
    "npc_three_abreast": 2, # how many 3-lane walls to place (they clear when the ego arrives)
    "npc_block_range": 32.0,  # [m] "blocked" = every lane occupied within this range ahead
    "npc_min_gap": 25.0,    # [m] never spawn closer than this to the ego
    "npc_lead_gap": 0.0,    # [m] if > 0, guarantee one slow NPC this far ahead of the
                            #     ego at reset (overtake curriculum); 0 keeps the
                            #     default random layout
    "npc_follow_gap": 0.0,  # [m] if > 0, plant one NPC this far *behind* the ego
                            #     in the target lane(s); 0 disables it
    "npc_follow_sides": "both",  # both | left | right (relative to the ego lane)
    "npc_follow_speed": 0.0,     # [m/s] <= 0 uses npc_v_max (a fast follower)
    "npc_range": 30.0,      # [m] detection range reported to the agent (radar)
    # task variation
    "random_start": True,
    "start_lateral": 1.5,   # [m] uniform +- around the centre lane
    "start_heading": 0.12,  # [rad] uniform +-
}

# (steering fraction of steer_max, throttle fraction)
# Sign convention: positive steering angle = counter-clockwise = LEFT turn
# (heading theta is measured from +x, so dtheta/dt = v/L * tan(delta) > 0).
ACTION_TABLE = np.array(
    [
        [+0.45, 0.50],   # 0 hard left
        [+0.22, 0.50],   # 1 left
        [0.00, 0.50],    # 2 straight (cruise)
        [-0.22, 0.50],   # 3 right
        [-0.45, 0.50],   # 4 hard right
        [0.00, 1.00],    # 5 accelerate
        [0.00, -1.00],   # 6 brake
    ],
    dtype=float,
)
ACTION_NAMES = ["hard-left", "left", "straight", "right", "hard-right", "accel", "brake"]


class MiniDriveEnv:
    n_actions = len(ACTION_TABLE)
    action_names = ACTION_NAMES

    def __init__(self, seed: int | None = None, **overrides):
        cfg = dict(DEFAULT_CONFIG)
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg.update(overrides)
        self.cfg = cfg

        # lane layout: centres at -(n-1)/2*w ... +(n-1)/2*w
        n_lanes = int(cfg["n_lanes"])
        lane_w = float(cfg["lane_width"])
        self.n_lanes = n_lanes
        self.lane_width = lane_w
        self.lane_centers = (np.arange(n_lanes) - (n_lanes - 1) / 2.0) * lane_w
        if abs(cfg["half_width"] - n_lanes * lane_w / 2.0) > 1e-6:
            print(f"[MiniDrive] warning: half_width={cfg['half_width']} != "
                  f"n_lanes*lane_width/2={n_lanes * lane_w / 2.0}")

        self.track = Track(half_width=cfg["half_width"])

        self.half_width = cfg["half_width"]
        self.dt = cfg["dt"]
        self.max_steps = cfg["max_steps"]
        self.v_max = cfg["v_max"]

        self.n_rays = int(cfg["lidar_rays"])
        fov = np.deg2rad(cfg["lidar_fov"])
        self.ray_angles = np.linspace(-fov / 2.0, fov / 2.0, self.n_rays)
        self.lidar_range = cfg["lidar_range"]

        self.obs_mode = "lidar_state"
        self.obs_dim = self.n_rays + 4 * int(cfg["n_npc_features"]) + 4

        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._adversary_hook = None

        self.obstacles = ObstacleSet(
            self.track,
            n=int(cfg["n_obstacles"]),
            rng=np.random.default_rng(None if seed is None else seed + 991),
            v_range=(cfg["npc_v_min"], cfg["npc_v_max"]),
            lane_centers=self.lane_centers,
            lane_width=lane_w,
            drift=cfg["npc_drift"],
            n_side_by_side=int(cfg["npc_side_by_side"]),
            n_three_abreast=int(cfg["npc_three_abreast"]),
            block_range=float(cfg["npc_block_range"]),
        )

        # state
        self.x = self.y = self.theta = self.v = 0.0
        self.steps = 0
        self.s_total = 0.0
        self.prev_s = 0.0
        self.idx = 0
        self.last_obs = np.zeros(self.obs_dim)

    # ------------------------------------------------------------------ helpers
    @property
    def episode_length(self) -> float:
        return self.track.length

    def _start_pose(self):
        i = 0
        p = self.track.points[i].copy()
        lateral = 0.0
        heading = 0.0
        if self.cfg["random_start"]:
            lateral = self.rng.uniform(-self.cfg["start_lateral"], self.cfg["start_lateral"])
            heading = self.rng.uniform(-self.cfg["start_heading"], self.cfg["start_heading"])
        p = p + self.track.normals[i] * lateral
        theta = float(np.arctan2(self.track.tangents[i, 1], self.track.tangents[i, 0])) + heading
        return p, theta

    def _observe(self) -> np.ndarray:
        ranges = self.track.raycast(
            (self.x, self.y), self.theta + self.ray_angles, max_range=self.lidar_range, hint=self.idx
        )
        if self.cfg["lidar_noise"] > 0:
            ranges = ranges + self.rng.normal(0.0, self.cfg["lidar_noise"], size=ranges.shape)
        ranges = np.clip(ranges, 0.0, self.lidar_range) / self.lidar_range

        # other vehicles: range, bearing (relative to the ego heading) and the
        # closing speed along the line of sight (radar-style Doppler term)
        k = int(self.cfg["n_npc_features"])
        npc_range = float(self.cfg["npc_range"])
        d, bearing, v_los = self.obstacles.nearest((self.x, self.y), k=k, max_range=npc_range)
        rel = wrap_angle(bearing - self.theta)
        toward = self.v * np.cos(wrap_angle(self.theta - bearing))   # ego speed towards the NPC
        closing = np.clip((toward - v_los) / self.v_max, -1.0, 1.0)
        npc = np.concatenate([
            np.clip(d, 0.0, npc_range) / npc_range,
            np.cos(rel),
            np.sin(rel),
            closing,
        ])

        lateral = self._lateral / self.half_width
        heading_err = wrap_angle(self.theta - self._track_heading)
        return np.concatenate([ranges, npc, [self.v / self.v_max, lateral,
                                             np.sin(heading_err), np.cos(heading_err)]])

    # -------------------------------------------------------------------- reset
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.obstacles.rng = np.random.default_rng(seed + 991)
        p, theta = self._start_pose()
        self.x, self.y = float(p[0]), float(p[1])
        self.theta = float(theta)
        self.v = 0.0
        self.steps = 0
        self.s_total = 0.0
        self.idx, s, self._lateral = self.track.project((self.x, self.y), hint=None, window=None)
        self.prev_s = s
        self._track_heading = float(np.arctan2(self.track.tangents[self.idx, 1], self.track.tangents[self.idx, 0]))
        self.obstacles.reset(ego_s=s, min_gap=self.cfg["npc_min_gap"])
        # Overtake curriculum: put a slow lead car in the ego's lane right away,
        # so the ego actually meets a blocked lane instead of an empty one.
        lead_gap = float(self.cfg["npc_lead_gap"])
        lead_idx = -1
        if lead_gap > 0.0:
            lead_idx = self.obstacles.place_lead(s, self.lane_of(), lead_gap)
        follow_gap = float(self.cfg["npc_follow_gap"])
        if follow_gap > 0.0:
            self._place_followers(s, self.lane_of(), follow_gap,
                                  exclude=() if lead_idx < 0 else (lead_idx,))
        if self._adversary_hook is not None and self.obstacles.n > 0:
            self.obstacles.place_adversary(s)
        self.last_obs = self._observe()
        return self.last_obs, {"lateral": self._lateral, "speed": self.v, "progress": 0.0,
                               "npc_min_distance": self.obstacles.min_distance((self.x, self.y))}
    # --------------------------------------------------------------------- step
    def step(self, action):
        a = int(action)
        steer_frac, thr_frac = ACTION_TABLE[a]
        delta = steer_frac * self.cfg["steer_max"]
        accel = thr_frac * (self.cfg["accel_max"] if thr_frac >= 0 else self.cfg["brake_max"])

        # Euler-midpoint integration of the kinematic bicycle model.
        # The yaw rate is capped by the friction limit v*omega <= a_lat_max, i.e.
        # the car understeers instead of cornering on rails at any speed.  This is
        # what makes braking before a corner necessary.
        v = self.v + (accel - self.cfg["drag"] * self.v) * self.dt
        v = float(np.clip(v, 0.0, self.v_max))
        omega_cmd = v / self.cfg["wheelbase"] * np.tan(delta)
        omega_max = self.cfg["a_lat_max"] / max(v, 1.0)
        omega = float(np.clip(omega_cmd, -omega_max, omega_max))
        theta_mid = self.theta + 0.5 * omega * self.dt
        self.x += v * np.cos(theta_mid) * self.dt
        self.y += v * np.sin(theta_mid) * self.dt
        self.theta = float(wrap_angle(self.theta + omega * self.dt))
        self.v = v
        self.steps += 1

        # other vehicles move at the same control rate; they also need to know
        # where the ego is so that a full 3-lane wall can open up ahead of it.
        # If an adversary controller is installed, vehicle 0 is driven by it.
        adversary = None
        if self._adversary_hook is not None and self.obstacles.n > 0:
            cmd = self._adversary_hook(self)
            if cmd is not None:
                adversary = (0, float(cmd[0]), float(cmd[1]))
        self.obstacles.step(self.dt, ego_s=self.prev_s, adversary=adversary)

        prev_idx = self.idx
        self.idx, s, self._lateral = self.track.project((self.x, self.y), hint=prev_idx)
        self._track_heading = float(
            np.arctan2(self.track.tangents[self.idx, 1], self.track.tangents[self.idx, 0])
        )

        # signed arc-length progress, robust to crossing the start/finish line
        ds = s - self.prev_s
        if ds > self.track.length / 2:
            ds -= self.track.length
        elif ds < -self.track.length / 2:
            ds += self.track.length
        self.prev_s = s
        self.s_total += ds

        off_track = abs(self._lateral) > self.half_width
        collision = self.obstacles.collision((self.x, self.y), self.theta)
        lap_done = self.s_total >= self.track.length

        reward = (
            self.cfg["w_progress"] * float(np.clip(ds, -2.0, 2.0))
            - self.cfg["w_lateral"] * (self._lateral / self.half_width) ** 2
            - self.cfg["w_step"]
        )
        terminated = False
        info = {
            "lateral": self._lateral,
            "speed": self.v,
            "ds": ds,
            "progress": self.s_total,
            "crash": False,
            "collision": False,
            "success": False,
            "action": a,
            "npc_min_distance": self.obstacles.min_distance((self.x, self.y)),
        }
        if collision:
            reward += self.cfg["r_collision"]
            terminated = True
            info["crash"] = True
            info["collision"] = True
        elif off_track:
            reward += self.cfg["r_crash"]
            terminated = True
            info["crash"] = True
        elif lap_done:
            reward += self.cfg["r_lap"]
            terminated = True
            info["success"] = True
        truncated = (not terminated) and (self.steps >= self.max_steps)

        self.last_obs = self._observe()
        return self.last_obs, float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- utilities
    def state_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "theta": self.theta, "v": self.v, "steps": self.steps}

    def set_adversary(self, hook) -> None:
        """Install (or clear) an external controller for vehicle 0.

        ``hook(env) -> (steer_cmd, throttle_cmd)`` with both in [-1, 1]; used by the
        RARL algorithm to drive an adversarial vehicle. ``None`` disables it.
        """
        self._adversary_hook = hook

    def lane_of(self, lateral: float | None = None) -> int:
        """Index of the lane whose centre is closest to ``lateral`` (default: the ego)."""
        lat = self._lateral if lateral is None else float(lateral)
        return int(np.argmin(np.abs(self.lane_centers - lat)))

    def _place_followers(self, ego_s: float, ego_lane: int, gap: float,
                         exclude=()) -> None:
        """Plant fast NPCs ``gap`` metres behind the ego in the target lane(s)."""
        sides = str(self.cfg["npc_follow_sides"]).lower()
        if sides not in ("both", "left", "right"):
            raise ValueError(f"npc_follow_sides must be both|left|right, got '{sides}'")
        speed = float(self.cfg["npc_follow_speed"])
        if speed <= 0.0:
            speed = float(self.cfg["npc_v_max"])
        candidates = []
        if sides in ("both", "left"):
            candidates.append(int(ego_lane) + 1)      # lane_centers grow to the left
        if sides in ("both", "right"):
            candidates.append(int(ego_lane) - 1)
        used = [int(i) for i in exclude]
        for lane in candidates:
            if not 0 <= lane < self.n_lanes:
                continue
            idx = self.obstacles.place_follow(ego_s, lane, gap, speed, exclude=used)
            if idx >= 0:
                used.append(idx)

    def describe(self) -> str:
        t = self.track
        return (
            f"MiniDrive | lap length {t.length:6.1f} m | road {2*self.half_width:.1f} m "
            f"in {self.n_lanes} lanes of {self.lane_width:.1f} m | "
            f"centre-line samples {t.n_points} | min turn radius {t.min_turn_radius():5.1f} m | "
            f"obs {self.obs_dim} dims | actions {self.n_actions} | "
            f"traffic: {self.obstacles.describe() if self.obstacles.n else 'none'}"
        )
