"""Tests for the camera-free SensorFusion / LaneRadar observation layer.

The sensor is the single source of lane geometry for ``lane_aware``,
``lane_clearance_features`` and ``overtake_lock``, so these tests pin both the
readings themselves and the wiring into RARL (default 48-dim input, opt-in
25-dim block, checkpoint compatibility, reward gating).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.algos import make_algorithm
from autodrive.env import MiniDriveEnv
from autodrive.sensors import SENSOR_DIM, LaneRadar, SensorFusion

BASE_POLICY_IN = 48     # 29-dim env obs + 4 risk + 3 vehicles x 5 belief
ROW = 7                 # features per lane row
SCALARS = 4


def make_agent(env, **overrides):
    cfg = {"ppo": {"n_steps": 32, "minibatch": 16, "epochs": 2}}
    cfg.update(overrides)
    agent = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg=cfg, seed=0)
    agent.bind_env(env)
    return agent


def place_traffic(env, front=None, rear=None):
    """Park every NPC far ahead, then place one per ``{lane: (gap, speed)}``.

    ``front`` gaps are metres ahead of the ego, ``rear`` gaps behind it.  Every
    placed car gets the given speed, so the closing-speed terms are testable.
    """
    ob = env.obstacles
    L = env.track.length
    ob.s[:] = (env.prev_s + 150.0) % L
    used = set()

    def take(lane):
        free = [int(i) for i in np.where(ob.lane == lane)[0] if int(i) not in used]
        assert free, f"lane {lane} has no unparked NPC left"
        used.add(free[0])
        return free[0]

    for lane, (gap, speed) in (front or {}).items():
        i = take(lane)
        ob.s[i] = (env.prev_s + gap) % L
        ob.v[i] = ob.v_nominal[i] = float(speed)
    for lane, (gap, speed) in (rear or {}).items():
        i = take(lane)
        ob.s[i] = (env.prev_s - gap) % L
        ob.v[i] = ob.v_nominal[i] = float(speed)
    ob._update_positions()


class TestSensorDimensions(unittest.TestCase):
    def test_row_layout_is_fixed(self):
        self.assertEqual(SENSOR_DIM, 3 * ROW + SCALARS)
        self.assertEqual(LaneRadar, SensorFusion)
        self.assertEqual(SensorFusion.DIM, SENSOR_DIM)

    def test_sensor_off_keeps_the_48_dim_input(self):
        env = MiniDriveEnv(seed=0, random_start=False)
        agent = make_agent(env)
        self.assertFalse(agent.use_sensor)
        self.assertEqual(agent.n_sensor, 0)
        self.assertEqual(agent.policy_in, BASE_POLICY_IN)
        obs, _ = env.reset(seed=0)
        self.assertEqual(agent._features(obs).shape, (BASE_POLICY_IN,))

    def test_sensor_on_adds_25_dims(self):
        env = MiniDriveEnv(seed=0, random_start=False)
        agent = make_agent(env, sensor_fusion=True)
        self.assertTrue(agent.use_sensor)
        self.assertEqual(agent.n_sensor, SENSOR_DIM)
        self.assertEqual(agent.policy_in, BASE_POLICY_IN + SENSOR_DIM)
        obs, _ = env.reset(seed=0)
        feats = agent._features(obs)
        self.assertEqual(feats.shape, (BASE_POLICY_IN + SENSOR_DIM,))
        self.assertTrue(np.all(np.isfinite(feats)))
        # the sensor block sits between action-risk and the lane blocks
        self.assertEqual(agent.policy_in, env.obs_dim + 4 + SENSOR_DIM + 5 * agent.n_belief)


class TestSensorReadings(unittest.TestCase):
    def setUp(self):
        self.env = MiniDriveEnv(seed=0, random_start=False)
        self.sensor = SensorFusion({"range": 40.0, "v_scale": 16.0, "ttc_horizon": 6.0})
        self.env.reset(seed=0)
        self.assertEqual(self.env.lane_of(), 1)

    def test_front_rear_gap_and_closing_speed_per_lane(self):
        self.env.v = 6.0
        place_traffic(self.env, front={1: (12.0, 3.0), 2: (20.0, 4.0)},
                      rear={1: (8.0, 6.0)})
        frame = self.sensor.scan(self.env)

        self.assertEqual([r.lane for r in frame.rows], [2, 1, 0])
        ego, left = frame.reading(1), frame.reading(2)

        self.assertAlmostEqual(ego.front_gap, 12.0, places=6)
        self.assertAlmostEqual(ego.front_speed, 3.0, places=6)
        self.assertAlmostEqual(ego.front_rel_speed, 3.0, places=6)     # 6 - 3
        self.assertAlmostEqual(ego.rear_gap, 8.0, places=6)
        self.assertAlmostEqual(ego.rear_rel_speed, 0.0, places=6)      # 6 - 6
        self.assertAlmostEqual(ego.front_ttc, (12.0 - 4.3) / 3.0, places=6)
        self.assertEqual(left.front_gap, 20.0)
        self.assertAlmostEqual(left.front_rel_speed, 2.0, places=6)    # 6 - 4

    def test_empty_lane_reports_free_range(self):
        env = MiniDriveEnv(seed=0, random_start=False, n_obstacles=1)
        env.reset(seed=0)
        ob = env.obstacles
        ob.lane[0] = 2                                   # the only car sits far left
        ob.s[0] = (env.prev_s + 10.0) % env.track.length
        ob._update_positions()
        env.v = 6.0
        frame = self.sensor.scan(env)
        right = frame.reading(0)
        self.assertEqual(right.front_gap, float("inf"))
        self.assertEqual(right.rear_gap, float("inf"))
        self.assertTrue(right.safe_to_enter)
        feats = self.sensor.features_from(frame)[:-SCALARS].reshape(3, ROW)
        self.assertAlmostEqual(feats[2, 0], 1.0, places=6)   # free ahead
        self.assertAlmostEqual(feats[2, 2], 1.0, places=6)   # free behind

    def test_ttc_only_exists_while_closing(self):
        self.env.v = 8.0
        place_traffic(self.env, front={1: (20.0, 1.0)}, rear={})
        frame = self.sensor.scan(self.env)
        self.assertAlmostEqual(frame.ego_lead_ttc, (20.0 - 4.3) / 7.0, places=6)

        # a lead that is as fast as the ego is not being approached
        place_traffic(self.env, front={1: (20.0, 8.0)}, rear={})
        frame = self.sensor.scan(self.env)
        self.assertEqual(frame.ego_lead_ttc, float("inf"))
        self.assertEqual(frame.ego_lead_speed_delta, 0.0)

    def test_safe_window_shrinks_when_a_follower_closes_in(self):
        self.env.v = 5.0
        place_traffic(self.env, front={}, rear={1: (10.0, 10.0)})
        frame = self.sensor.scan(self.env)
        ego = frame.reading(1)
        self.assertAlmostEqual(ego.rear_rel_speed, 5.0, places=6)
        # (gap - min_rear_gap) / closing = (10 - 5) / 5
        self.assertAlmostEqual(ego.safe_window, 1.0, places=6)

        # a follower at the ego's own speed never closes the gap
        place_traffic(self.env, front={}, rear={1: (10.0, 5.0)})
        frame = self.sensor.scan(self.env)
        self.assertAlmostEqual(frame.reading(1).safe_window, 6.0, places=6)

    def test_safe_to_accelerate_flag(self):
        self.env.v = 6.0
        place_traffic(self.env, front={1: (12.0, 6.0)}, rear={})
        self.assertTrue(self.sensor.scan(self.env).safe_to_accelerate)

        place_traffic(self.env, front={1: (9.0, 6.0)}, rear={})
        self.assertFalse(self.sensor.scan(self.env).safe_to_accelerate,
                         "too little room ahead")

        self.env.v = 8.0
        place_traffic(self.env, front={1: (20.0, 1.0)}, rear={})
        self.assertFalse(self.sensor.scan(self.env).safe_to_accelerate,
                         "closing on the lead much too fast")

    def test_target_lane_and_safe_window(self):
        self.env.v = 6.0
        place_traffic(self.env, front={0: (8.0, 4.0), 1: (10.0, 4.0),
                                       2: (40.0, 4.0)}, rear={})
        frame = self.sensor.scan(self.env)
        self.assertTrue(frame.target_available)
        self.assertEqual(frame.best_lane, 2)          # left neighbour has room
        self.assertEqual(frame.direction, +1)
        self.assertAlmostEqual(frame.target_gap, 40.0, places=6)
        self.assertAlmostEqual(frame.target_safe_window, 6.0, places=6)

        feats = self.sensor.features_from(frame)
        self.assertAlmostEqual(feats[-1], 1.0, places=6)    # available
        self.assertAlmostEqual(feats[-2], 1.0, places=6)    # window saturated

        # an unblocked ego lane is not an overtake opportunity
        place_traffic(self.env, front={0: (8.0, 4.0), 1: (35.0, 6.0),
                                       2: (40.0, 4.0)}, rear={})
        frame = self.sensor.scan(self.env)
        self.assertFalse(frame.target_available)
        self.assertEqual(frame.direction, 0)

    def test_missing_lane_is_encoded_minus_one(self):
        env = MiniDriveEnv(seed=0, random_start=False, n_lanes=2, lane_width=4.8,
                           half_width=4.8, n_obstacles=6)
        env.reset(seed=0)
        self.assertEqual(env.lane_of(), 0)
        feats = SensorFusion().features(env)[:-SCALARS].reshape(3, ROW)
        np.testing.assert_allclose(feats[2], np.full(ROW, -1.0))   # right lane absent

    def test_full_readings_cover_every_lane(self):
        """``rows`` is the 3-lane feature window; ``readings`` is the whole road."""
        self.env.v = 6.0
        place_traffic(self.env, front={2: (15.0, 3.0)}, rear={})
        self.assertEqual(self.env.lane_of(), 1)
        frame = self.sensor.scan(self.env)
        self.assertEqual(len(frame.readings), self.env.n_lanes)
        self.assertAlmostEqual(frame.reading(2).front_gap, 15.0, places=6)

        # two lanes away from the target the feature window no longer has it,
        # but the full readings still do
        self.env._lateral = float(self.env.lane_centers[0])
        frame = self.sensor.scan(self.env)
        self.assertEqual([r.lane for r in frame.rows], [1, 0, -1])
        self.assertTrue(frame.reading(2).exists)
        self.assertAlmostEqual(frame.reading(2).front_gap, 15.0, places=6)

    def test_signed_offsets_follow_front_and_rear_gaps(self):
        """The state machine needs the blocker's signed position, not just a gap."""
        self.env.v = 6.0
        place_traffic(self.env, front={1: (12.0, 3.0)}, rear={1: (8.0, 6.0)})
        frame = self.sensor.scan(self.env)
        front_id = int(frame.front_ids[1])
        self.assertAlmostEqual(frame.npc_offsets[front_id], 12.0, places=6)
        behind = [off for npc, off in frame.npc_offsets.items()
                  if npc != front_id and off < 0.0]
        self.assertEqual(len(behind), 1)
        self.assertAlmostEqual(behind[0], -8.0, places=6)

    def test_sensor_does_not_mutate_the_environment(self):
        self.env.v = 6.0
        place_traffic(self.env, front={1: (12.0, 3.0)}, rear={1: (8.0, 6.0)})
        before = (self.env.x, self.env.y, self.env.theta, self.env.v, self.env.prev_s,
                  self.env.obstacles.s.copy(), self.env.obstacles.v.copy())
        self.sensor.features(self.env)
        after = (self.env.x, self.env.y, self.env.theta, self.env.v, self.env.prev_s,
                 self.env.obstacles.s.copy(), self.env.obstacles.v.copy())
        self.assertEqual(before[0:5], after[0:5])
        np.testing.assert_allclose(before[5], after[5])
        np.testing.assert_allclose(before[6], after[6])


class TestSensorCheckpoints(unittest.TestCase):
    def _env(self):
        return MiniDriveEnv(seed=0, random_start=False)

    def test_sensor_checkpoint_rejects_default_agent(self):
        env = self._env()
        agent = make_agent(env, sensor_fusion=True)
        env.reset(seed=0)
        with self.assertRaises(ValueError) as ctx:
            make_agent(env).load_state_dict(agent.state_dict())
        msg = str(ctx.exception)
        self.assertIn("sensor_fusion", msg)
        self.assertIn(str(BASE_POLICY_IN + SENSOR_DIM), msg)

    def test_default_checkpoint_rejects_sensor_agent(self):
        env = self._env()
        agent = make_agent(env)
        env.reset(seed=0)
        with self.assertRaises(ValueError) as ctx:
            make_agent(env, sensor_fusion=True).load_state_dict(agent.state_dict())
        self.assertIn("sensor_fusion is on", str(ctx.exception))

    def test_sensor_checkpoint_round_trips(self):
        env = self._env()
        agent = make_agent(env, sensor_fusion=True)
        env.reset(seed=0)
        other = make_agent(env, sensor_fusion=True)
        other.load_state_dict(agent.state_dict())
        probe = np.random.default_rng(0).normal(size=agent.policy_in)
        diff = np.max(np.abs(agent.policy.actor.forward(probe)
                             - other.policy.actor.forward(probe)))
        self.assertLess(float(diff), 1e-12)

    def test_default_checkpoint_still_loads_in_the_legacy_agent(self):
        env = self._env()
        agent = make_agent(env)
        env.reset(seed=0)
        other = make_agent(env)
        other.load_state_dict(agent.state_dict())
        self.assertEqual(other.policy_in, BASE_POLICY_IN)


class TestLaneTacticsUseTheSensor(unittest.TestCase):
    def setUp(self):
        self.env = MiniDriveEnv(seed=0, random_start=False)
        self.env.reset(seed=0)
        self.agent = make_agent(self.env, sensor_fusion=True, lane_aware=True,
                                lane_clearance_features=True, overtake_lock=True,
                                overtake_lock_pause_in_window=True,
                                overtake_lock_min_speed=3.0)

    def test_lane_aware_block_matches_the_frame(self):
        self.env.v = 6.0
        place_traffic(self.env, front={1: (10.0, 4.0), 2: (40.0, 4.0)},
                      rear={1: (6.0, 8.0)})
        feats = self.agent._lane_features()
        frame = self.agent.sensor.last_frame
        scale = self.agent.lane_clearance_range

        for row, reading in enumerate(frame.rows):
            self.assertAlmostEqual(feats[2 * row], min(reading.front_gap, scale) / scale)
            self.assertAlmostEqual(feats[2 * row + 1], min(reading.rear_gap, scale) / scale)
        self.assertEqual(self.agent.last_lane["best_lane"], frame.best_lane)
        self.assertEqual(self.agent.last_lane["direction"], frame.direction)
        self.assertEqual(self.agent.last_lane["ego_lead_ttc"], frame.ego_lead_ttc)
        self.assertAlmostEqual(self.agent.last_lane["ego_lead_speed_delta"],
                               frame.ego_lead_speed_delta)

    def test_lane_clearance_uses_the_same_readings(self):
        self.env.v = 6.0
        place_traffic(self.env, front={1: (12.0, 3.0)}, rear={1: (4.0, 6.0)})
        clear = self.agent._lane_clearance().reshape(3, 2)
        frame = self.agent.sensor.last_frame
        self.assertAlmostEqual(clear[1, 0], frame.reading(1).front_gap / 30.0)
        self.assertAlmostEqual(clear[1, 1], frame.reading(1).rear_gap / 30.0)

    def test_overtake_lock_targets_the_sensor_candidate(self):
        self.env.v = 6.0
        place_traffic(self.env, front={0: (8.0, 4.0), 1: (10.0, 4.0),
                                       2: (40.0, 5.0)}, rear={})
        self.agent._lane_features()
        self.assertEqual(self.agent.last_lane["best_lane"], 2)
        self.assertEqual(self.agent._overtake_target, 2)

    def test_lock_does_not_force_a_swerve_at_a_crawl(self):
        """A hard steer at 0.3 m/s only arcs the car off the road."""
        self.env.v = 0.3
        place_traffic(self.env, front={0: (8.0, 4.0), 1: (10.0, 4.0),
                                       2: (40.0, 5.0)}, rear={})
        self.agent._lane_features()
        self.assertTrue(self.agent.last_lane["overtake_available"])
        self.assertIsNone(self.agent._overtake_target)

        self.env.v = 6.0
        self.agent._lane_features()
        self.assertEqual(self.agent._overtake_target, 2)

    def test_lock_pauses_inside_the_pass_window(self):
        """After a verified change the policy must be free to accelerate, not chain locks."""
        self.env.v = 6.0
        place_traffic(self.env, front={0: (8.0, 4.0), 1: (10.0, 4.0),
                                       2: (40.0, 5.0)}, rear={})
        self.agent._overtake_window = 5
        self.agent._lane_features()
        self.assertIsNone(self.agent._overtake_target)

        self.agent._overtake_window = 0
        self.agent._lane_features()
        self.assertEqual(self.agent._overtake_target, 2)


class TestRewardIsNotActionBased(unittest.TestCase):
    def setUp(self):
        env = MiniDriveEnv(seed=0, random_start=False)
        cfg = {"lane_aware": True, "safe_progress_weight": 0.2,
               "safe_speed_target": 8.0, "safe_progress_max_hazard": 0.2,
               "progress_reward_only_in_overtake_window": True,
               "overtake_progress_multiplier": 2.0, "overtake_window_steps": 10,
               "ppo": {"n_steps": 8, "minibatch": 4}}
        self.env = env
        self.agent = make_agent(env, **cfg)
        obs, _ = env.reset(seed=0)
        self.agent.reset_episode(obs)
        self.agent.last_risk.update({"vehicle_risk": 0.0, "road_risk": 0.0,
                                     "ttc_front_hazard": 0.0, "total_risk": 0.0})
        env.v = 5.0

    def test_progress_outside_the_window_pays_nothing(self):
        self.assertAlmostEqual(self.agent.shaped_reward(0.0, ds=0.5), 0.0)

    def test_progress_inside_the_window_is_paid(self):
        self.agent._overtake_window = 5
        self.assertAlmostEqual(self.agent.shaped_reward(0.0, ds=0.5), 0.2 * 0.5 * 2.0)

    def test_sensor_block_does_not_change_the_reward(self):
        plain = make_agent(self.env)
        plain.bind_env(self.env)
        plain._features(self.env.last_obs)
        self.agent._features(self.env.last_obs)
        for a in (plain, self.agent):
            a.last_risk.update({"vehicle_risk": 0.5, "road_risk": 0.2, "total_risk": 0.7,
                                "ttc_front_hazard": 0.0})
            a.cfg["safe_progress_weight"] = 0.0
        self.assertAlmostEqual(plain.shaped_reward(1.0), self.agent.shaped_reward(1.0), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
