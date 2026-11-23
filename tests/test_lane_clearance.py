"""Focused tests for RARL's optional lane-level front/rear clearance observation.

Standalone on purpose: this module imports only the environment and the
algorithm registry, so it can run even when ``test_rarl`` cannot be imported.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.algos import make_algorithm
from autodrive.algos.rarl import LANE_CLEARANCE_DIM
from autodrive.env import MiniDriveEnv

BASE_POLICY_IN = 48     # 29-dim env obs + 4 risk + 3 vehicles x 5 belief


def make_agent(env, **overrides):
    cfg = {"ppo": {"n_steps": 32, "minibatch": 16, "epochs": 2}}
    cfg.update(overrides)
    agent = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg=cfg, seed=0)
    agent.bind_env(env)
    return agent


def park_traffic(env, front, rear):
    """Move every NPC far away, then place one per ``{lane: gap}`` entry.

    ``front`` gaps are arc lengths ahead of the ego, ``rear`` gaps behind it.
    """
    ob = env.obstacles
    ob.s[:] = (env.prev_s + 100.0) % env.track.length
    used = set()

    def place(lane, s_value):
        free = [int(i) for i in np.where(ob.lane == lane)[0] if int(i) not in used]
        assert free, f"lane {lane} has no unparked NPC left"
        used.add(free[0])
        ob.s[free[0]] = s_value % env.track.length

    for lane, gap in front.items():
        place(lane, env.prev_s + gap)
    for lane, gap in rear.items():
        place(lane, env.prev_s - gap)
    ob._update_positions()


class TestLaneClearanceOffByDefault(unittest.TestCase):
    def test_default_policy_input_is_unchanged(self):
        env = MiniDriveEnv(seed=0, random_start=False)
        agent = make_agent(env)
        self.assertFalse(agent.use_lane_clearance)
        self.assertEqual(agent.n_lane_features, 0)
        self.assertEqual(agent.policy_in, env.obs_dim + 4 + 5 * agent.n_belief)
        self.assertEqual(agent.policy_in, BASE_POLICY_IN)

        obs, _ = env.reset(seed=0)
        feats = agent._features(obs)
        self.assertEqual(feats.shape, (BASE_POLICY_IN,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_default_features_keep_the_old_layout(self):
        """With the switch off, the env / risk / belief blocks are untouched."""
        env = MiniDriveEnv(seed=0, random_start=False)
        agent = make_agent(env)
        obs, _ = env.reset(seed=0)
        feats = agent._features(obs)
        np.testing.assert_allclose(feats[:env.obs_dim], obs)
        self.assertEqual(feats.shape, (env.obs_dim + 4 + 5 * agent.n_belief,))
        self.assertEqual(agent._checkpoint_policy_input(agent.state_dict()), BASE_POLICY_IN)


class TestLaneClearanceFeatures(unittest.TestCase):
    def setUp(self):
        self.env = MiniDriveEnv(seed=0, random_start=False)
        self.agent = make_agent(self.env, lane_clearance_features=True,
                                lane_clearance_range=30.0)
        self.obs, _ = self.env.reset(seed=0)

    def test_dimension_grows_by_six(self):
        self.assertTrue(self.agent.use_lane_clearance)
        self.assertEqual(self.agent.n_lane_features, LANE_CLEARANCE_DIM)
        self.assertEqual(LANE_CLEARANCE_DIM, 6)
        self.assertEqual(self.agent.policy_in, BASE_POLICY_IN + 6)
        feats = self.agent._features(self.obs)
        self.assertEqual(feats.shape, (BASE_POLICY_IN + 6,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_front_rear_gaps_are_per_lane(self):
        self.assertEqual(self.env.lane_of(), 1)          # ego in the centre lane
        park_traffic(self.env,
                     front={1: 12.0, 2: 6.0},            # 12 m ahead here, 6 m one lane left
                     rear={1: 4.0})                      # 4 m behind here
        clear = self.agent._lane_clearance().reshape(3, 2)

        # rows: [left adjacent (lane 2), ego (lane 1), right adjacent (lane 0)]
        left, ego, right = clear
        self.assertAlmostEqual(ego[0], 12.0 / 30.0, places=6)
        self.assertAlmostEqual(ego[1], 4.0 / 30.0, places=6)
        self.assertAlmostEqual(left[0], 6.0 / 30.0, places=6)
        self.assertAlmostEqual(left[1], 1.0, places=6)    # nothing behind -> saturated
        self.assertAlmostEqual(right[0], 1.0, places=6)   # empty lane
        self.assertAlmostEqual(right[1], 1.0, places=6)
        self.assertTrue(np.all(clear >= 0.0))

    def test_lane_identity_matters(self):
        """A car ahead in the ego lane is not the same observation as one lane over."""
        park_traffic(self.env, front={1: 10.0}, rear={})
        ego_blocked = self.agent._lane_clearance().copy()
        self.assertLess(ego_blocked[2], 0.5)              # ego front
        self.assertAlmostEqual(ego_blocked[4], 1.0, places=6)

        park_traffic(self.env, front={0: 10.0}, rear={})
        neighbour_blocked = self.agent._lane_clearance()
        self.assertAlmostEqual(neighbour_blocked[2], 1.0, places=6)
        self.assertLess(neighbour_blocked[4], 0.5)        # right-lane front
        self.assertFalse(np.allclose(ego_blocked, neighbour_blocked))

    def test_gaps_saturate_at_the_configured_range(self):
        park_traffic(self.env, front={1: 25.0}, rear={})
        clear = self.agent._lane_clearance().reshape(3, 2)
        self.assertAlmostEqual(clear[1, 0], 25.0 / 30.0, places=6)
        park_traffic(self.env, front={1: 120.0}, rear={})
        clear = self.agent._lane_clearance().reshape(3, 2)
        self.assertAlmostEqual(clear[1, 0], 1.0, places=6)

    def test_missing_lane_is_marked_minus_one(self):
        env = MiniDriveEnv(seed=0, random_start=False, n_lanes=2, lane_width=4.8,
                           half_width=4.8, n_obstacles=6)
        agent = make_agent(env, lane_clearance_features=True)
        env.reset(seed=0)
        clear = agent._lane_clearance().reshape(3, 2)
        self.assertEqual(env.lane_of(), 0)                    # centres at -2.4 / +2.4
        np.testing.assert_allclose(clear[2], [-1.0, -1.0])     # right lane does not exist
        self.assertTrue(np.all(clear[0] >= 0.0))               # left lane does exist


class TestCheckpointCompatibility(unittest.TestCase):
    def _env(self):
        return MiniDriveEnv(seed=0, random_start=False)

    def test_lane_clearance_checkpoint_rejects_default_agent(self):
        env = self._env()
        enabled = make_agent(env, lane_clearance_features=True)
        env.reset(seed=0)
        state = enabled.state_dict()

        disabled = make_agent(env)
        with self.assertRaises(ValueError) as ctx:
            disabled.load_state_dict(state)
        msg = str(ctx.exception)
        self.assertIn("lane_clearance_features", msg)
        self.assertIn(str(BASE_POLICY_IN + 6), msg)
        self.assertIn(str(BASE_POLICY_IN), msg)

    def test_default_checkpoint_rejects_lane_clearance_agent(self):
        env = self._env()
        disabled = make_agent(env)
        env.reset(seed=0)
        state = disabled.state_dict()

        enabled = make_agent(env, lane_clearance_features=True)
        with self.assertRaises(ValueError) as ctx:
            enabled.load_state_dict(state)
        self.assertIn("lane_clearance_features is on", str(ctx.exception))

    def test_default_checkpoint_still_round_trips(self):
        env = self._env()
        agent = make_agent(env)
        env.reset(seed=0)
        state = agent.state_dict()

        other = make_agent(env)
        other.load_state_dict(state)                      # must not raise
        probe = np.random.default_rng(0).normal(size=agent.policy_in)
        diff = np.max(np.abs(agent.policy.actor.forward(probe)
                             - other.policy.actor.forward(probe)))
        self.assertLess(float(diff), 1e-12)

    def test_enabled_checkpoint_round_trips(self):
        env = self._env()
        agent = make_agent(env, lane_clearance_features=True)
        env.reset(seed=0)
        other = make_agent(env, lane_clearance_features=True)
        other.load_state_dict(agent.state_dict())
        self.assertEqual(other.policy_in, agent.policy_in)


class TestLaneClearanceDoesNotTouchReward(unittest.TestCase):
    def test_shaped_reward_matches_default(self):
        env = MiniDriveEnv(seed=0, random_start=False)
        plain = make_agent(env)
        lane = make_agent(env, lane_clearance_features=True)
        env.reset(seed=0)
        risk = {"road_risk": 0.4, "vehicle_risk": 0.7, "total_risk": 1.1}
        for agent in (plain, lane):
            agent._features(env.last_obs)
            agent.last_risk.update(risk)
        self.assertAlmostEqual(plain.shaped_reward(1.0), lane.shaped_reward(1.0), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
