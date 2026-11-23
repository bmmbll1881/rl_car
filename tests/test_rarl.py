"""Unit tests for the risk-aware method (risk field + GM belief + risk-aware PPO)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.env import ACTION_TABLE, MiniDriveEnv
from autodrive.risk import GaussianMixtureBelief, TTCCollisionRisk, TimeVaryingRiskField
from autodrive.algos import REGISTRY, make_algorithm


class TestRiskField(unittest.TestCase):
    def setUp(self):
        self.field = TimeVaryingRiskField()
        self.centers = np.array([-3.2, 0.0, 3.2])

    def test_road_risk_grows_towards_the_boundary(self):
        center = self.field.road_risk(0.0, 4.8, self.centers)
        edge = self.field.road_risk(4.3, 4.8, self.centers)
        self.assertLess(center, edge)
        self.assertGreater(edge, 10.0 * center)     # 1/gap^2 blows up near the edge

    def test_vehicle_risk_peaks_inside_the_ellipse(self):
        modes = [[(np.array([0.0, 0.0]), np.eye(2) * 0.01, 1.0)]]
        inside = self.field.vehicle_risk((0.0, 0.0), modes)
        far = self.field.vehicle_risk((200.0, 200.0), modes)
        # p is clamped to 0.999, so the plateau is 0.999 * vehicle_weight
        self.assertAlmostEqual(inside, 2.0, delta=0.05)
        self.assertLess(far, 0.01 * inside)

    def test_features_are_normalised(self):
        modes = [[(np.array([5.0, 0.0]), np.eye(2), 0.7)]]
        feats = self.field.features((0.0, 0.0), 0.5, 4.8, self.centers, modes)
        self.assertEqual(feats.shape, (4,))
        self.assertTrue(np.all(feats >= 0.0) and np.all(feats <= 1.0))


class TestBelief(unittest.TestCase):
    def test_modes_are_normalised_and_features_have_right_shape(self):
        belief = GaussianMixtureBelief()
        meas = {0: (np.array([0.0, 0.0]), np.array([8.0, 0.0]), np.array([0.0, 1.0]), 8.0)}
        for _ in range(5):
            belief.update(meas, 0.1)
        modes = belief.modes_for([0])[0]
        self.assertEqual(len(modes), 7)
        self.assertAlmostEqual(sum(m[2] for m in modes), 1.0, places=6)
        feats = belief.features([0], np.array([1.0, 0.0]), 3)
        self.assertEqual(feats.shape, (15,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_missing_vehicle_is_dropped(self):
        belief = GaussianMixtureBelief()
        meas = {0: (np.array([0.0, 0.0]), np.array([8.0, 0.0]), np.array([0.0, 1.0]), 8.0)}
        belief.update(meas, 0.1)
        belief.update({}, 0.1)
        self.assertTrue(belief.empty)


class TestTTCCollisionRisk(unittest.TestCase):
    def setUp(self):
        self.field = TTCCollisionRisk({
            "ttc_safe_time": 2.0,
            "ttc_horizon": 3.0,
            "ttc_clearance_margin": 0.3,
            "ttc_uncertainty_weight": 0.5,
        })

    @staticmethod
    def _mode(x, y, vx, vy, cov_scale=0.01, p=1.0):
        return [[(np.array([x, y, vx, vy]), np.eye(2) * cov_scale, p)]]

    def test_closing_front_vehicle_has_positive_hazard(self):
        front = self.field.estimate((0.0, 0.0), 0.0, 10.0,
                                    self._mode(10.0, 0.0, 6.0, 0.0))
        self.assertGreater(front["ttc_hazard"], 0.0)
        self.assertLess(front["ttc_min"], 2.0)

    def test_safe_adjacent_overtake_is_not_penalised(self):
        adjacent = self.field.estimate((0.0, 0.0), 0.0, 10.0,
                                       self._mode(10.0, 3.2, 6.0, 0.0))
        self.assertEqual(adjacent["ttc_hazard"], 0.0)
        self.assertEqual(adjacent["ttc_proximity"], 0.0)

    def test_vehicle_behind_without_closing_is_not_penalised(self):
        behind = self.field.estimate((0.0, 0.0), 0.0, 10.0,
                                     self._mode(-10.0, 0.0, 6.0, 0.0))
        self.assertEqual(behind["ttc_hazard"], 0.0)

    def test_hazard_increases_as_front_gap_closes(self):
        far = self.field.estimate((0.0, 0.0), 0.0, 10.0,
                                  self._mode(14.0, 0.0, 6.0, 0.0))
        near = self.field.estimate((0.0, 0.0), 0.0, 10.0,
                                   self._mode(8.0, 0.0, 6.0, 0.0))
        self.assertGreater(near["ttc_hazard"], far["ttc_hazard"])
        self.assertLess(near["ttc_min"], far["ttc_min"])

    def test_action_conditioned_risk_sees_adjacent_lane_change(self):
        modes = self._mode(4.0, 3.2, 8.0, 0.0)
        result = self.field.action_risks(
            (0.0, 0.0, 0.0, 8.0), ACTION_TABLE, modes,
            horizon=1.0, dt=0.1,
        )
        self.assertEqual(result["risk"][2], 0.0)
        self.assertGreater(result["risk"][0], result["risk"][2])
        self.assertGreater(result["risk"][0], 0.0)

    def test_action_risk_decreases_after_dense_schedule(self):
        field = self.field
        original = field.cfg.copy()
        try:
            from autodrive.algos import make_algorithm

            env = MiniDriveEnv(seed=0)
            agent = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg={
                "hazard_mode": "ttc",
                "action_risk": True,
                "dense_risk_weight": 0.2,
                "dense_risk_end": 0.05,
                "dense_risk_decay_steps": 100,
                "ppo": {"n_steps": 8, "minibatch": 4},
            }, seed=0)
            agent.total_steps = 50
            self.assertAlmostEqual(agent._dense_weight(), 0.125)
            agent.total_steps = 200
            self.assertAlmostEqual(agent._dense_weight(), 0.05)
        finally:
            field.cfg = original


class TestRiskAwareLearner(unittest.TestCase):
    def _make(self, extra_cfg=None):
        env = MiniDriveEnv(seed=0)
        cfg = {"ppo": {"n_steps": 32, "minibatch": 16, "epochs": 2}}
        cfg.update(extra_cfg or {})
        agent = make_algorithm("rarl", env.obs_dim, env.n_actions,
                               cfg=cfg, seed=0)
        agent.bind_env(env)
        return env, agent

    def test_registered(self):
        self.assertIn("rarl", REGISTRY)

    def test_no_adversary_vehicle(self):
        """The adversarial vehicle was removed: binding must not hijack any NPC."""
        env, agent = self._make()
        env.reset(seed=0)
        self.assertIsNone(env._adversary_hook)
        lat = env.obstacles.lat.copy()
        for _ in range(10):
            env.step(agent.act(env.last_obs))
        # NPC 0 keeps following its lane (small drift only)
        self.assertLess(float(np.max(np.abs(env.obstacles.lat - lat))), 1.0)

    def test_observation_is_augmented_with_risk_and_belief(self):
        env, agent = self._make()
        obs, _ = env.reset(seed=0)
        self.assertEqual(agent.policy_in, env.obs_dim + 4 + 5 * agent.n_belief)
        feats = agent._features(obs)
        self.assertEqual(feats.shape, (agent.policy_in,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_reward_is_risk_penalised(self):
        env, agent = self._make()
        obs, _ = env.reset(seed=0)
        a = agent.act(obs)
        nobs, r, term, trunc, info = env.step(a)
        agent.update(obs, a, r, nobs, term, trunc, info)
        self.assertGreater(agent.last_risk.get("total_risk", -1.0), 0.0)

    def test_ttc_mode_adds_features_and_shapes_with_hazard(self):
        env, agent = self._make({"hazard_mode": "ttc", "dense_risk_weight": 0.25})
        obs, _ = env.reset(seed=0)
        feats = agent._features(obs)
        self.assertEqual(agent.policy_in, env.obs_dim + 4 + 2 + 5 * agent.n_belief)
        self.assertEqual(feats.shape, (agent.policy_in,))
        agent.last_risk.update({
            "road_risk": 0.9,
            "vehicle_risk": 0.3,
            "ttc_hazard": 0.2,
        })
        expected = -(agent.cfg["risk_weight"] * (0.2 + 0.25 * 0.3)
                     + agent.cfg["road_risk_weight"] * 0.9)
        self.assertAlmostEqual(agent.shaped_reward(0.0), expected, places=9)

    def test_finetune_mode_keeps_baseline_input_dimension(self):
        env, agent = self._make({"hazard_mode": "ttc", "ttc_features": False})
        obs, _ = env.reset(seed=0)
        feats = agent._features(obs)
        self.assertEqual(agent.policy_in, env.obs_dim + 4 + 5 * agent.n_belief)
        self.assertEqual(feats.shape, (agent.policy_in,))
        self.assertIn("ttc_hazard", agent.last_risk)

    def test_ttc_shield_masks_longitudinal_actions(self):
        env, agent = self._make({
            "hazard_mode": "ttc",
            "ttc_features": False,
            "safety_shield": True,
        })
        agent.last_risk.update({"ttc_front_hazard": 3.0, "ttc_rear_hazard": 0.0})
        front_mask = agent._action_mask()
        self.assertFalse(front_mask[0])
        self.assertTrue(front_mask[6])
        self.assertFalse(front_mask[2])
        self.assertFalse(front_mask[5])

        agent.last_risk.update({"ttc_front_hazard": 0.0, "ttc_rear_hazard": 3.0})
        rear_mask = agent._action_mask()
        self.assertFalse(rear_mask[6])
        self.assertTrue(rear_mask[5])

    def test_action_risk_is_observed_and_masked(self):
        env, agent = self._make({
            "hazard_mode": "ttc",
            "action_risk": True,
            "action_risk_mask_threshold": 0.25,
            "safety_shield": True,
        })
        obs, _ = env.reset(seed=0)
        feats = agent._features(obs)
        self.assertEqual(agent.policy_in,
                         env.obs_dim + 4 + 2 + env.n_actions + 5 * agent.n_belief)
        self.assertEqual(feats.shape, (agent.policy_in,))
        agent.last_action_risk = np.array([0.9, 0.1, 0.0, 0.0, 0.2, 0.0, 0.3])
        agent.last_risk.update({"ttc_front_hazard": 0.0, "ttc_rear_hazard": 0.0})
        mask = agent._action_mask()
        self.assertFalse(mask[0])
        self.assertFalse(mask[6])
        self.assertTrue(mask[1])

    def test_evaluation_hook_advances_belief(self):
        env, agent = self._make()
        obs, _ = env.reset(seed=0)
        agent.reset_episode(obs)
        self.assertFalse(agent.belief.empty)
        actions = []
        for _ in range(3):
            a = agent.act(obs)
            obs, _, term, trunc, _ = env.step(a)
            agent.observe_transition(obs, term, trunc)
            actions.append(a)
        self.assertEqual(len(actions), 3)
        self.assertFalse(agent.belief.empty)
        self.assertIs(agent._feature_obs, obs)

    def test_lane_aware_prefers_open_adjacent_lane(self):
        env = MiniDriveEnv(seed=0, random_start=False)
        agent = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg={
            "hazard_mode": "ttc",
            "action_risk": True,
            "lane_aware": True,
            "safety_shield": True,
            "overtake_lock": True,
            "ppo": {"n_steps": 8, "minibatch": 4},
        }, seed=0)
        agent.bind_env(env)
        obs, _ = env.reset(seed=0)
        self.assertEqual(env.lane_of(), 1)

        ob = env.obstacles
        ob.s[:] = (env.prev_s + 100.0) % env.track.length
        for lane, gap in ((0, 35.0), (1, 10.0), (2, 40.0)):
            idx = int(np.where(ob.lane == lane)[0][0])
            ob.s[idx] = (env.prev_s + gap) % env.track.length
        ob._update_positions()

        feats = agent._lane_features()
        self.assertEqual(feats.shape, (9,))
        self.assertTrue(agent.last_lane["overtake_available"])
        self.assertEqual(agent.last_lane["best_lane"], 2)
        self.assertEqual(agent.last_lane["direction"], 1)
        self.assertAlmostEqual(feats[-1], 1.0)

        agent._feature_obs = None
        agent._features(obs)
        mask = agent._action_mask()
        self.assertFalse(mask[6], "braking should be suppressed while an open lane exists")
        self.assertTrue(mask[0], "the tactical lock should use hard-left until the lane change completes")
        self.assertFalse(mask[1], "only the selected completion action should remain available")
        self.assertFalse(mask[2], "straight following should be suppressed while changing lanes")
        self.assertEqual(agent._overtake_target, 2)

        env._lateral = float(env.lane_centers[2])
        agent._action_mask()
        self.assertIsNone(agent._overtake_target)
        self.assertGreater(agent._overtake_cooldown, 0)

        agent.last_risk.update({"ttc_hazard": 0.0, "road_risk": 0.0})
        agent._action_risk_taken = 0.0
        plan = {"ego_lane": 1, "overtake_available": True, "direction": 1, "best_lane": 2}

        agent._lane_taken = dict(plan)
        env._lateral = float(env.lane_centers[1])
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)   # still in the ego lane
        hold_reward = agent.shaped_reward(0.0)
        self.assertAlmostEqual(agent._lane_bonus_taken, 0.0)

        agent._lane_taken = dict(plan)
        agent._overtake_target = 0                      # lock committed to the wrong lane
        env._lateral = float(env.lane_centers[2])
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)
        self.assertAlmostEqual(agent.shaped_reward(0.0), hold_reward)

        agent._lane_taken = dict(plan)
        env._lateral = float(env.lane_centers[1])
        agent._overtake_target = 2                      # the planned change completes
        env._lateral = float(env.lane_centers[2])
        self.assertAlmostEqual(agent._advance_lane_state(), agent.cfg["lane_change_bonus"])
        change_reward = agent.shaped_reward(0.0)
        self.assertGreater(change_reward, hold_reward)
        self.assertAlmostEqual(change_reward - hold_reward,
                               agent.cfg["lane_change_bonus"], places=9)
        self.assertEqual(agent.overtake_completions, 1)
        self.assertEqual(agent._advance_lane_state(), 0.0,
                         "a settled transition must not pay twice")

        agent._lane_taken = dict(plan)                  # swerve to the other side
        agent._overtake_target = 2
        env._lateral = float(env.lane_centers[0])
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0,
                               msg="an unplanned lane change must pay nothing")

        agent._feature_obs = None
        augmented = agent._features(obs)
        self.assertEqual(agent.policy_in,
                         env.obs_dim + 4 + 2 + env.n_actions + 9 + 5 * agent.n_belief)
        self.assertEqual(augmented.shape, (agent.policy_in,))

    def test_lane_side_clearance_rejects_occupied_target(self):
        env = MiniDriveEnv(seed=0, random_start=False)
        agent = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg={
            "hazard_mode": "ttc",
            "action_risk": True,
            "lane_aware": True,
            "overtake_side_clearance": 7.0,
            "ppo": {"n_steps": 8, "minibatch": 4},
        }, seed=0)
        agent.bind_env(env)
        env.reset(seed=0)
        ob = env.obstacles
        ob.s[:] = (env.prev_s + 100.0) % env.track.length
        for lane, gap in ((0, 35.0), (1, 10.0), (2, 40.0)):
            idx = int(np.where(ob.lane == lane)[0][0])
            ob.s[idx] = (env.prev_s + gap) % env.track.length
        side_idx = int(np.where(ob.lane == 2)[0][1])
        ob.s[side_idx] = (env.prev_s + 2.0) % env.track.length
        ob._update_positions()

        agent._lane_features()
        self.assertLess(agent.last_lane["lane_side_clearance"][2], 7.0)
        self.assertEqual(agent.last_lane["best_lane"], 0)
        self.assertEqual(agent.last_lane["direction"], -1)

    def test_shield_vetoes_the_tactical_lock(self):
        """A dangerous front gap must abort the lane-change lock instead of forcing a swerve."""
        env = MiniDriveEnv(seed=0, random_start=False)
        agent = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg={
            "hazard_mode": "ttc",
            "lane_aware": True,
            "safety_shield": True,
            "overtake_lock": True,
            "ppo": {"n_steps": 8, "minibatch": 4},
        }, seed=0)
        agent.bind_env(env)
        obs, _ = env.reset(seed=0)
        ob = env.obstacles
        ob.s[:] = (env.prev_s + 100.0) % env.track.length
        for lane, gap in ((0, 35.0), (1, 10.0), (2, 40.0)):
            idx = int(np.where(ob.lane == lane)[0][0])
            ob.s[idx] = (env.prev_s + gap) % env.track.length
        ob._update_positions()

        agent._feature_obs = None
        agent._features(obs)
        self.assertEqual(agent._overtake_target, 2)
        agent.last_risk.update({"ttc_front_hazard": 0.9, "ttc_rear_hazard": 0.0})

        mask = agent._action_mask()
        self.assertIsNone(agent._overtake_target)
        self.assertTrue(mask[6], "braking must stay available")
        self.assertFalse(mask[:6].any(), "no steering / accelerating with a car right ahead")

    def test_risk_weight_sits_on_vehicle_risk(self):
        """The shaping must be dominated by the vehicle (traffic) risk."""
        env, agent = self._make()
        env.reset(seed=0)
        agent.last_risk = {"road_risk": 0.9, "vehicle_risk": 0.3, "total_risk": 0.99}
        shaped = agent.shaped_reward(0.0)
        vehicle_part = agent.cfg['risk_weight'] * 0.3
        road_part = agent.cfg['road_risk_weight'] * 0.9
        self.assertAlmostEqual(shaped, -(vehicle_part + road_part), places=9)
        self.assertGreater(vehicle_part, 5.0 * road_part)   # weight really sits on traffic

    def test_checkpoint_roundtrip(self):
        env, agent = self._make()
        obs, _ = env.reset(seed=0)
        for _ in range(60):
            a = agent.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            agent.update(obs, a, r, nobs, term, trunc, info)
            obs = nobs
            if term or trunc:
                obs, _ = env.reset(seed=7)
        state = agent.state_dict()
        other = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg={"ppo": {"hidden": 64}}, seed=1)
        other.load_state_dict(state)
        probe = np.random.default_rng(0).normal(size=agent.policy_in)
        diff = np.max(np.abs(agent.policy.actor.forward(probe)
                             - other.policy.actor.forward(probe)))
        self.assertLess(float(diff), 1e-12)
        self.assertEqual(other.total_steps, agent.total_steps)


class TestOvertakeReward(unittest.TestCase):
    """The overtake reward may only pay for verified, safe forward progress.

    These tests exist because the previous loop collapsed to "brake forever"
    with a reward that was collectable by steering alone: a 30k-step policy
    braked on 94% of steps (0.14 m/s) while the lane-change bonus paid for
    weaving back and forth.
    """

    CFG = {
        "hazard_mode": "ttc",
        "action_risk": True,
        "lane_aware": True,
        "safety_shield": True,
        "overtake_lock": True,
        "safe_progress_weight": 0.2,
        "safe_speed_target": 8.0,
        "safe_progress_max_hazard": 0.2,
        "overtake_window_steps": 10,
        "overtake_pass_margin": 3.0,
        "overtake_progress_multiplier": 2.0,
        "lane_change_min_progress": 12.0,
        "lane_change_min_front_gap": 8.0,
        "ppo": {"n_steps": 8, "minibatch": 4},
    }

    def _make(self, **over):
        env = MiniDriveEnv(seed=0, random_start=False)
        cfg = dict(self.CFG)
        cfg.update(over)
        agent = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg=cfg, seed=0)
        agent.bind_env(env)
        obs, _ = env.reset(seed=0)
        agent.reset_episode(obs)
        # a quiet, hazard-free state so only the term under test can pay
        agent.last_risk.update({"ttc_hazard": 0.0, "road_risk": 0.0,
                                "ttc_front_hazard": 0.0, "vehicle_risk": 0.0})
        agent._action_risk_taken = 0.0
        agent.last_action_risk = np.zeros(env.n_actions)
        return env, agent

    @staticmethod
    def _park(env, gaps, speed=None):
        """Move every NPC far ahead, then place one ``gaps[lane]`` metres ahead."""
        ob = env.obstacles
        ob.s[:] = (env.prev_s + 100.0) % env.track.length
        for lane, gap in gaps.items():
            idx = int(np.where(ob.lane == lane)[0][0])
            ob.s[idx] = (env.prev_s + gap) % env.track.length
            if speed is not None:
                ob.v[idx] = float(speed)
                ob.v_nominal[idx] = float(speed)
        ob._update_positions()

    @staticmethod
    def _refresh(agent, env):
        """Re-seed the belief after the test moved the traffic."""
        agent.belief.reset()
        ids = agent._nearest_ids(agent.n_track)
        if ids:
            agent.belief.update(agent._measurements(ids), env.dt)
        agent._feature_obs = None

    @staticmethod
    def _plan(agent, previous, target, available=True, blocker=None):
        agent._lane_taken = {
            "ego_lane": int(previous),
            "overtake_available": bool(available),
            "direction": int(np.sign(target - previous)),
            "best_lane": int(target),
            "blocker": blocker,
        }

    # ------------------------------------------------------------ completion
    def test_bonus_requires_a_real_crossing(self):
        env, agent = self._make()
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        self._plan(agent, 1, 2)
        agent._overtake_target = 2
        env._lateral = float(env.lane_centers[1])
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)
        self.assertEqual(agent.overtake_completions, 0)

        self._plan(agent, 1, 2)
        agent._overtake_target = 2
        env._lateral = float(env.lane_centers[2])
        self.assertAlmostEqual(agent._advance_lane_state(),
                               agent.cfg["lane_change_bonus"])
        self.assertEqual(agent.overtake_completions, 1)

    def test_unavailable_or_unplanned_change_pays_nothing(self):
        env, agent = self._make()
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        env._lateral = float(env.lane_centers[2])

        self._plan(agent, 1, 2, available=False)
        agent._overtake_target = 2
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)

        self._plan(agent, 1, 2)                     # right lane is planned...
        env._lateral = float(env.lane_centers[0])   # ...but the ego went left
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)

        self._plan(agent, 1, 2)                     # planned lane, wrong lock target
        agent._overtake_target = 0
        env._lateral = float(env.lane_centers[2])
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0,
                               msg="a lane the shield vetoed must not pay")
        self.assertEqual(agent.overtake_completions, 0)

    def test_occupied_destination_is_not_a_completion(self):
        env, agent = self._make()
        # a car 3 m ahead in the destination lane: cutting in, not overtaking
        self._park(env, {0: 35.0, 1: 10.0, 2: 3.0})
        self._plan(agent, 1, 2)
        agent._overtake_target = 2
        env._lateral = float(env.lane_centers[2])
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)

        # a car alongside the destination lane
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        ob = env.obstacles
        idx = int(np.where(ob.lane == 2)[0][1])
        ob.s[idx] = (env.prev_s + 1.0) % env.track.length
        ob._update_positions()
        self._plan(agent, 1, 2)
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)
        self.assertEqual(agent.overtake_completions, 0)

    def test_weaving_back_and_forth_pays_once(self):
        env, agent = self._make()
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        env._lateral = float(env.lane_centers[2])
        self._plan(agent, 1, 2)
        agent._overtake_target = 2
        self.assertAlmostEqual(agent._advance_lane_state(),
                               agent.cfg["lane_change_bonus"])

        # straight back, then a second "planned" change within 12 m -> no pay
        env._lateral = float(env.lane_centers[1])
        agent._overtake_target = 2
        self._plan(agent, 1, 2)
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)
        self.assertEqual(agent.overtake_completions, 1)

        # after real arc progress the next planned change is paid again
        env.s_total += 20.0
        env._lateral = float(env.lane_centers[2])
        self._plan(agent, 1, 2)
        self.assertAlmostEqual(agent._advance_lane_state(),
                               agent.cfg["lane_change_bonus"])
        self.assertEqual(agent.overtake_completions, 2)

    def test_stuck_lock_is_abandoned_after_the_deadline(self):
        env, agent = self._make(overtake_lock_steps=5)
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        agent._overtake_target = 2
        agent._lock_step = 0
        env.steps = 100
        mask = agent._action_mask()
        self.assertIsNone(agent._overtake_target)
        self.assertGreater(agent._overtake_cooldown, 0)
        self.assertTrue(mask.any())
        # and the abandoned lock cannot collect the bonus later
        self._plan(agent, 1, 2)
        env._lateral = float(env.lane_centers[2])
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)

    # -------------------------------------------------- deferred (pass) bonus
    def test_deferred_bonus_requires_the_actual_pass(self):
        """The completion bonus may not be farmable by crawling and crossing."""
        env, agent = self._make(lane_change_bonus_requires_pass=True,
                                overtake_window_steps=10, lane_change_bonus=3.0)
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        blocker = agent._lane_lead_id(1)
        self.assertIsNotNone(blocker)
        self._plan(agent, 1, 2, blocker=blocker)
        agent._overtake_target = 2
        env._lateral = float(env.lane_centers[2])
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0,
                               msg="the crossing alone must not pay")
        self.assertEqual(agent.overtake_completions, 0)
        self.assertAlmostEqual(agent._lane_bonus_pending, 0.0)
        self.assertEqual(agent._overtake_window, 9)     # 10 set, one step consumed

        ob = env.obstacles
        ob.s[int(blocker)] = (env.prev_s - 5.0) % env.track.length
        ob._update_positions()
        agent._lane_taken = {}
        self.assertAlmostEqual(agent._advance_lane_state(), 3.0)
        self.assertTrue(agent._overtake_passed)
        self.assertEqual(agent.overtake_completions, 1)

    def test_deferred_bonus_expires_without_a_pass(self):
        env, agent = self._make(lane_change_bonus_requires_pass=True,
                                overtake_window_steps=3)
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        blocker = agent._lane_lead_id(1)
        self._plan(agent, 1, 2, blocker=blocker)
        agent._overtake_target = 2
        env._lateral = float(env.lane_centers[2])
        self.assertEqual(agent._advance_lane_state(), 0.0)
        for _ in range(5):
            agent._lane_taken = {}
            self.assertEqual(agent._advance_lane_state(), 0.0)
        self.assertEqual(agent.overtake_completions, 0)
        self.assertEqual(agent._overtake_window, 0)
        self.assertFalse(agent._overtake_passed)

    def test_progress_window_continues_after_the_pass(self):
        env, agent = self._make(overtake_window_steps=10, overtake_after_pass_steps=5)
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        blocker = agent._lane_lead_id(1)
        self._plan(agent, 1, 2, blocker=blocker)
        agent._overtake_target = 2
        env._lateral = float(env.lane_centers[2])
        self.assertAlmostEqual(agent._advance_lane_state(),
                               agent.cfg["lane_change_bonus"])
        ob = env.obstacles
        ob.s[int(blocker)] = (env.prev_s - 5.0) % env.track.length
        ob._update_positions()
        agent._lane_taken = {}
        agent._advance_lane_state()
        self.assertTrue(agent._overtake_passed)
        self.assertEqual(agent._overtake_window, 5)
        agent.shaped_reward(0.0)                     # consume the one-off bonus
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=0.5), 0.2 * 0.5 * 2.0)

    # --------------------------------------------------------- safe progress
    def test_safe_progress_requires_forward_motion_and_low_hazard(self):
        env, agent = self._make()
        env.v = 5.0
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=0.5), 0.2 * 0.5)
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=-0.5), 0.0,
                               msg="reversing must not pay")
        paid = agent._episode_progress
        agent.last_risk["ttc_front_hazard"] = 0.35          # too close to the lead
        agent.shaped_reward(0.0, ds=0.5)
        self.assertAlmostEqual(agent._episode_progress, paid)
        agent.last_risk["ttc_front_hazard"] = 0.0
        agent._action_risk_taken = 0.9                     # action the mask rejects
        agent.shaped_reward(0.0, ds=0.5)
        self.assertAlmostEqual(agent._episode_progress, paid)
        agent._action_risk_taken = 0.0
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=0.5), 0.1)

    def test_safe_progress_is_capped_by_the_speed_target(self):
        env, agent = self._make()
        env.v = 20.0
        # 2 m in one 0.1 s step == 20 m/s, capped to the 8 m/s target
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=2.0), 0.2 * 8.0 * env.dt)
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=0.05), 0.2 * 0.05)

    # ------------------------------------------------------------ pass window
    def test_window_boosts_progress_until_the_blocker_is_passed(self):
        env, agent = self._make()
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        blocker = agent._lane_lead_id(1)
        self.assertIsNotNone(blocker)

        self._plan(agent, 1, 2, blocker=blocker)
        agent._overtake_target = 2
        env._lateral = float(env.lane_centers[2])
        self.assertAlmostEqual(agent._advance_lane_state(),
                               agent.cfg["lane_change_bonus"])
        agent.shaped_reward(0.0)                     # consume the one-off bonus
        self.assertEqual(agent._overtake_window, 10)
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=0.5), 0.2 * 0.5 * 2.0)

        # the car that blocked the old lane is now behind the ego
        ob = env.obstacles
        ob.s[int(blocker)] = (env.prev_s - 5.0) % env.track.length
        ob._update_positions()
        agent._lane_taken = {}
        self.assertAlmostEqual(agent._advance_lane_state(), 0.0)
        self.assertTrue(agent._overtake_passed)
        self.assertEqual(agent._overtake_window, 0)
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=0.5), 0.2 * 0.5)

    def test_window_expires(self):
        env, agent = self._make()
        self._park(env, {0: 35.0, 1: 10.0, 2: 40.0})
        blocker = agent._lane_lead_id(1)
        self._plan(agent, 1, 2, blocker=blocker)
        agent._overtake_target = 2
        env._lateral = float(env.lane_centers[2])
        agent._advance_lane_state()
        agent.shaped_reward(0.0)                     # consume the one-off bonus
        for _ in range(10):
            agent._lane_taken = {}
            agent._advance_lane_state()
        self.assertEqual(agent._overtake_window, 0)
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=0.5), 0.2 * 0.5)

    # ------------------------------------------------------------- collision
    def test_safe_progress_requires_room_in_the_ego_lane(self):
        """The hazard gate saturates (mode probabilities), the gap gate does not."""
        env, agent = self._make(safe_progress_min_gap=12.0)
        self._park(env, {0: 200.0, 1: 8.0, 2: 200.0})
        agent._lane_features()                    # refresh the cached lane gaps
        agent.shaped_reward(0.0, ds=0.5)
        self.assertAlmostEqual(agent._episode_progress, 0.0,
                               msg="no progress reward while tailgating")
        self._park(env, {0: 200.0, 1: 25.0, 2: 200.0})
        agent._lane_features()
        self.assertAlmostEqual(agent.shaped_reward(0.0, ds=0.5), 0.1)

    # ------------------------------------------------------------- collision
    def test_ttc_hazard_saturates_below_the_shield_threshold(self):
        """Documented hole: the shield threshold cannot be reached by one vehicle.

        ``TTCCollisionRisk.estimate`` multiplies each mode's hazard by that
        mode's belief weight, and the 7-mode belief gives ~1/7 per mode, so a
        vehicle that is 0.6 s away reports ``ttc_front_hazard`` well below
        ``shield_hazard_threshold=0.5``.  The shield therefore stays open; this
        test pins the current behaviour so the gap is not forgotten.
        """
        env, agent = self._make()
        self._park(env, {0: 200.0, 1: 8.0, 2: 200.0}, speed=2.0)   # lead 8 m ahead, 2 m/s
        self._refresh(agent, env)
        env.v = 8.0
        agent._features(env.last_obs)
        self.assertLess(agent.last_risk["ttc_min"], 1.0,
                        "the stack does see the imminent contact")
        self.assertLess(agent.last_risk["ttc_front_hazard"],
                        agent.cfg["shield_hazard_threshold"],
                        "but the weighted hazard never reaches the threshold")
        self.assertFalse(agent._shield_blocks_action(5),
                         "so accelerating into the lead is not masked")

    def test_collision_terminates_with_the_penalty(self):
        env, agent = self._make()
        self._park(env, {0: 200.0, 1: 8.0, 2: 200.0}, speed=2.0)
        self._refresh(agent, env)
        env.v = 8.0
        agent._features(env.last_obs)

        obs, r, term, trunc, info = env.step(5)
        steps = 0
        while not (term or trunc) and steps < 60:
            obs, r, term, trunc, info = env.step(5)
            steps += 1
        self.assertTrue(info["collision"])
        self.assertTrue(term)
        self.assertLess(r, env.cfg["r_collision"] + 1.0)   # collision dominates the step

    def test_collision_step_pays_no_overtake_bonus(self):
        env, agent = self._make(safe_progress_min_gap=12.0)
        self._park(env, {0: 200.0, 1: 5.0, 2: 200.0}, speed=2.0)
        self._refresh(agent, env)
        env.v = 8.0
        feats = agent._features(env.last_obs)
        obs = env.last_obs
        obs, r, term, trunc, info = env.step(5)
        steps = 0
        while not (term or trunc) and steps < 60:
            obs, r, term, trunc, info = env.step(5)
            steps += 1
        self.assertTrue(info["collision"])
        # the training path: refresh the new state, then shape the reward
        self._refresh(agent, env)
        agent._features(obs)
        agent._advance_lane_state()
        agent.shaped_reward(r, ds=float(info["ds"]))
        self.assertAlmostEqual(agent._lane_bonus_taken, 0.0)
        self.assertAlmostEqual(agent._episode_progress, 0.0,
                               msg="a collision step must not pay for progress")
        self.assertTrue(np.all(np.isfinite(feats)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
