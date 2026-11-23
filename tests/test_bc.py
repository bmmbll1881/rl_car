"""Focused tests for the behaviour-cloning path.

Three things must hold, and each is checked here directly:

1. ``PPO.pretrain`` actually minimises the supervised loss and raises the
   probability of the demonstrated action (not just "runs without error");
2. a pretrained actor still round-trips through the normal checkpoint format,
   and an input-width mismatch is rejected instead of silently broadcasting;
3. the expert collector stores the *learner's* policy input -- including the
   SensorFusion block for a sensor-fusion RARL policy -- and the labels are the
   actions the deterministic driver actually took.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.algos.ppo import PPO
from autodrive.algos.rarl import RARL
from autodrive.demonstrations import collect_expert_demonstrations
from autodrive.env import MiniDriveEnv

STAGE1 = {
    "n_obstacles": 12,
    "npc_v_min": 2.5,
    "npc_v_max": 4.0,
    "npc_block_range": 22.0,
    "npc_min_gap": 18.0,
    "npc_lead_gap": 20.0,
    "npc_side_by_side": 2,
    "npc_three_abreast": 1,
    "v_max": 10.0,
}


def _synthetic(n=128, dim=5, seed=9):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, dim))
    actions = (obs[:, 0] + 0.5 * obs[:, 1] > 0.0).astype(np.int64)
    return obs, actions


class TestPPOPretrain(unittest.TestCase):
    def test_loss_falls_and_demonstrated_action_probability_rises(self):
        obs, actions = _synthetic()
        agent = PPO(5, 2, {"bc_lr": 3e-3}, seed=2)
        out = agent.pretrain(obs, actions, epochs=25, batch_size=16)
        self.assertEqual(out["bc_samples"], 128)
        self.assertEqual(out["bc_feature_dim"], 5)
        self.assertLess(out["bc_loss_end"], out["bc_loss_start"],
                        "cross-entropy did not decrease")
        self.assertGreater(out["bc_action_prob_end"], out["bc_action_prob_start"],
                           "the demonstrated action did not become more likely")
        self.assertGreaterEqual(out["bc_accuracy_end"], out["bc_accuracy_start"])

    def test_pretrain_does_not_touch_the_critic(self):
        obs, actions = _synthetic(32)
        agent = PPO(5, 2, {"bc_lr": 3e-3}, seed=3)
        before = [w.copy() for w in agent.critic.params]
        agent.pretrain(obs, actions, epochs=3, batch_size=8)
        for a, b in zip(before, agent.critic.params):
            self.assertTrue(np.array_equal(a, b))

    def test_rejects_mismatched_widths_and_labels(self):
        agent = PPO(5, 2, seed=0)
        obs, actions = _synthetic(8)
        with self.assertRaises(ValueError):
            agent.pretrain(obs[:, :4], actions, epochs=1)
        with self.assertRaises(ValueError):
            agent.pretrain(obs, actions + 5, epochs=1)
        with self.assertRaises(ValueError):
            agent.pretrain(obs, actions[:-1], epochs=1)

    def test_pretrained_checkpoint_round_trips(self):
        obs, actions = _synthetic(64)
        agent = PPO(5, 2, {"bc_lr": 3e-3}, seed=4)
        agent.pretrain(obs, actions, epochs=10, batch_size=16)
        state = agent.state_dict()

        clone = PPO(5, 2, seed=99)
        clone.load_state_dict(state)
        self.assertLess(float(np.max(np.abs(clone.actor.forward(obs) - agent.actor.forward(obs)))),
                        1e-12, "checkpoint did not restore the pretrained actor")

        wider = PPO(6, 2, seed=0)
        with self.assertRaises(ValueError):
            wider.load_state_dict(state)


class TestMaskedPretrain(unittest.TestCase):
    """The supervised update must use the same masked logits as the PPO update."""

    def test_mask_concentrates_probability_on_allowed_actions(self):
        obs, _ = _synthetic(64, dim=5)
        actions = np.zeros(64, dtype=np.int64)          # every label is action 0
        mask = np.tile(np.array([True, False, False]), (64, 1))
        agent = PPO(5, 3, {"bc_lr": 3e-3}, seed=11)
        out = agent.pretrain(obs, actions, epochs=40, batch_size=16, action_mask=mask)
        self.assertTrue(out["bc_mask_used"])
        self.assertEqual(out["bc_label_forced"], 0)
        self.assertEqual(out["bc_masked_samples"], 64)
        self.assertGreater(out["bc_action_prob_end"], 0.99,
                           "the only allowed action did not become (nearly) certain")
        self.assertAlmostEqual(out["bc_accuracy_end"], 1.0)
        # the forbidden actions must be exactly zero-probability under the mask
        logits = np.where(mask, agent.actor.forward(obs), -1e9)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        self.assertLess(float(p[:, 1:].max()), 1e-9)

    def test_labels_forbidden_by_the_mask_are_re_admitted_and_counted(self):
        obs, _ = _synthetic(32, dim=5)
        actions = np.ones(32, dtype=np.int64)           # label is action 1 ...
        mask = np.tile(np.array([True, False, True]), (32, 1))   # ... which is masked
        agent = PPO(5, 3, {"bc_lr": 1e-3}, seed=12)
        out = agent.pretrain(obs, actions, epochs=20, batch_size=8, action_mask=mask)
        self.assertEqual(out["bc_label_forced"], 32)
        self.assertTrue(np.isfinite(out["bc_loss_end"]))
        self.assertGreater(out["bc_action_prob_end"], out["bc_action_prob_start"])

    def test_mask_shape_is_validated(self):
        obs, actions = _synthetic(16, dim=5)
        agent = PPO(5, 2, seed=0)
        with self.assertRaises(ValueError):
            agent.pretrain(obs, actions, epochs=1,
                           action_mask=np.ones((16, 3), dtype=bool))


class TestSensorFusionDemoCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MiniDriveEnv(seed=0, **STAGE1)
        cls.agent = RARL(cls.env.obs_dim, cls.env.n_actions, cfg={"sensor_fusion": True}, seed=0)
        cls.agent.bind_env(cls.env)
        cls.obs, cls.actions, cls.stats = collect_expert_demonstrations(
            cls.env, cls.agent, episodes=1, seed_base=30_000)

    def test_policy_input_is_env_obs_plus_sensor_block(self):
        expected = self.env.obs_dim + 4 + 25 + 5 * self.agent.n_belief
        self.assertEqual(self.agent.policy_in, expected)
        self.assertEqual(self.agent.policy_in, 73, "SensorFusion adds 25 dims to the 48-dim RARL input")

    def test_demo_dimensions_match_the_policy(self):
        x, y = self.obs, self.actions
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[1], self.agent.policy_in)
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertGreater(x.shape[0], 0)
        self.assertEqual(self.stats["feature_dim"], self.agent.policy_in)
        self.assertEqual(self.stats["steps"], x.shape[0])
        self.assertTrue(np.all(y >= 0) and np.all(y < self.env.n_actions))

    def test_stats_report_overtake_activity_honestly(self):
        for key in ("overtake_opportunity_steps", "blocked_steps", "lane_changes",
                    "blocked_lane_changes", "overtakes_completed"):
            self.assertIn(key, self.stats)
            self.assertGreaterEqual(int(self.stats[key]), 0)
        self.assertEqual(self.stats["episodes"], 1)
        # a completion can never exceed the number of blocked lane changes
        self.assertLessEqual(self.stats["overtakes_completed"],
                             self.stats["blocked_lane_changes"])

    def test_rarl_pretrain_accepts_policy_space_and_rejects_raw_env_obs(self):
        out = self.agent.pretrain(self.obs, self.actions, epochs=2, batch_size=64)
        self.assertEqual(out["bc_feature_dim"], self.agent.policy_in)
        self.assertEqual(out["bc_policy_in"], self.agent.policy_in)
        raw = self.obs[:, : self.env.obs_dim]
        with self.assertRaises(ValueError):
            self.agent.pretrain(raw, self.actions, epochs=1)


class TestDemoLabelsAreExpertActions(unittest.TestCase):
    def test_collection_is_deterministic(self):
        """Re-running the deterministic expert on the same seeds reproduces the labels."""
        env = MiniDriveEnv(seed=0, **STAGE1)
        agent = RARL(env.obs_dim, env.n_actions, cfg={"sensor_fusion": True}, seed=0)
        agent.bind_env(env)
        x1, y1, s1 = collect_expert_demonstrations(env, agent, episodes=1, seed_base=31_000,
                                                   max_steps=80)
        env2 = MiniDriveEnv(seed=0, **STAGE1)
        agent2 = RARL(env2.obs_dim, env2.n_actions, cfg={"sensor_fusion": True}, seed=0)
        agent2.bind_env(env2)
        x2, y2, s2 = collect_expert_demonstrations(env2, agent2, episodes=1, seed_base=31_000,
                                                   max_steps=80)
        self.assertTrue(np.array_equal(y1, y2))
        self.assertLess(float(np.max(np.abs(x1 - x2))), 1e-12)
        self.assertTrue(np.array_equal(s1["masks"], s2["masks"]))


class TestMemorylessExpert(unittest.TestCase):
    """The default demonstrations must be Markov in the policy input."""

    def test_driver_has_no_hidden_state(self):
        env = MiniDriveEnv(seed=0, **STAGE1)
        from autodrive.baselines import MemorylessExpertDriver
        drv = MemorylessExpertDriver(env)
        for attr in ("current_offset", "target_lane", "_cooldown"):
            self.assertFalse(hasattr(drv, attr),
                             f"memoryless driver still keeps '{attr}'")

    def test_action_depends_only_on_the_current_state(self):
        from autodrive.baselines import MemorylessExpertDriver
        env = MiniDriveEnv(seed=0, **STAGE1)
        env.reset(seed=32_000)
        drv = MemorylessExpertDriver(env)
        first = drv.act()
        # calling act repeatedly without stepping the simulator must not drift:
        # a cooldown/latched lane would change the answer with the call count
        repeats = {drv.act() for _ in range(50)}
        self.assertEqual(repeats, {first})

    def test_collector_uses_memoryless_expert_and_stores_masks(self):
        env = MiniDriveEnv(seed=0, **STAGE1)
        agent = RARL(env.obs_dim, env.n_actions,
                     cfg={"sensor_fusion": True, "safety_shield": False}, seed=0)
        agent.bind_env(env)
        x, y, stats = collect_expert_demonstrations(env, agent, episodes=1,
                                                    seed_base=32_100, max_steps=60)
        self.assertEqual(stats["expert"], "expert-memoryless")
        self.assertTrue(stats["memoryless"])
        self.assertEqual(stats["masks"].shape, (x.shape[0], env.n_actions))
        self.assertEqual(stats["masks"].dtype, np.bool_)
        self.assertIn("mask_conflicts", stats)
        out = agent.pretrain(x, y, epochs=1, batch_size=32, action_mask=stats["masks"])
        self.assertTrue(out["bc_mask_used"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
