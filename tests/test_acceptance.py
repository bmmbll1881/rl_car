"""Tests for the strict overtake verifier, acceptance gate and selection rule."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.acceptance import StrictOvertakeMonitor, gate
from autodrive.baselines import MemorylessExpertDriver
from autodrive.env import MiniDriveEnv
from autodrive.logging_utils import evaluate_agent, selection_key

STAGE1_SM = {
    "n_obstacles": 1,
    "npc_v_min": 2.5,
    "npc_v_max": 4.0,
    "npc_drift": 0.0,
    "npc_min_gap": 30.0,
    "npc_lead_gap": 20.0,
    "npc_side_by_side": 0,
    "npc_three_abreast": 0,
    "v_max": 10.0,
    "random_start": False,
    "start_lateral": 0.0,
    "start_heading": 0.0,
}


class _BrakeOnly:
    """A policy that never moves: must never be certified as overtaking."""

    def act(self, obs, greedy=False):
        return 6


class TestGate(unittest.TestCase):
    def test_gate_requires_both_parts(self):
        ok = gate(7, 4, 10, 6, 3)
        self.assertTrue(ok["passed"])
        self.assertTrue(ok["clean_lap_pass"])
        self.assertTrue(ok["strict_safe_overtake_pass"])

        few_laps = gate(5, 9, 10, 6, 3)
        self.assertFalse(few_laps["passed"])
        self.assertFalse(few_laps["clean_lap_pass"])

        few_overtakes = gate(10, 2, 10, 6, 3)
        self.assertFalse(few_overtakes["passed"])
        self.assertFalse(few_overtakes["strict_safe_overtake_pass"])

    def test_rates_are_reported_against_the_episode_count(self):
        g = gate(6, 3, 10, 6, 3)
        self.assertAlmostEqual(g["clean_lap_rate"], 0.6)
        self.assertAlmostEqual(g["strict_safe_overtake_rate"], 0.3)


class TestSelectionKey(unittest.TestCase):
    def test_return_is_not_used_and_clean_laps_win(self):
        rows_standing = [{"return": -50.0, "success": 0, "crash": 0, "collision": 0,
                          "clean": True, "clean_lap": False, "strict_safe_overtake": False,
                          "distance": 2.0} for _ in range(10)]
        rows_driving = [{"return": -300.0, "success": 1, "crash": 0, "collision": 0,
                         "clean": True, "clean_lap": True, "strict_safe_overtake": True,
                         "distance": 460.0} for _ in range(6)]
        rows_driving += [{"return": -300.0, "success": 0, "crash": 1, "collision": 1,
                          "clean": False, "clean_lap": False, "strict_safe_overtake": False,
                          "distance": 100.0} for _ in range(4)]
        a = selection_key({"eval_return_mean": -50.0, "_rows": rows_standing})
        b = selection_key({"eval_return_mean": -300.0, "_rows": rows_driving})
        self.assertGreater(b, a,
                           "a high-return standing policy must not beat a driving one")

    def test_falls_back_to_aggregate_fields(self):
        key = selection_key({"eval_clean_laps": 7, "eval_strict_safe_overtake_episodes": 4,
                             "eval_clean_episodes": 8, "eval_success_rate": 0.7,
                             "eval_distance_mean": 400.0})
        self.assertEqual(key[0], 7)
        self.assertEqual(key[1], 4)


class TestStrictMonitor(unittest.TestCase):
    def test_expert_is_certified_on_the_single_lead_environment(self):
        env = MiniDriveEnv(seed=0, **STAGE1_SM)
        drv = MemorylessExpertDriver(env)
        res = evaluate_agent(env, drv, episodes=1, base_seed=40_000)
        rows = res["_rows"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["clean_lap"])
        self.assertGreaterEqual(rows[0]["strict_events"], 1)
        self.assertTrue(rows[0]["strict_safe_overtake"])
        events = rows[0]["strict_events_detail"]
        self.assertTrue(any(e["reason"] == "passed" and e["valid"] for e in events))
        self.assertGreaterEqual(rows[0]["strict_blocked_lane_changes"], 1)

    def test_a_stationary_policy_is_never_certified(self):
        env = MiniDriveEnv(seed=0, **STAGE1_SM)
        res = evaluate_agent(env, _BrakeOnly(), episodes=1, base_seed=40_100)
        row = res["_rows"][0]
        self.assertEqual(row["strict_events"], 0)
        self.assertFalse(row["strict_safe_overtake"])
        self.assertFalse(row["clean_lap"])

    def test_monitor_rejects_a_change_that_leaves_the_target_lane(self):
        """Direct unit check: leaving the destination before the pass is invalid."""
        env = MiniDriveEnv(seed=0, **STAGE1_SM)
        mon = StrictOvertakeMonitor({"block_distance": 40.0, "pass_margin": 3.0,
                                     "min_hold_steps": 3})
        env.reset(seed=1)
        mon.reset_episode(env, episode=0, seed=1)
        # pretend a manoeuvre into lane 2 is in flight, then keep the ego in lane 1
        from autodrive.acceptance import StrictOvertakeEvent
        mon._active = StrictOvertakeEvent(episode=0, seed=1, start_step=0,
                                          src_lane=1, dst_lane=2, blocker_id=0)
        mon.before_step(env)
        env.step(6)
        mon.after_step(env, {"crash": False, "collision": False})
        out = mon.finish_episode({})
        self.assertEqual(out["strict_events"], 0)
        self.assertEqual([e["reason"] for e in out["events"]], ["left_target_lane"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
