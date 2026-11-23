"""Tests for the sensor-driven high-level overtake state machine.

Two layers are pinned here:

* :class:`autodrive.overtake.OvertakeController` is fed real
  :class:`SensorFrame` objects (no mocked readings), so every transition is
  driven by the same facts the policy sees: ego-lane gap and closing speed,
  target-lane front/rear gap, safe window and TTC, and the blocker's signed
  arc offset.
* RARL's ``lane_aware`` / ``overtake_lock`` / reward consume that output, and
  the observation width and checkpoint compatibility stay unchanged.
"""
import os
import json
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.algos import make_algorithm
from autodrive.env import MiniDriveEnv
from autodrive.overtake import OvertakeController, OvertakePhase
from autodrive.sensors import SensorFusion

BASE_POLICY_IN = 48


def park(env, front=None, rear=None, ego_v=8.0):
    """Park every NPC far ahead, then place one per ``{lane: (gap, speed)}``."""
    ob = env.obstacles
    ob.s[:] = (env.prev_s + 150.0) % env.track.length
    used = set()

    def take(lane):
        free = [int(i) for i in np.where(ob.lane == lane)[0] if int(i) not in used]
        assert free, f"lane {lane} has no unparked NPC left"
        used.add(free[0])
        return free[0]

    for lane, (gap, speed) in (front or {}).items():
        i = take(lane)
        ob.s[i] = (env.prev_s + gap) % env.track.length
        ob.v[i] = ob.v_nominal[i] = float(speed)
    for lane, (gap, speed) in (rear or {}).items():
        i = take(lane)
        ob.s[i] = (env.prev_s - gap) % env.track.length
        ob.v[i] = ob.v_nominal[i] = float(speed)
    ob._update_positions()
    env.v = float(ego_v)


class TestOvertakeStateMachine(unittest.TestCase):
    def setUp(self):
        self.env = MiniDriveEnv(seed=0, random_start=False, n_obstacles=6)
        self.env.reset(seed=0)
        self.sensor = SensorFusion()
        self.ctrl = OvertakeController({
            "opportunity_gap": 18.0,
            "min_closing": 0.5,
            "release_gap": 30.0,
            "entry_min_front": 12.0,
            "entry_min_rear": 8.0,
            "entry_min_window": 2.0,
            "abort_min_window": 1.0,
            "complete_hold_steps": 3,
            "abort_hold_steps": 2,
            "complete_hold_after": 2,
        })

    def _scan(self):
        return self.sensor.scan(self.env)

    def test_follow_when_ego_lane_is_clear(self):
        park(self.env, front={})
        cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.FOLLOW)
        self.assertFalse(cmd.available)
        self.assertFalse(cmd.lock)
        self.assertEqual(cmd.direction, 0)

    def test_follow_when_blocked_but_not_closing(self):
        park(self.env, front={1: (10.0, 8.0)}, ego_v=8.0)
        cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.FOLLOW)
        self.assertEqual(cmd.reason, "no_opportunity")

    def test_prepare_left_when_both_target_lanes_are_empty(self):
        park(self.env, front={1: (10.0, 2.0)}, ego_v=8.0)
        cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.PREPARE_LEFT)
        self.assertEqual(cmd.target_lane, 2)
        self.assertEqual(cmd.direction, 1)
        self.assertTrue(cmd.lock)
        self.assertTrue(cmd.available)
        self.assertEqual(cmd.event, "prepare")
        self.assertEqual(cmd.blocker_id, int(self._scan().front_ids[1]))

    def test_prepare_right_when_the_left_lane_is_occupied(self):
        park(self.env, front={1: (10.0, 2.0), 2: (2.0, 2.0)}, ego_v=8.0)
        frame = self._scan()
        self.assertFalse(frame.reading(2).safe_to_enter)   # a car alongside
        cmd = self.ctrl.step(frame)
        self.assertIs(cmd.phase, OvertakePhase.PREPARE_RIGHT)
        self.assertEqual(cmd.target_lane, 0)
        self.assertEqual(cmd.direction, -1)

    def test_abort_when_the_target_rear_ttc_collapses(self):
        park(self.env, front={1: (10.0, 2.0)}, ego_v=8.0)
        self.ctrl.step(self._scan())
        self.assertIs(self.ctrl.phase, OvertakePhase.PREPARE_LEFT)

        # a fast follower enters the target lane's rear window
        park(self.env, front={1: (10.0, 2.0)}, rear={2: (8.0, 14.0)}, ego_v=8.0)
        cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.ABORT)
        self.assertEqual(cmd.event, "abort")
        self.assertEqual(cmd.reason, "window")
        self.assertFalse(cmd.lock)
        self.assertIsNone(cmd.target_lane)

    def test_abort_when_the_lead_clears(self):
        park(self.env, front={1: (10.0, 2.0)}, ego_v=8.0)
        self.ctrl.step(self._scan())
        park(self.env, front={1: (40.0, 2.0)}, ego_v=8.0)
        cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.ABORT)
        self.assertEqual(cmd.reason, "cleared")

    def test_complete_needs_the_target_lane_hold_and_the_actual_pass(self):
        park(self.env, front={1: (10.0, 2.0)}, ego_v=8.0)
        cmd = self.ctrl.step(self._scan())
        blocker = cmd.blocker_id
        target = cmd.target_lane
        self.assertIsNotNone(blocker)

        # still in the ego lane: holding cannot complete
        for _ in range(4):
            self.assertIs(self.ctrl.step(self._scan()).phase, OvertakePhase.PREPARE_LEFT)

        # enter the target lane; the blocker is still ahead, so no completion
        self.env._lateral = float(self.env.lane_centers[target])
        for _ in range(4):
            cmd = self.ctrl.step(self._scan())
            self.assertIs(cmd.phase, OvertakePhase.PREPARE_LEFT)
            self.assertFalse(cmd.blocker_passed)

        # the blocker is now behind the ego -> the pass is complete
        ob = self.env.obstacles
        ob.s[int(blocker)] = (self.env.prev_s - 8.0) % self.env.track.length
        ob._update_positions()
        cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.COMPLETE)
        self.assertEqual(cmd.event, "complete")
        self.assertTrue(cmd.blocker_passed)
        self.assertEqual(cmd.reason, "blocker_passed")
        self.assertFalse(cmd.lock)
        self.assertEqual(self.ctrl.completions, 1)

    def test_complete_waits_for_the_hold(self):
        park(self.env, front={1: (10.0, 2.0)}, ego_v=8.0)
        cmd = self.ctrl.step(self._scan())
        blocker, target = cmd.blocker_id, cmd.target_lane
        self.env._lateral = float(self.env.lane_centers[target])
        ob = self.env.obstacles
        ob.s[int(blocker)] = (self.env.prev_s - 8.0) % self.env.track.length
        ob._update_positions()

        # first and second step in the lane: hold counter not yet at 3
        self.assertIs(self.ctrl.step(self._scan()).phase, OvertakePhase.PREPARE_LEFT)
        self.assertIs(self.ctrl.step(self._scan()).phase, OvertakePhase.PREPARE_LEFT)
        cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.COMPLETE)

    def test_abort_holds_then_returns_to_follow(self):
        park(self.env, front={1: (10.0, 2.0)}, ego_v=8.0)
        self.ctrl.step(self._scan())
        park(self.env, front={1: (10.0, 2.0)}, rear={2: (8.0, 14.0)}, ego_v=8.0)
        self.assertIs(self.ctrl.step(self._scan()).phase, OvertakePhase.ABORT)

        # keep the traffic free of opportunities; the hold releases to FOLLOW
        park(self.env, front={})
        for _ in range(4):
            cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.FOLLOW)

    def test_prepare_survives_a_drift_to_the_other_lane(self):
        """A committed manoeuvre keeps correcting even if the ego overshoots."""
        park(self.env, front={1: (10.0, 2.0)}, ego_v=8.0)
        cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.PREPARE_LEFT)

        self.env._lateral = float(self.env.lane_centers[0])   # drifted two lanes right
        cmd = self.ctrl.step(self._scan())
        self.assertIs(cmd.phase, OvertakePhase.PREPARE_LEFT)
        self.assertEqual(cmd.target_lane, 2)
        self.assertTrue(cmd.lock, "the lock must keep steering back to the target")

    def test_controller_does_not_mutate_the_environment(self):
        park(self.env, front={1: (10.0, 2.0)}, rear={2: (20.0, 6.0)}, ego_v=8.0)
        before = (self.env.x, self.env.y, self.env.theta, self.env.v,
                  self.env.prev_s, self.env.obstacles.s.copy(),
                  self.env.obstacles.v.copy())
        self.ctrl.step(self._scan())
        after = (self.env.x, self.env.y, self.env.theta, self.env.v,
                 self.env.prev_s, self.env.obstacles.s.copy(),
                 self.env.obstacles.v.copy())
        self.assertEqual(before[0:5], after[0:5])
        np.testing.assert_allclose(before[5], after[5])
        np.testing.assert_allclose(before[6], after[6])


def make_sm_agent(env, **overrides):
    cfg = {
        "hazard_mode": "ttc",
        "ttc_features": False,
        "action_risk": False,
        "safety_shield": True,
        "sensor_fusion": True,
        "lane_aware": True,
        "lane_clearance_features": True,
        "overtake_state_machine": True,
        "overtake_lock": True,
        "overtake_after_pass_steps": 10,
        "overtake_complete_bonus": 3.0,
        "safe_progress_weight": 0.5,
        "safe_speed_target": 8.0,
        "progress_reward_only_in_overtake_window": True,
        "overtake_progress_multiplier": 2.0,
        "ppo": {"n_steps": 8, "minibatch": 4},
    }
    cfg.update(overrides)
    agent = make_algorithm("rarl", env.obs_dim, env.n_actions, cfg=cfg, seed=0)
    agent.bind_env(env)
    return agent


class TestRARLUsesTheStateMachine(unittest.TestCase):
    def setUp(self):
        self.env = MiniDriveEnv(seed=0, random_start=False)
        self.agent = make_sm_agent(self.env)
        obs, _ = self.env.reset(seed=0)
        self.agent.reset_episode(obs)
        self.agent.last_risk.update({"ttc_front_hazard": 0.0, "ttc_rear_hazard": 0.0,
                                     "vehicle_risk": 0.0, "road_risk": 0.0})
        self.agent.last_action_risk = np.zeros(self.env.n_actions)
        self.agent._action_risk_taken = 0.0

    def _prepare(self, ego_v=8.0):
        park(self.env, front={1: (10.0, 2.0)}, ego_v=ego_v)
        self.agent._features(self.env.last_obs)
        cmd = self.agent._sm_command
        self.assertIsNotNone(cmd)
        self.assertIs(cmd.phase, OvertakePhase.PREPARE_LEFT)
        return cmd

    def _zero_risk(self):
        self.agent.last_risk.update({"ttc_front_hazard": 0.0, "ttc_rear_hazard": 0.0,
                                     "vehicle_risk": 0.0, "road_risk": 0.0,
                                     "ttc_hazard": 0.0, "total_risk": 0.0})
        self.agent._action_risk_taken = 0.0
        self.agent.last_action_risk = np.zeros(self.env.n_actions)

    def _advance(self):
        self.env.steps += 1
        return self.agent._advance_lane_state()

    def test_default_off_and_dimensions_unchanged(self):
        agent = make_algorithm("rarl", self.env.obs_dim, self.env.n_actions,
                               cfg={"ppo": {"n_steps": 8, "minibatch": 4}}, seed=0)
        agent.bind_env(self.env)
        self.assertFalse(agent.use_overtake_sm)
        self.assertEqual(agent.policy_in, BASE_POLICY_IN)
        self.assertIsNone(agent.overtake)

        # same feature flags with the SM off: identical width, so an old
        # checkpoint still loads into the state-machine build
        legacy = make_sm_agent(self.env, overtake_state_machine=False)
        self.assertEqual(legacy.policy_in, self.agent.policy_in)
        self.env.reset(seed=0)
        other = make_sm_agent(self.env)
        other.load_state_dict(legacy.state_dict())
        self.assertEqual(other.policy_in, self.agent.policy_in)

    def test_lock_forces_the_state_machine_steer(self):
        self._prepare()
        mask = self.agent._action_mask()
        self.assertEqual(int(np.sum(mask)), 1)
        self.assertTrue(mask[0], "left target lane -> hard-left is the only action")

    def test_lock_holds_the_target_lane_and_pulls_away(self):
        self._prepare()
        target = int(self.agent._sm_command.target_lane)
        self.env._lateral = float(self.env.lane_centers[target]) - 0.6
        self.env.theta = float(self.env._track_heading)
        self.env.steps += 1
        self.agent._features(self.env.last_obs)
        self._zero_risk()
        mask = self.agent._action_mask()
        self.assertTrue(mask[1], "right of the target centre -> soft left")
        self.assertEqual(int(np.sum(mask)), 1)

        self.env._lateral = float(self.env.lane_centers[target])
        self.env.theta = float(self.env._track_heading)
        self.env.steps += 1
        self.agent._features(self.env.last_obs)
        self._zero_risk()
        mask = self.agent._action_mask()
        self.assertTrue(mask[5], "centred and clear -> accelerate past the blocker")
        self.assertEqual(int(np.sum(mask)), 1)

    def test_shield_veto_aborts_instead_of_swerving(self):
        self._prepare()
        self.agent.last_risk["ttc_front_hazard"] = 0.9
        mask = self.agent._action_mask()
        self.assertIs(self.agent._sm_command.phase, OvertakePhase.ABORT)
        self.assertIsNone(self.agent._overtake_target)
        self.assertTrue(mask[6], "braking stays available")
        self.assertFalse(mask[:6].any(), "no steering with a car right ahead")

    def test_reward_requires_a_safe_complete(self):
        self._prepare()
        self._zero_risk()
        # before the manoeuvre: forward motion pays nothing, actions pay nothing
        self.assertAlmostEqual(self.agent.shaped_reward(0.0, ds=0.5), 0.0)
        self.assertAlmostEqual(self.agent.shaped_reward(0.0, ds=0.0), 0.0)

        target = self.agent._sm_command.target_lane
        blocker = self.agent._sm_command.blocker_id
        self.env._lateral = float(self.env.lane_centers[target])
        for _ in range(3):
            self.assertAlmostEqual(self._advance(), 0.0)
        # still no completion while the blocker is ahead
        ob = self.env.obstacles
        ob.s[int(blocker)] = (self.env.prev_s - 8.0) % self.env.track.length
        ob._update_positions()

        bonus = self._advance()
        self.assertAlmostEqual(bonus, self.agent.cfg["overtake_complete_bonus"])
        self.assertIs(self.agent._sm_command.phase, OvertakePhase.COMPLETE)
        self.assertEqual(self.agent.overtake_completions, 1)
        self.agent._features(self.env.last_obs)
        self._zero_risk()

        # the one-off bonus is paid once, then safe progress inside the window
        self.assertAlmostEqual(self.agent.shaped_reward(0.0, ds=0.0),
                               self.agent.cfg["overtake_complete_bonus"])
        self.assertAlmostEqual(self.agent.shaped_reward(0.0, ds=0.5), 0.5 * 0.5 * 2.0)
        self.assertAlmostEqual(self.agent.shaped_reward(0.0, ds=0.5), 0.5 * 0.5 * 2.0)

    def test_abort_pays_nothing(self):
        self._prepare()
        park(self.env, front={1: (10.0, 2.0)}, rear={2: (8.0, 14.0)})
        self.env.steps += 1
        self.assertAlmostEqual(self.agent._advance_lane_state(), 0.0)
        self.assertIs(self.agent._sm_command.phase, OvertakePhase.ABORT)
        self.agent._features(self.env.last_obs)
        self._zero_risk()
        self.assertAlmostEqual(self.agent.shaped_reward(0.0, ds=0.5), 0.0)

    def test_state_machine_checkpoint_roundtrips(self):
        self.env.reset(seed=0)
        other = make_sm_agent(self.env)
        other.load_state_dict(self.agent.state_dict())
        probe = np.random.default_rng(0).normal(size=self.agent.policy_in)
        diff = np.max(np.abs(self.agent.policy.actor.forward(probe)
                             - other.policy.actor.forward(probe)))
        self.assertLess(float(diff), 1e-12)


class TestOvertakeCurriculumEnvironments(unittest.TestCase):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _env(self, name):
        with open(os.path.join(self.ROOT, "configs", name), encoding="utf-8") as fh:
            cfg = json.load(fh)
        env = MiniDriveEnv(seed=0, **cfg)
        env.reset(seed=0)
        return env

    def test_stage1_one_slow_lead_and_empty_target_lanes(self):
        env = self._env("env_overtake_stage1_sm.json")
        self.assertEqual(env.obstacles.n, 1)
        ego_lane = env.lane_of()
        gap, lead_v = env.obstacles.lane_lead(ego_lane, env.prev_s)
        self.assertAlmostEqual(gap, 20.0, places=6)
        self.assertAlmostEqual(lead_v, 2.5, places=6)
        frame = SensorFusion().scan(env)
        for lane in (ego_lane - 1, ego_lane + 1):
            reading = frame.reading(lane)
            self.assertTrue(reading.exists)
            self.assertEqual(reading.front_gap, float("inf"))
            self.assertEqual(reading.rear_gap, float("inf"))

    def test_stage2_has_fast_rear_cars_in_both_target_lanes(self):
        env = self._env("env_overtake_stage2_sm.json")
        self.assertEqual(env.obstacles.n, 3)
        ego_lane = env.lane_of()
        frame = SensorFusion().scan(env)
        self.assertAlmostEqual(frame.reading(ego_lane).front_gap, 20.0, places=6)
        for lane in (ego_lane - 1, ego_lane + 1):
            reading = frame.reading(lane)
            self.assertAlmostEqual(reading.rear_gap, 10.0, places=6)
            self.assertAlmostEqual(reading.rear_speed, 16.0, places=6)
            self.assertGreater(reading.rear_rel_speed, 0.0, "the rear car is closing in")

    def test_stage3_is_random_mixed_traffic(self):
        env = self._env("env_overtake_stage3_sm.json")
        self.assertEqual(env.obstacles.n, 18)
        self.assertEqual(float(env.cfg["npc_lead_gap"]), 0.0)
        self.assertEqual(float(env.cfg["npc_follow_gap"]), 0.0)
        self.assertGreater(len(set(env.obstacles.lane.tolist())), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
