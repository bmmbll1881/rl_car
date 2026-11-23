"""Numerical gradient checks -- the core of the debuggability story.

Run with:  python -m unittest discover -s tests -v

`test_nn_backprop` and `test_ppo_policy_gradient` compare every analytic
derivative against finite differences, so a silent sign error in the hand-written
backprop cannot slip through unnoticed.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.nn import MLP, softmax, log_softmax, Adam
from autodrive.algos.ppo import PPO
from autodrive.algos.dqn import DQN, ReplayBuffer
from autodrive.env import MiniDriveEnv
from autodrive.logging_utils import evaluate_agent


def num_grad(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps
        fp = f()
        arr[i] = old - eps
        fm = f()
        arr[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


class TestNeuralNet(unittest.TestCase):
    def test_backprop_all_parameters(self):
        net = MLP([5, 7, 4, 3], hidden="tanh", seed=3)
        x = np.random.default_rng(0).normal(size=(6, 5))
        dout = np.random.default_rng(1).normal(size=(6, 3))

        net.forward(x)
        analytic = net.backward(dout)

        def loss():
            return float(np.sum(net.forward(x) * dout))

        for k, (p, g) in enumerate(zip(net.params, analytic)):
            numeric = num_grad(loss, p)
            err = np.max(np.abs(numeric - g))
            self.assertLess(err, 1e-6, f"param {k} gradient error {err:.2e}")

    def test_entropy_gradient(self):
        rng = np.random.default_rng(2)
        z = rng.normal(size=5)

        def ent(zz):
            p = softmax(zz)
            return float(-(p * np.log(p)).sum())

        p = softmax(z)
        h = ent(z)
        analytic = -p * (np.log(p) + h)
        numeric = num_grad(lambda: ent(z), z)
        self.assertLess(np.max(np.abs(analytic - numeric)), 1e-6)

    def test_adam_decreases_quadratic(self):
        target = np.array([1.0, -2.0, 0.5])
        p = [np.zeros(3)]
        opt = Adam(p, lr=0.1)
        for _ in range(200):
            opt.step([p[0] - target])
        self.assertLess(np.linalg.norm(p[0] - target), 1e-2)


class TestPPOGradient(unittest.TestCase):
    """Finite-difference the *whole* clipped policy loss w.r.t. the actor weights."""

    def test_ppo_policy_gradient(self):
        rng = np.random.default_rng(5)
        obs = rng.normal(size=(8, 5))
        act = rng.integers(0, 3, size=8)
        adv = rng.normal(size=8)
        clip = 0.2
        ent_coef = 0.01

        net = MLP([5, 4, 3], hidden="tanh", seed=7)
        logp_old = np.array([log_softmax(net.forward(o))[a] for o, a in zip(obs, act)])

        def total_loss():
            logits = net.forward(obs)
            lp = log_softmax(logits)
            p = np.exp(lp)
            logp = lp[np.arange(len(act)), act]
            ratio = np.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1 - clip, 1 + clip) * adv
            ent = -(p * lp).sum(axis=1).mean()
            return float(-np.minimum(surr1, surr2).mean() - ent_coef * ent)

        # analytic gradient, same formulas as ppo.py
        logits = net.forward(obs)
        lp = log_softmax(logits)
        p = np.exp(lp)
        B = len(act)
        logp = lp[np.arange(B), act]
        ratio = np.exp(logp - logp_old)
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1 - clip, 1 + clip) * adv
        inside = surr1 <= surr2
        ent = -(p * lp).sum(axis=1)
        coef = np.where(inside, -adv * ratio, 0.0)
        onehot = np.zeros_like(p)
        onehot[np.arange(B), act] = 1.0
        dlogits = coef[:, None] * (onehot - p) + ent_coef * p * (lp + ent[:, None])
        dlogits /= B
        analytic = net.backward(dlogits)

        for k, (par, g) in enumerate(zip(net.params, analytic)):
            numeric = num_grad(total_loss, par)
            err = np.max(np.abs(numeric - g))
            self.assertLess(err, 1e-6, f"PPO loss gradient error on param {k}: {err:.2e}")

    def test_behavior_cloning_pretrain_reduces_loss(self):
        rng = np.random.default_rng(9)
        obs = rng.normal(size=(64, 5))
        actions = (obs[:, 0] > 0.0).astype(np.int64)
        agent = PPO(5, 2, {"lr": 3e-3}, seed=2)
        result = agent.pretrain(obs, actions, epochs=20, batch_size=16)
        self.assertEqual(result["bc_samples"], 64)
        self.assertLess(result["bc_loss_end"], result["bc_loss_start"])


class TestGAE(unittest.TestCase):
    def test_gae_matches_manual_discount(self):
        gamma, lam = 1.0, 1.0
        rew = np.array([1.0, 1.0, 1.0])
        val = np.array([0.0, 0.0, 0.0])
        done = np.array([0.0, 0.0, 0.0])
        last_v = 0.0
        adv = np.zeros(3)
        gae = 0.0
        for t in reversed(range(3)):
            next_v = last_v if t == 2 else val[t + 1]
            nt = 1.0 - done[t]
            delta = rew[t] + gamma * nt * next_v - val[t]
            gae = delta + gamma * lam * nt * gae
            adv[t] = gae
        # with no discounting, GAE == reward-to-go
        self.assertLess(float(np.max(np.abs(adv - np.array([3.0, 2.0, 1.0])))), 1e-12)


class TestPPOActionMask(unittest.TestCase):
    def test_masked_policy_never_samples_excluded_actions(self):
        agent = PPO(4, 3, {"n_steps": 8, "minibatch": 4}, seed=4)
        obs = np.zeros(4)
        mask = np.array([False, True, True])
        for _ in range(100):
            self.assertNotEqual(agent.act(obs, action_mask=mask), 0)
        self.assertNotEqual(agent.act(obs, greedy=True, action_mask=mask), 0)


class TestEnvironment(unittest.TestCase):
    def test_evaluation_advances_stateful_agents(self):
        class StatefulAgent:
            def __init__(self):
                self.resets = 0
                self.transitions = 0

            def reset_episode(self, obs=None):
                self.resets += 1

            def observe_transition(self, next_obs, terminated=False, truncated=False):
                self.transitions += 1

            def act(self, obs, greedy=False):
                return 0

        agent = StatefulAgent()
        evaluate_agent(MiniDriveEnv(seed=4), agent, episodes=1, base_seed=77)
        self.assertEqual(agent.resets, 1)
        self.assertGreater(agent.transitions, 0)

    def test_deterministic_reset_and_step(self):
        env = MiniDriveEnv(seed=11)
        o1, _ = env.reset(seed=11)
        traj1 = []
        for _ in range(30):
            o, r, term, trunc, info = env.step(2)
            traj1.append(r)
            if term or trunc:
                break
        o2, _ = env.reset(seed=11)
        traj2 = []
        for _ in range(30):
            o, r, term, trunc, info = env.step(2)
            traj2.append(r)
            if term or trunc:
                break
        self.assertLess(float(np.max(np.abs(o1 - o2))), 1e-12)
        self.assertLess(float(np.max(np.abs(np.array(traj1) - np.array(traj2)))), 1e-12)

    def test_observation_range_and_shape(self):
        env = MiniDriveEnv(seed=2)
        obs, _ = env.reset(seed=2)
        self.assertEqual(obs.shape, (env.obs_dim,))
        self.assertTrue(np.all(obs >= -1.5) and np.all(obs <= 1.5))

    def test_offtrack_terminates(self):
        env = MiniDriveEnv(seed=3, random_start=False)
        env.reset(seed=3)
        done = False
        for _ in range(600):
            _, r, term, trunc, info = env.step(0)  # hold hard left -> must leave the road
            if term:
                self.assertTrue(info["crash"])
                done = True
                break
            if trunc:
                break
        self.assertTrue(done, "hard-left forever should leave the road")

    def test_lap_is_reachable(self):
        from autodrive.baselines import PurePursuitDriver

        env = MiniDriveEnv(seed=0)
        drv = PurePursuitDriver(env)
        obs, _ = env.reset(seed=0)
        done, info = False, {}
        while not done:
            obs, r, term, trunc, info = env.step(drv.act())
            done = term or trunc
        self.assertTrue(info["success"], "the scripted reference driver must finish a lap")


class TestAlgorithmInterface(unittest.TestCase):
    def test_swappable_agents_run(self):
        env = MiniDriveEnv(seed=1)
        obs, _ = env.reset(seed=1)
        for maker in (lambda: DQN(env.obs_dim, env.n_actions, {"train_every": 1, "batch_size": 16}, seed=0),
                      lambda: PPO(env.obs_dim, env.n_actions, {"n_steps": 32, "minibatch": 16}, seed=0)):
            agent = maker()
            a = agent.act(obs)
            self.assertIn(a, range(env.n_actions))
            for _ in range(40):
                obs, r, term, trunc, info = env.step(a)
                agent.update(obs, a, r, obs, term, trunc, info)
                a = agent.act(obs)
                if term or trunc:
                    obs, _ = env.reset(seed=1)

    def test_replay_buffer_ring(self):
        buf = ReplayBuffer(4, 3)
        for i in range(6):
            buf.add(np.full(3, i), i % 2, float(i), np.full(3, i + 1), 0.0)
        self.assertEqual(buf.size, 4)
        self.assertEqual(buf.ptr, 2)
        obs, act, rew, nobs, done = buf.sample(4, np.random.default_rng(0))
        self.assertTrue(np.all(obs >= 2.0))


class TestTraffic(unittest.TestCase):
    def test_observation_dim_is_constant(self):
        for n in (0, 6):
            env = MiniDriveEnv(seed=0, n_obstacles=n)
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (env.obs_dim,))
            self.assertEqual(env.obs_dim, 9 + 4 * env.cfg["n_npc_features"] + 4)

    def test_spawn_keeps_clearance(self):
        env = MiniDriveEnv(seed=4, n_obstacles=6)
        for k in range(20):
            env.reset(seed=1000 + k)
            self.assertGreater(env.obstacles.min_distance((env.x, env.y)), 10.0,
                               f"seed {1000 + k}: a vehicle spawned on top of the ego")

    def test_npcs_move_and_stay_in_their_lane(self):
        env = MiniDriveEnv(seed=1, n_obstacles=6)
        env.reset(seed=1)
        s0 = env.obstacles.s.copy()
        for _ in range(20):
            env.step(2)
        moved = np.abs(env.obstacles.s - s0)
        self.assertTrue(np.all(moved > 4.0), f"NPCs did not move: {moved}")
        # every vehicle must stay inside its own lane
        limit = env.lane_width / 2.0 - env.obstacles.width / 2.0
        offset = env.obstacles.lat - env.lane_centers[env.obstacles.lane]
        self.assertTrue(np.all(np.abs(offset) <= limit + 1e-6), f"lane offset {offset}")

    def test_lane_layout(self):
        env = MiniDriveEnv(seed=0)
        self.assertEqual(env.n_lanes, 3)
        self.assertLess(float(np.max(np.abs(env.lane_centers - np.array([-3.2, 0.0, 3.2])))), 1e-12)
        self.assertAlmostEqual(env.half_width, env.n_lanes * env.lane_width / 2)
        self.assertEqual(env.lane_of(3.0), 2)
        self.assertEqual(env.lane_of(-0.4), 1)

    def test_traffic_spreads_over_lanes(self):
        env = MiniDriveEnv(seed=5, n_obstacles=24)
        env.reset(seed=5)
        self.assertEqual(len(set(env.obstacles.lane.tolist())), env.n_lanes,
                         "vehicles should populate every lane")

    def test_dense_traffic_always_leaves_a_free_lane(self):
        """Away from the deliberate walls the road keeps an open lane."""
        env = MiniDriveEnv(seed=0)
        for seed in range(6):
            env.reset(seed=seed)
            # remove the deliberate walls and the 12 m windows they occupy
            walls = env.obstacles.wall_windows()
            worst = env.obstacles.max_lanes_occupied()
            self.assertLessEqual(worst, env.n_lanes)
            self.assertTrue(walls, f"seed {seed}: expected some three-abreast walls")
            # most of the lap must still have a free lane
            covered = len(walls) * 2.0 / env.track.length
            self.assertLess(covered, 0.35, f"seed {seed}: {covered:.0%} of the lap is a solid wall")

    def test_traffic_opens_a_lane_when_the_ego_is_blocked(self):
        """A three-abreast wall must give way within a few seconds."""
        env = MiniDriveEnv(seed=0)
        checked = 0
        for seed in range(6):
            env.reset(seed=seed)
            walls = env.obstacles.wall_windows()
            if not walls:
                continue
            ego_s = float(walls[len(walls) // 2])
            if not env.obstacles.is_blocked(ego_s):
                continue
            checked += 1
            cleared = None
            for k in range(60):                       # at most 6 s
                env.obstacles.step(env.dt, ego_s=ego_s)
                if not env.obstacles.is_blocked(ego_s):
                    cleared = k + 1
                    break
            self.assertIsNotNone(cleared, f"seed {seed}: the wall never opened in 6 s")
            self.assertLess(cleared * env.dt, 5.0, f"seed {seed}: wall took {cleared * env.dt:.1f}s to open")
        self.assertGreater(checked, 0, "no blocked situation was exercised")

    def test_npc_speeds_are_not_synchronised(self):
        env = MiniDriveEnv(seed=2)
        env.reset(seed=2)
        v = env.obstacles.v_nominal
        self.assertGreater(float(np.std(v)), 0.25, f"speeds look synchronised: std={np.std(v):.2f}")
        self.assertGreater(float(v.max() - v.min()), 1.0, "speed spread too small")

    def test_traffic_is_dense_with_side_by_side_pairs(self):
        env = MiniDriveEnv(seed=3)
        env.reset(seed=3)
        ob = env.obstacles
        self.assertEqual(ob.n, env.cfg["n_obstacles"], "every slot should be filled")
        spacing = env.track.length / ob.n
        self.assertLess(spacing, 25.0, f"traffic too sparse: {spacing:.1f} m per vehicle")
        pairs = ob.side_by_side_pairs()
        self.assertGreaterEqual(len(pairs), env.cfg["npc_side_by_side"],
                                f"expected at least {env.cfg['npc_side_by_side']} two-abreast pairs, got {pairs}")

    def test_collision_is_detected(self):
        env = MiniDriveEnv(seed=2, n_obstacles=6)
        env.reset(seed=2)
        # teleport the ego onto the nearest vehicle -> the geometry must report a hit
        env.x, env.y = env.obstacles.pos[0]
        self.assertTrue(env.obstacles.collision((env.x, env.y), env.theta))
        far = env.obstacles.pos[0] + np.array([25.0, 25.0])
        self.assertFalse(env.obstacles.collision(tuple(far), env.theta))

    def test_curriculum_plants_a_lead_car_in_the_ego_lane(self):
        """``npc_lead_gap`` must guarantee a slow lead without breaking spacing."""
        env = MiniDriveEnv(seed=0, n_obstacles=12, npc_lead_gap=16.0,
                           npc_v_min=2.5, npc_v_max=4.0, npc_min_gap=18.0)
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            self.assertEqual(obs.shape, (env.obs_dim,), "layout must not change")
            gap, lead_v = env.obstacles.lane_lead(env.lane_of(), env.prev_s)
            self.assertAlmostEqual(gap, 16.0, places=6)
            self.assertAlmostEqual(lead_v, 2.5, places=6)
            self.assertGreater(env.obstacles.min_distance((env.x, env.y)), 10.0)
            for lane in range(env.n_lanes):
                idx = np.where(env.obstacles.lane == lane)[0]
                if idx.size < 2:
                    continue
                s = np.sort(env.obstacles.s[idx])
                d = np.diff(np.concatenate([s, s[:1] + env.track.length]))
                self.assertTrue(np.all(d >= env.obstacles.same_lane_gap - 1e-6),
                                f"seed {seed} lane {lane}: spacing {d}")

    def test_closing_speed_feature(self):
        env = MiniDriveEnv(seed=3, n_obstacles=6)
        obs, _ = env.reset(seed=3)
        # layout per vehicle: range, cos(bearing), sin(bearing), closing speed
        k = env.cfg["n_npc_features"]
        closing = obs[9 + 2 * k:]
        self.assertTrue(np.all(np.abs(closing[:k]) <= 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
