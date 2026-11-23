"""Deep Q-Network (double DQN + soft target updates + Huber loss), pure numpy."""
from __future__ import annotations

import numpy as np

from ..nn import MLP, Adam
from .base import Algorithm

DEFAULT_CONFIG = {
    "hidden": 64,
    "lr": 5e-4,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_size": 100_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 40_000,
    "target_tau": 0.01,
    "train_every": 4,
    "grad_updates_per_step": 1,
    "huber_delta": 1.0,
    "max_grad_norm": 10.0,
    "double": True,
    # multiply rewards before they enter the replay buffer.  With crash = -150 and
    # step = -0.1 the true Q is ~-180 while the network starts at 0, and Adam can
    # only move an output by ~lr per update, so at 1x scale the critic never gets
    # there within a normal budget (see docs: DQN diagnosis).
    "reward_scale": 1.0,
}


class ReplayBuffer:
    """Fixed-size ring buffer of transitions (fast, no python objects)."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng):
        idx = rng.integers(0, self.size, size=batch)
        return (
            self.obs[idx].astype(np.float64),
            self.action[idx],
            self.reward[idx].astype(np.float64),
            self.next_obs[idx].astype(np.float64),
            self.done[idx].astype(np.float64),
        )


class DQN(Algorithm):
    name = "dqn"

    def __init__(self, obs_dim, n_actions, cfg=None, seed=0):
        c = dict(DEFAULT_CONFIG)
        c.update(cfg or {})
        super().__init__(obs_dim, n_actions, c, seed)
        self.cfg = c
        h = int(c["hidden"])
        self.q = MLP([obs_dim, h, h, n_actions], hidden="tanh", seed=seed)
        self.q_target = MLP([obs_dim, h, h, n_actions], hidden="tanh", seed=seed + 1)
        self.q_target.set_state(self.q.get_state())
        self.opt = Adam(self.q.params, lr=c["lr"], max_grad_norm=c["max_grad_norm"])
        self.buffer = ReplayBuffer(c["buffer_size"], obs_dim)
        self.rng = np.random.default_rng(seed)
        self.epsilon = float(c["eps_start"])
        self.last_loss = {}

    # ------------------------------------------------------------------ acting
    def _epsilon(self) -> float:
        frac = min(1.0, self.total_steps / float(self.cfg["eps_decay_steps"]))
        return float(self.cfg["eps_start"] + frac * (self.cfg["eps_end"] - self.cfg["eps_start"]))

    def act(self, obs, greedy: bool = False) -> int:
        self.epsilon = self._epsilon()
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q.forward(obs)
        return int(np.argmax(q))

    # ----------------------------------------------------------------- learning
    def update(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        self.total_steps += 1
        self.buffer.add(obs, action, float(reward) * float(self.cfg["reward_scale"]),
                        next_obs, terminated)
        c = self.cfg
        if self.buffer.size < max(c["batch_size"], 1000):
            return None
        if self.total_steps % int(c["train_every"]) != 0:
            return None

        losses = []
        for _ in range(int(c["grad_updates_per_step"])):
            s, a, r, s2, done = self.buffer.sample(int(c["batch_size"]), self.rng)
            q2_online = self.q.forward(s2)
            if c["double"]:
                a_star = np.argmax(q2_online, axis=1)
                q2 = self.q_target.forward(s2)[np.arange(len(a_star)), a_star]
            else:
                q2 = self.q_target.forward(s2).max(axis=1)
            target = r + c["gamma"] * (1.0 - done) * q2

            q_sa_all = self.q.forward(s)
            q_sa = q_sa_all[np.arange(len(a)), a]
            err = q_sa - target
            delta = c["huber_delta"]
            # Huber loss  L = 0.5*e^2 if |e|<=d else d*(|e|-0.5*d)
            dq_sa = np.where(np.abs(err) <= delta, err, delta * np.sign(err)) / len(a)

            # gradient flows only through the chosen action's Q value
            dq = np.zeros_like(q_sa_all)
            dq[np.arange(len(a)), a] = dq_sa
            grads = self.q.backward(dq)
            self.opt.step(grads)

            # Polyak target update
            tau = c["target_tau"]
            for wt, w in zip(self.q_target.params, self.q.params):
                wt *= (1.0 - tau)
                wt += tau * w
            losses.append(float(np.mean(np.abs(err))))

        self.last_loss = {"loss": float(np.mean(losses)), "td_abs": float(np.mean(losses))}
        return self.last_loss

    # ------------------------------------------------------------- persistence
    def state_dict(self):
        st = {"q_" + k: v for k, v in self.q.get_state().items()}
        st.update({"qt_" + k: v for k, v in self.q_target.get_state().items()})
        st["meta"] = np.array([self.obs_dim, self.n_actions, self.total_steps, self.seed])
        return st

    def load_state_dict(self, state):
        self.q.set_state({k[2:]: v for k, v in state.items() if k.startswith("q_")})
        self.q_target.set_state({k[3:]: v for k, v in state.items() if k.startswith("qt_")})
        self.total_steps = int(state["meta"][2])

    def stats(self):
        return {"epsilon": self.epsilon, "buffer": self.buffer.size}
