"""Common algorithm interface.

Every learning algorithm exposes the same four hooks, which is what makes the
agent swappable (``train.py --algo dqn`` vs ``--algo ppo``) without touching the
training loop or the environment:

    act(obs, greedy)                      -> int action
    update(obs, a, r, next_obs, done...)  -> dict of loss metrics or None
    state_dict() / load_state_dict()      -> checkpointing
    stats()                               -> extra scalars for the log (eps, lr, ...)

Adding a new algorithm = one new file implementing this interface + one line in
``algos/__init__.py``.  Nothing else in the pipeline changes.
"""
from __future__ import annotations

from typing import Optional


class Algorithm:
    name = "base"

    def __init__(self, obs_dim: int, n_actions: int, cfg: Optional[dict] = None, seed: int = 0):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = dict(cfg or {})
        self.seed = seed
        self.total_steps = 0

    # -- acting ---------------------------------------------------------------
    def act(self, obs, greedy: bool = False) -> int:
        raise NotImplementedError

    # -- learning -------------------------------------------------------------
    def update(self, obs, action, reward, next_obs, terminated: bool, truncated: bool, info=None):
        """Consume one transition.  Return loss scalars when a gradient step ran."""
        raise NotImplementedError

    def on_episode_end(self, info=None):
        """Optional hook (PPO uses it to close the advantage computation)."""

    def bind_env(self, env):
        """Optional hook for algorithms that need the environment itself.

        RARL uses it to install an adversarial vehicle controller
        (``env.set_adversary``); plain learners ignore it.
        """

    # -- persistence ----------------------------------------------------------
    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError

    def stats(self) -> dict:
        return {}
