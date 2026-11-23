"""Algorithm registry -- the single place a new algorithm gets plugged in."""

from .base import Algorithm
from .dqn import DQN
from .ppo import PPO
from .rarl import RARL

REGISTRY = {
    "dqn": DQN,
    "ppo": PPO,
    "rarl": RARL,
}


def make_algorithm(name: str, obs_dim: int, n_actions: int, cfg=None, seed: int = 0) -> Algorithm:
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"unknown algorithm '{name}', available: {sorted(REGISTRY)}")
    return REGISTRY[key](obs_dim, n_actions, cfg=cfg, seed=seed)


__all__ = ["Algorithm", "DQN", "PPO", "RARL", "REGISTRY", "make_algorithm"]
