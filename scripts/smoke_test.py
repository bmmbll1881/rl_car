"""Quick smoke test used during development (not part of the deliverable API)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autodrive.env import MiniDriveEnv
from autodrive.baselines import PurePursuitDriver


def main():
    env = MiniDriveEnv(seed=0)
    print(env.describe())
    print("track bbox:", env.track.bbox.tolist())
    print("min turn radius: %.2f m" % env.track.min_turn_radius())

    # --- random rollout -----------------------------------------------------
    obs, info = env.reset(seed=0)
    print("obs shape", obs.shape, "lidar part", np.round(obs[:9], 2), "state part", np.round(obs[9:], 2))
    total = 0.0
    for t in range(50):
        obs, r, term, trunc, info = env.step(env.rng.integers(env.n_actions))
        total += r
        if term or trunc:
            break
    print(f"random 50 steps: return {total:7.2f} crashed={info['crash']} speed={info['speed']:.2f}")

    # --- timings ------------------------------------------------------------
    env.reset(seed=1)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        obs, r, term, trunc, info = env.step(2)
        if term or trunc:
            env.reset(seed=1)
    dt = (time.perf_counter() - t0) / n
    print(f"env step: {dt*1e6:.0f} us/step -> {1/dt:,.0f} steps/s")

    # --- expert controller --------------------------------------------------
    drv = PurePursuitDriver(env)
    drv.reset()
    obs, info = env.reset(seed=0)
    ret, done = 0.0, False
    while not done:
        a = drv.act()
        obs, r, term, trunc, info = env.step(a)
        ret += r
        done = term or trunc
    print(
        f"expert: return {ret:8.2f} steps {env.steps:5d} progress {env.s_total:6.1f}/"
        f"{env.track.length:.1f} m success={info['success']} crash={info['crash']}"
    )


if __name__ == "__main__":
    main()
