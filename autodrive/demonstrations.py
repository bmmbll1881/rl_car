"""Expert demonstrations for behaviour cloning.

The expert is the project's scripted controller.  Two variants exist:

``MemorylessExpertDriver`` (default)
    every action is a pure function of the current simulator state -- the lane
    choice comes from :class:`autodrive.sensors.SensorFusion` and the lateral
    target is that lane's centre, with no latched lane, no slewed offset and no
    cooldown.  This is what makes the labels learnable: the observation the
    learner sees determines the action, so behaviour cloning is not asked to
    infer hidden variables.
``PurePursuitDriver`` (``memoryless=False``)
    the legacy driver, kept for comparison.  Its ``target_lane`` /
    ``current_offset`` / ``_cooldown`` make the label depend on history, so its
    demonstrations are *not* Markov in the policy input.

Every label in a demonstration set is the action the deterministic driver
actually took in the simulator; nothing is inferred, re-labelled or synthesised.

The observations stored are **the learner's own policy input**.  For RARL that
means ``agent._features(obs)`` -- environment observation, risk features and,
when the flags are on, the SensorFusion / lane blocks and the belief block --
so the width matches the policy a caller wants to warm-start.  A sensor-fusion
policy therefore trains on sensor-fusion features, not on raw 29-dim lidar
observations.

The **safety mask** in force at each demonstration step is stored alongside the
observation.  ``PPO.pretrain`` applies the same masked logits the PPO update
uses, so the supervised distribution is the one the policy is actually allowed
to sample from.  A demonstration whose action the mask forbids is counted
(``mask_conflicts``) and the label is re-admitted for that sample only, so the
data stays learnable without pretending the conflict never happened.

Because the whole point of the exercise is *overtaking*, the collector also
reports whether the expert produced any of it:

``overtake_opportunity_steps``
    steps on which the lane planner saw a safe adjacent lane (SensorFusion's
    ``target_available``) -- i.e. the same signal the RARL policy receives;
``blocked_steps``
    steps with a lead car closer than ``lane_block_distance`` in the ego lane;
``blocked_lane_changes``
    lane changes the expert made *while* its lane was blocked;
``overtakes_completed``
    of those, the number where the car that blocked the old lane ended up
    behind the ego -- the manoeuvre the task actually cares about.

If those counters come back zero, the expert did not demonstrate an overtake in
that environment, and the caller should say so rather than treat the data as an
overtake demonstration.
"""
from __future__ import annotations

import numpy as np

from .baselines import MemorylessExpertDriver, PurePursuitDriver
from .logging_utils import advance_agent, reset_agent
from .sensors import SensorFusion


def _signed_gap(env, npc_id: int) -> float:
    """Arc distance from the ego to NPC ``npc_id`` (negative == behind)."""
    if npc_id is None or not 0 <= int(npc_id) < env.obstacles.n:
        return float("-inf")
    length = float(env.track.length)
    gap = (float(env.obstacles.s[int(npc_id)]) - float(env.prev_s)) % length
    return gap - length if gap > length / 2.0 else gap


def _lane_state(sensor, agent, env, block_distance: float):
    """``(opportunity, blocked, lead_gap)`` plus the frame that produced them.

    RARL already scanned the traffic for the policy input; reuse that frame so
    the counters describe the same picture the learner sees.  A plain PPO agent
    has no sensor, so a read-only :class:`SensorFusion` scan is used instead --
    it is the same sensor implementation, not a different one.
    """
    frame = getattr(agent, "last_sensor", None)
    if frame is None:
        frame = sensor.scan(env)
    return (bool(frame.target_available),
            float(frame.ego_lead_gap) < block_distance,
            float(frame.ego_lead_gap),
            frame)


def _policy_mask(agent, n_actions: int) -> np.ndarray:
    """The action mask the learner would apply in the current state.

    ``RARL.action_mask`` is the public form of exactly the mask ``act()`` passes
    to PPO; a plain PPO agent (or any agent without one) gets "no mask"."""
    fn = getattr(agent, "action_mask", None)
    if not callable(fn):
        return np.ones(n_actions, dtype=bool)
    mask = np.asarray(fn(), dtype=bool).reshape(-1)
    if mask.size != n_actions:
        return np.ones(n_actions, dtype=bool)
    return mask


def collect_expert_demonstrations(env, agent, episodes: int, seed_base: int = 30_000,
                                  block_distance: float | None = None,
                                  max_steps: int | None = None,
                                  memoryless: bool = True,
                                  keep_clean_only: bool = False):
    """Drive ``episodes`` episodes with the scripted expert and record the policy input.

    Parameters
    ----------
    env, agent
        ``agent`` must be bound to ``env`` (``bind_env``) so its feature builder
        can read the traffic.  Plain PPO agents without ``_features`` are also
        supported; their raw observation is used as the policy input.
    block_distance
        Ego-lane gap below which the lane counts as blocked.  Defaults to the
        agent's ``lane_block_distance`` (18 m), then 18 m.
    max_steps
        Optional per-episode step cap (the environment's own limit applies
        otherwise).
    memoryless
        Use :class:`MemorylessExpertDriver` (default) so the label is a function
        of the observation.  ``False`` restores the legacy stateful driver.
    keep_clean_only
        Drop every sample from an episode that crashed or collided.  A crash is
        not a demonstration of the target behaviour, so a caller that wants
        *driving* data should set this.  The counters report how many samples
        and episodes were dropped.

    Returns
    -------
    ``(observations, actions, stats)`` where ``observations`` is
    ``[n_samples, policy_in]``, ``actions`` is ``[n_samples]`` and ``stats``
    holds the counters described in the module docstring plus the per-sample
    ``masks`` under ``stats["masks"]`` and one summary per episode.
    """
    if block_distance is None:
        block_distance = float(getattr(agent, "cfg", {}).get("lane_block_distance", 18.0))
    block_distance = float(block_distance)

    driver = MemorylessExpertDriver(env, block_distance=block_distance) if memoryless \
        else PurePursuitDriver(env)
    sensor = SensorFusion({"block_distance": block_distance})
    use_features = callable(getattr(agent, "_features", None))
    n_actions = int(env.n_actions)

    observations, actions, masks = [], [], []
    stats = {
        "expert": driver.name,
        "memoryless": bool(memoryless),
        "keep_clean_only": bool(keep_clean_only),
        "episodes": 0,
        "episodes_kept": 0,
        "episodes_dropped_dirty": 0,
        "samples_dropped_dirty": 0,
        "steps": 0,
        "feature_dim": int(getattr(agent, "policy_in", env.obs_dim)),
        "block_distance": block_distance,
        "mask_conflicts": 0,
        "masked_samples": 0,
        "overtake_opportunity_steps": 0,
        "blocked_steps": 0,
        "lane_changes": 0,
        "blocked_lane_changes": 0,
        "overtakes_completed": 0,
        "episode_stats": [],
    }

    limit = int(max_steps) if max_steps else int(env.max_steps)

    for episode in range(int(episodes)):
        obs, _ = env.reset(seed=seed_base + episode)
        reset_agent(agent, obs)
        driver.reset()
        ep = {"episode": episode, "seed": seed_base + episode, "steps": 0,
              "overtake_opportunity_steps": 0, "blocked_steps": 0,
              "lane_changes": 0, "blocked_lane_changes": 0, "overtakes_completed": 0,
              "success": 0, "crash": 0, "collision": 0, "progress": 0.0,
              "kept": 0}

        prev_lane = int(env.lane_of())
        blocker_id = None
        done = False
        info = {"success": False, "crash": False, "collision": False, "progress": 0.0}
        ep_obs, ep_actions, ep_masks = [], [], []

        while not done:
            if use_features:
                policy_obs = agent._features(obs)
            else:
                policy_obs = np.asarray(obs, dtype=float)

            opportunity, blocked, _gap, frame = _lane_state(
                sensor, agent, env, block_distance)
            if opportunity:
                ep["overtake_opportunity_steps"] += 1
            if blocked:
                ep["blocked_steps"] += 1
            lead_id = int(frame.front_ids[frame.ego_lane])

            mask = _policy_mask(agent, n_actions)
            action = int(driver.act())

            ep_obs.append(np.array(policy_obs, dtype=float, copy=True))
            ep_actions.append(action)
            ep_masks.append(mask.copy())
            if not mask[action]:
                stats["mask_conflicts"] += 1
            if not mask.all():
                stats["masked_samples"] += 1

            next_obs, _r, term, trunc, info = env.step(action)
            advance_agent(agent, next_obs, term, trunc)

            lane_now = int(env.lane_of())
            if lane_now != prev_lane:
                ep["lane_changes"] += 1
                if blocked:
                    ep["blocked_lane_changes"] += 1
                    # the car in the lane we just left is the one we are passing
                    blocker_id = lead_id if lead_id >= 0 else None
                prev_lane = lane_now

            if blocker_id is not None:
                if _signed_gap(env, blocker_id) < -3.0:      # overtake_pass_margin
                    ep["overtakes_completed"] += 1
                    blocker_id = None

            ep["steps"] += 1
            obs = next_obs
            done = term or trunc or ep["steps"] >= limit

        ep["success"] = int(info.get("success", False))
        ep["crash"] = int(info.get("crash", False))
        ep["collision"] = int(info.get("collision", False))
        ep["progress"] = float(info.get("progress", 0.0))

        dirty = bool(ep["crash"] or ep["collision"])
        keep = not (keep_clean_only and dirty)
        if keep:
            observations.extend(ep_obs)
            actions.extend(ep_actions)
            masks.extend(ep_masks)
            ep["kept"] = 1
            stats["episodes_kept"] += 1
            for key in ("steps", "overtake_opportunity_steps", "blocked_steps",
                        "lane_changes", "blocked_lane_changes", "overtakes_completed"):
                stats[key] += ep[key]
        else:
            stats["episodes_dropped_dirty"] += 1
            stats["samples_dropped_dirty"] += len(ep_actions)
        stats["episode_stats"].append(ep)
        stats["episodes"] += 1

    observations = np.asarray(observations, dtype=float)
    actions = np.asarray(actions, dtype=np.int64)
    masks = np.asarray(masks, dtype=bool)
    if observations.size:
        stats["feature_dim"] = int(observations.shape[1])
    else:
        masks = np.zeros((0, n_actions), dtype=bool)
    stats["masks"] = masks
    return observations, actions, stats
