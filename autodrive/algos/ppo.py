"""Proximal Policy Optimization (clipped surrogate + GAE), pure numpy.

Everything is explicit: the policy gradient, the entropy bonus and the value
regression all have a hand-written backward pass, so ``tests/test_gradients.py``
can finite-difference check them.
"""
from __future__ import annotations

import numpy as np

from ..nn import MLP, Adam, softmax, log_softmax
from .base import Algorithm

DEFAULT_CONFIG = {
    "hidden": 64,
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip": 0.2,
    "n_steps": 2048,
    "epochs": 8,
    "minibatch": 256,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_adv": True,
    # behaviour cloning (``PPO.pretrain``) only; PPO training never reads it
    "bc_lr": 1e-3,
}


class PPO(Algorithm):
    name = "ppo"

    def __init__(self, obs_dim, n_actions, cfg=None, seed=0):
        c = dict(DEFAULT_CONFIG)
        c.update(cfg or {})
        super().__init__(obs_dim, n_actions, c, seed)
        self.cfg = c
        h = int(c["hidden"])
        self.actor = MLP([obs_dim, h, h, n_actions], hidden="tanh", seed=seed)
        self.critic = MLP([obs_dim, h, h, 1], hidden="tanh", seed=seed + 7)
        self.actor_opt = Adam(self.actor.params, lr=c["lr"], max_grad_norm=c["max_grad_norm"])
        self.critic_opt = Adam(self.critic.params, lr=c["lr"], max_grad_norm=c["max_grad_norm"])
        self.rng = np.random.default_rng(seed + 101)

        n = int(c["n_steps"])
        self.buf_obs = np.zeros((n, obs_dim))
        self.buf_act = np.zeros(n, dtype=np.int64)
        self.buf_logp = np.zeros(n)
        self.buf_val = np.zeros(n)
        self.buf_rew = np.zeros(n)
        self.buf_done = np.zeros(n)
        # behaviour mask per transition: a safety shield can forbid actions, and
        # the same mask has to be applied when the ratio is recomputed
        self.buf_mask = np.ones((n, n_actions), dtype=bool)
        self.buf_ptr = 0
        self.last_next_obs = np.zeros(obs_dim)

        self._last_logp = 0.0
        self._last_value = 0.0
        self._last_mask = np.ones(n_actions, dtype=bool)
        self.last_loss = {}
        self.update_count = 0

    # ------------------------------------------------------------------ acting
    def act(self, obs, greedy: bool = False, action_mask=None) -> int:
        """Sample an action, optionally restricted to ``action_mask``.

        ``action_mask`` is a boolean array of length ``n_actions``; ``False``
        entries get zero probability.  An all-``False`` mask (which would be a
        bug in the caller) is treated as "no mask" so the policy can never get
        stuck with an empty distribution.
        """
        logits = self.actor.forward(obs)
        mask = np.ones(self.n_actions, dtype=bool) if action_mask is None \
            else np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.size != self.n_actions or not mask.any():
            mask = np.ones(self.n_actions, dtype=bool)
        self._last_mask = mask
        masked = np.where(mask, logits, -1e9)
        lp = log_softmax(masked)
        if greedy:
            a = int(np.argmax(masked))
        else:
            p = np.exp(lp)
            a = int(self.rng.choice(self.n_actions, p=p))
        self._last_logp = float(lp[a])
        self._last_value = float(self.critic.forward(obs)[0])
        return a

    # ----------------------------------------------------------------- learning
    def clear_buffer(self):
        """Drop the partially collected rollout (used when the opponent changes)."""
        self.buf_ptr = 0

    def update(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        self.total_steps += 1
        i = self.buf_ptr
        self.buf_obs[i] = obs
        self.buf_act[i] = action
        self.buf_logp[i] = self._last_logp
        self.buf_val[i] = self._last_value
        self.buf_rew[i] = reward
        self.buf_done[i] = float(terminated or truncated)  # time-out counts as failure
        self.buf_mask[i] = self._last_mask
        self.buf_ptr += 1
        self.last_next_obs = np.asarray(next_obs, dtype=float)

        if self.buf_ptr < int(self.cfg["n_steps"]):
            return None
        out = self._update_from_buffer()
        self.buf_ptr = 0
        return out

    def _update_from_buffer(self):
        c = self.cfg
        T = self.buf_ptr
        obs = self.buf_obs[:T]
        act = self.buf_act[:T]
        logp_old = self.buf_logp[:T]
        val = self.buf_val[:T]
        rew = self.buf_rew[:T]
        done = self.buf_done[:T]
        mask = self.buf_mask[:T]

        # ---- GAE -----------------------------------------------------------
        last_v = 0.0 if done[-1] else float(self.critic.forward(self.last_next_obs)[0])
        adv = np.zeros(T)
        gae = 0.0
        for t in range(T - 1, -1, -1):
            next_v = last_v if t == T - 1 else val[t + 1]
            nonterminal = 1.0 - done[t]
            delta = rew[t] + c["gamma"] * nonterminal * next_v - val[t]
            gae = delta + c["gamma"] * c["lam"] * nonterminal * gae
            adv[t] = gae
        ret = adv + val

        if c["normalize_adv"]:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- epochs / minibatches -----------------------------------------
        n = T
        mb = int(c["minibatch"])
        stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        n_updates = 0
        for _ in range(int(c["epochs"])):
            order = self.rng.permutation(n)
            for start in range(0, n - mb + 1, mb):
                idx = order[start:start + mb]
                b_obs, b_act = obs[idx], act[idx]
                b_logp_old, b_adv, b_ret = logp_old[idx], adv[idx], ret[idx]
                b_mask = mask[idx]
                B = len(idx)

                logits = np.where(b_mask, self.actor.forward(b_obs), -1e9)
                lp = log_softmax(logits)
                p = np.exp(lp)
                logp = lp[np.arange(B), b_act]

                ratio = np.exp(logp - b_logp_old)
                surr1 = ratio * b_adv
                surr2 = np.clip(ratio, 1.0 - c["clip"], 1.0 + c["clip"]) * b_adv
                inside = surr1 <= surr2
                policy_loss = -np.minimum(surr1, surr2).mean()

                ent = -(p * lp).sum(axis=1)
                entropy = ent.mean()

                # d(-min(surr1,surr2))/d logp  (zero where the clipped branch wins)
                coef = np.where(inside, -b_adv * ratio, 0.0)
                onehot = np.zeros_like(p)
                onehot[np.arange(B), b_act] = 1.0
                dlogits = coef[:, None] * (onehot - p)                       # policy
                dlogits += c["ent_coef"] * p * (lp + ent[:, None])           # entropy
                dlogits = np.where(b_mask, dlogits, 0.0)
                dlogits /= B
                self.actor_opt.step(self.actor.backward(dlogits))

                v = self.critic.forward(b_obs).reshape(-1)
                value_loss = 0.5 * np.mean((v - b_ret) ** 2)
                self.critic_opt.step(self.critic.backward(((v - b_ret) / B)[:, None]))

                stats["loss"] += policy_loss + c["vf_coef"] * value_loss - c["ent_coef"] * entropy
                stats["policy_loss"] += policy_loss
                stats["value_loss"] += value_loss
                stats["entropy"] += entropy
                stats["approx_kl"] += float(np.mean(b_logp_old - logp))
                stats["clip_frac"] += float(np.mean(~inside))
                n_updates += 1

        self.update_count += 1
        out = {k: float(v / max(n_updates, 1)) for k, v in stats.items()}
        out["explained_var"] = float(
            1.0 - np.var(ret - val) / (np.var(ret) + 1e-8)
        )
        self.last_loss = out
        return out

    # ------------------------------------------------- behaviour cloning
    def pretrain(self, obs, actions, epochs=12, batch_size=256, label_smoothing=0.0,
                 action_mask=None):
        """Supervised warm-start of the actor on ``(obs, actions)`` pairs.

        This is plain cross-entropy on the same MLP and Adam implementation the
        PPO update uses -- no new network, no new optimiser.  The critic is left
        untouched, and a *fresh* Adam is used so PPO's own optimiser state (and
        therefore the first on-policy update) is unaffected.

        ``action_mask`` (``[n, n_actions]`` boolean, optional) is the safety mask
        that was in force when each demonstration was recorded.  When given, the
        logits are masked **exactly as** :meth:`_update_from_buffer` masks them
        (``-1e9`` on forbidden actions) before the log-softmax, so the
        supervised distribution, its gradient and the PPO ratio all live in the
        same space.  Without it the legacy unmasked update is unchanged.

        A demonstrated action that the mask forbids is re-admitted for that
        sample only (otherwise its cross-entropy would be ``+inf``); the count is
        returned as ``bc_label_forced`` so a caller can see the conflict instead
        of silently training through it.

        ``label_smoothing`` mixes the one-hot target with a uniform distribution
        (``y = (1-e) * onehot + e / n_actions``), which keeps the logits from
        saturating on a small demonstration set.  The supervised actions are
        never altered: every label comes from the collector.

        Returns a dict with the full-dataset loss / accuracy / probability of the
        demonstrated action, measured before the first update and after the last
        one, so a caller can verify that the supervision actually took effect.
        """
        obs = np.asarray(obs, dtype=float)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if obs.ndim != 2:
            raise ValueError(f"pretrain expects 2-D obs, got shape {obs.shape}")
        if obs.shape[1] != self.obs_dim:
            raise ValueError(
                f"pretrain obs width {obs.shape[1]} != policy input {self.obs_dim}; "
                f"the demonstrations were recorded for a different observation layout")
        if actions.shape[0] != obs.shape[0]:
            raise ValueError(
                f"pretrain got {obs.shape[0]} observations but {actions.shape[0]} actions")
        if obs.shape[0] == 0:
            raise ValueError("pretrain needs at least one demonstration sample")
        if actions.min() < 0 or actions.max() >= self.n_actions:
            raise ValueError(
                f"pretrain actions must lie in [0, {self.n_actions}), "
                f"got [{actions.min()}, {actions.max()}]")

        n = int(obs.shape[0])
        mask = np.ones((n, self.n_actions), dtype=bool)
        label_forced = 0
        masked_samples = 0
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool)
            if mask.shape != (n, self.n_actions):
                raise ValueError(
                    f"pretrain action_mask must be ({n}, {self.n_actions}), "
                    f"got {mask.shape}")
            masked_samples = int(np.count_nonzero(~mask.all(axis=1)))
            rows = np.arange(n)
            forbidden = ~mask[rows, actions]
            label_forced = int(np.count_nonzero(forbidden))
            if label_forced:
                mask[rows[forbidden], actions[forbidden]] = True
        rows = np.arange(n)

        target = np.zeros((n, self.n_actions))
        target[rows, actions] = 1.0
        eps = float(label_smoothing)
        if eps > 0.0:
            target = target * (1.0 - eps) + eps / float(self.n_actions)

        def masked_logp(batch_obs, batch_mask):
            logits = np.where(batch_mask, self.actor.forward(batch_obs), -1e9)
            return log_softmax(logits)

        def measure():
            lp = masked_logp(obs, mask)
            p = np.exp(lp)
            loss = float(-np.mean(np.sum(target * lp, axis=1)))
            prob = float(np.mean(p[rows, actions]))
            acc = float(np.mean(np.argmax(lp, axis=1) == actions))
            return loss, prob, acc

        loss_start, prob_start, acc_start = measure()

        opt = Adam(self.actor.params, lr=float(self.cfg.get("bc_lr", 1e-3)))
        mb = max(1, min(int(batch_size), n))
        epochs = max(1, int(epochs))
        for _ in range(epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, mb):
                idx = order[start:start + mb]
                b_mask = mask[idx]
                logits = np.where(b_mask, self.actor.forward(obs[idx]), -1e9)
                p = softmax(logits)
                dlogits = (p - target[idx]) / float(len(idx))
                # the mask also zeroes the gradient of forbidden logits, exactly
                # like the PPO update does over its rollout masks
                dlogits = np.where(b_mask, dlogits, 0.0)
                opt.step(self.actor.backward(dlogits))

        loss_end, prob_end, acc_end = measure()
        return {
            "bc_samples": n,
            "bc_epochs": epochs,
            "bc_batch_size": mb,
            "bc_feature_dim": int(obs.shape[1]),
            "bc_mask_used": action_mask is not None,
            "bc_masked_samples": masked_samples,
            "bc_label_forced": label_forced,
            "bc_loss_start": loss_start,
            "bc_loss_end": loss_end,
            "bc_action_prob_start": prob_start,
            "bc_action_prob_end": prob_end,
            "bc_accuracy_start": acc_start,
            "bc_accuracy_end": acc_end,
        }

    # ------------------------------------------------------------- persistence
    def state_dict(self):
        st = {"a_" + k: v for k, v in self.actor.get_state().items()}
        st.update({"c_" + k: v for k, v in self.critic.get_state().items()})
        st["meta"] = np.array([self.obs_dim, self.n_actions, self.total_steps, self.seed])
        return st

    def load_state_dict(self, state):
        self.actor.set_state({k[2:]: v for k, v in state.items() if k.startswith("a_")})
        self.critic.set_state({k[2:]: v for k, v in state.items() if k.startswith("c_")})
        self.total_steps = int(state["meta"][2])

    def stats(self):
        return {"updates": self.update_count, "buffer_fill": self.buf_ptr}
