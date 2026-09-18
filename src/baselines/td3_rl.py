"""TD3-RL baseline (main.tex §IV-A, fourth baseline).

Description from main.tex:
    "TD3-RL substitutes the dual-decomposition solver with a twin-delayed
     deep deterministic policy gradient (TD3) scheduler trained over 3,000
     episodes under configuration randomization."

Implementation details:
  * Per-user policy with global-summary input -- the action vector is sampled
    independently per user, conditioned on local request/channel features and
    aggregate satellite load. This is the standard simplification for
    multi-user resource allocation with TD3, since otherwise the joint action
    space grows linearly in U and centralized training scales poorly.
  * State (R^9): (n_txt_norm, n_vis_norm, n_out_norm, H, gamma_G2S_log,
                 gamma_S2G_log, sat0_load, mean_load, max_load).
  * Action (R^3): (eta, x_logit, s_offset) -> mapped to
                  eta in [eta_min, 1], x in {0, 1}, s in {0..S-1}.
  * Resources are equally split per satellite at deployment (as in Zhou 2025).
  * Reward: per-user QoE.
  * Training: 3000 episodes x 30 users/ep = 90k transitions. Replay buffer
    capacity 50k. Twin critics, target nets with tau=0.005, gamma=0.99,
    policy-update delay 2. CPU runtime ~5-15 minutes.

The class :class:`TD3Agent` is a stand-alone trainable agent. A module-level
:func:`solve_per_slot_td3` looks up a cached trained checkpoint
(``data/td3_checkpoint.pt``) and runs the policy on incoming slots. If the
checkpoint is missing, ``solve_per_slot_td3`` raises a helpful error so the
caller can run ``train_td3.py`` first.
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from src.data_gen import SlotData, Request, make_slot
from src.inner_solve import _ttft_tpot_sat, _ttft_tpot_loc
from src.semantic_profile import Q_eff
from src.optimizer import Decision, SlotResult


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                                "data", "td3_checkpoint.pt")


STATE_DIM = 9
ACTION_DIM = 3


# ---------------------------------------------------------------------------
#  Feature engineering
# ---------------------------------------------------------------------------
def _state_for(req: Request, sat_loads: np.ndarray) -> np.ndarray:
    """Normalize request + global summary into the TD3 state vector."""
    n_txt_n = (req.n_txt - C.N_TXT_RANGE[0]) / max(1, C.N_TXT_RANGE[1] - C.N_TXT_RANGE[0])
    n_vis_n = (req.n_vis - C.N_VIS_RANGE[0]) / max(1, C.N_VIS_RANGE[1] - C.N_VIS_RANGE[0])
    n_out_n = (req.n_out - C.N_OUT_RANGE[0]) / max(1, C.N_OUT_RANGE[1] - C.N_OUT_RANGE[0])
    gamma_G_log = math.log10(req.gamma_G2S + 1e-9) / 2.0   # ~0.5..1.5 dB-scale
    gamma_S_log = math.log10(req.gamma_S2G + 1e-9) / 2.0
    return np.array([
        n_txt_n, n_vis_n, n_out_n, req.H,
        gamma_G_log, gamma_S_log,
        float(sat_loads[req.s0]),
        float(np.mean(sat_loads)),
        float(np.max(sat_loads)),
    ], dtype=np.float32)


def _decode_action(action: np.ndarray, req: Request, S: int) -> Tuple[int, int, float]:
    """Map a continuous action vector (in [-1, 1]^3) to (x, s, eta)."""
    a = np.clip(action, -1.0, 1.0)
    eta = float(C.ETA_MIN + 0.5 * (a[0] + 1.0) * (1.0 - C.ETA_MIN))   # [eta_min, 1]
    x   = 1 if a[1] >= 0.0 else 0
    # s_offset in [-1, 1] mapped to an integer ring offset in [-S//2, S//2] from the access sat
    offset = int(round(a[2] * (S // 2)))
    s = (req.s0 + offset) % S
    return x, s, eta


# ---------------------------------------------------------------------------
#  Neural networks
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(s))   # action in [-1, 1]^3


class Critic(nn.Module):
    """Twin-Q critic (TD3 standard)."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 64):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


# ---------------------------------------------------------------------------
#  Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: np.ndarray, r: float,
             s2: np.ndarray, done: bool) -> None:
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int) -> tuple:
        idxs = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idxs]
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.float32),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
#  TD3 agent
# ---------------------------------------------------------------------------
@dataclass
class TD3Config:
    actor_lr: float       = 3e-4
    critic_lr: float      = 3e-4
    gamma: float          = 0.99
    tau: float            = 5e-3
    batch_size: int       = 128
    buffer_capacity: int  = 50_000
    policy_noise: float   = 0.20
    noise_clip: float     = 0.50
    policy_update_delay: int = 2
    exploration_noise: float = 0.10
    seed: int             = 0


class TD3Agent:
    def __init__(self, cfg: TD3Config = TD3Config()):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        self.actor = Actor()
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = ReplayBuffer(capacity=cfg.buffer_capacity)
        self.update_step = 0

    # ----------------- action selection -----------------
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s).squeeze(0).numpy()
        if not deterministic:
            noise = np.random.normal(0, self.cfg.exploration_noise, size=ACTION_DIM)
            a = np.clip(a + noise, -1.0, 1.0)
        return a

    # ----------------- TD3 update step -----------------
    def update(self) -> None:
        if len(self.buffer) < self.cfg.batch_size:
            return
        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
        s  = torch.from_numpy(s)
        a  = torch.from_numpy(a)
        r  = torch.from_numpy(r).unsqueeze(-1)
        s2 = torch.from_numpy(s2)
        d  = torch.from_numpy(d).unsqueeze(-1)

        # Target action with smoothed noise (TD3-specific)
        with torch.no_grad():
            noise = torch.randn_like(a) * self.cfg.policy_noise
            noise = torch.clamp(noise, -self.cfg.noise_clip, self.cfg.noise_clip)
            a2 = torch.clamp(self.actor_target(s2) + noise, -1.0, 1.0)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_t = torch.min(q1_t, q2_t)
            y = r + self.cfg.gamma * (1.0 - d) * q_t

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.update_step += 1
        if self.update_step % self.cfg.policy_update_delay == 0:
            actor_loss = -self.critic.q1(torch.cat([s, self.actor(s)], dim=-1)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Polyak target update
            with torch.no_grad():
                for p, p_t in zip(self.actor.parameters(),  self.actor_target.parameters()):
                    p_t.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * p_t.data)
                for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                    p_t.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * p_t.data)

    # ----------------- save / load -----------------
    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# ---------------------------------------------------------------------------
#  Per-slot evaluation under a fixed policy
# ---------------------------------------------------------------------------
def _evaluate_slot(slot: SlotData, agent: TD3Agent,
                    w_q: float, w_1: float, w_2: float,
                    deterministic: bool = True,
                    B_up_budget: float = C.B_UP_MAX,
                    B_dn_budget: float = C.B_DN_MAX,
                    F_budget: float = C.F_SAT_MAX) -> List[Decision]:
    """Apply the policy to every user; equal-split resources per sat."""
    S = slot.topology.S
    requests = slot.requests

    n_per_sat = np.zeros(S)
    for r in requests:
        n_per_sat[r.s0] += 1
    U_norm = max(1.0, len(requests) / S)
    sat_loads = n_per_sat / U_norm

    raw = []
    for req in requests:
        state = _state_for(req, sat_loads)
        action = agent.act(state, deterministic=deterministic)
        x, s, eta = _decode_action(action, req, S)
        raw.append((req, state, action, x, s, eta))

    # Count assignments for equal-split resources
    n_sat = np.zeros(S, dtype=int)
    n_acc = np.zeros(S, dtype=int)
    for (req, _, _, x, s, _) in raw:
        n_acc[req.s0] += 1
        if x == 1:
            n_sat[s] += 1

    decisions = []
    for (req, state, action, x, s, eta) in raw:
        if x == 0:
            TTFT, TPOT = _ttft_tpot_loc(req, eta)
            Qe = Q_eff(eta, x=0, delta=req.delta, theta=req.theta)
            QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
            decisions.append(Decision(
                u=req.u, x=0, s=req.s0, eta=eta,
                B_up=0.0, B_dn=0.0, f=C.F_LOC,
                QoE=QoE, TTFT=TTFT, TPOT=TPOT, Q_eff=Qe,
            ))
        else:
            U_acc = max(1, n_acc[req.s0])
            U_sat = max(1, n_sat[s])
            Bup = B_up_budget / U_acc
            Bdn = B_dn_budget / U_acc
            fs  = F_budget   / U_sat
            TTFT, TPOT = _ttft_tpot_sat(req, slot.topology, s, eta, Bup, Bdn, fs)
            Qe = Q_eff(eta, x=1, delta=req.delta, theta=req.theta)
            QoE = w_q * math.log(1.0 + Qe) - w_1 * TTFT - w_2 * TPOT
            decisions.append(Decision(
                u=req.u, x=1, s=s, eta=eta,
                B_up=Bup, B_dn=Bdn, f=fs,
                QoE=QoE, TTFT=TTFT, TPOT=TPOT, Q_eff=Qe,
            ))
    return decisions


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------
def train(n_episodes: int = 3000, save_path: str = CHECKPOINT_PATH,
          U_range: tuple = (10, 40), S: int = 6,
          verbose: bool = True) -> TD3Agent:
    """Train per-user TD3 policy over configuration-randomized episodes."""
    cfg = TD3Config()
    agent = TD3Agent(cfg)

    rng = np.random.default_rng(2026)
    t0 = time.perf_counter()
    qoe_history: List[float] = []

    for ep in range(n_episodes):
        U = int(rng.integers(U_range[0], U_range[1] + 1))
        slot = make_slot(U=U, S=S, rng=rng)

        n_per_sat = np.zeros(S)
        for r in slot.requests:
            n_per_sat[r.s0] += 1
        sat_loads = n_per_sat / max(1.0, len(slot.requests) / S)

        states  = [_state_for(req, sat_loads) for req in slot.requests]
        actions = [agent.act(s) for s in states]

        # Step the environment forward by one "slot" (single-step MDP per user)
        n_sat = np.zeros(S, dtype=int)
        n_acc = np.zeros(S, dtype=int)
        decoded = []
        for req, act in zip(slot.requests, actions):
            x, s_u, eta = _decode_action(act, req, S)
            n_acc[req.s0] += 1
            if x == 1:
                n_sat[s_u] += 1
            decoded.append((x, s_u, eta))

        rewards = []
        for req, (x, s_u, eta) in zip(slot.requests, decoded):
            if x == 0:
                TTFT, TPOT = _ttft_tpot_loc(req, eta)
                Qe = Q_eff(eta, x=0, delta=req.delta, theta=req.theta)
            else:
                U_acc = max(1, n_acc[req.s0])
                U_sat = max(1, n_sat[s_u])
                Bup = C.B_UP_MAX / U_acc
                Bdn = C.B_DN_MAX / U_acc
                fs  = C.F_SAT_MAX / U_sat
                TTFT, TPOT = _ttft_tpot_sat(req, slot.topology, s_u, eta, Bup, Bdn, fs)
                Qe = Q_eff(eta, x=1, delta=req.delta, theta=req.theta)
            r = C.W_Q * math.log(1.0 + Qe) - C.W_1 * TTFT - C.W_2 * TPOT
            rewards.append(r)

        # Treat each user as a one-step MDP: terminal next-state = current (done = True)
        for state, action, reward in zip(states, actions, rewards):
            agent.buffer.push(state, action, reward, state, True)

        # Multiple update steps per episode
        for _ in range(len(slot.requests)):
            agent.update()

        ep_avg_qoe = float(np.mean(rewards))
        qoe_history.append(ep_avg_qoe)

        if verbose and (ep + 1) % 200 == 0:
            avg100 = float(np.mean(qoe_history[-200:]))
            print(f"  ep={ep+1:5d}/{n_episodes}  U={U:2d}  "
                  f"avg(reward, last200)={avg100:+7.3f}  "
                  f"({time.perf_counter() - t0:6.1f}s elapsed)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    # training log (per-episode mean reward) for the convergence criterion / response letter
    np.savez(os.path.join(os.path.dirname(save_path), "td3_train_log.npz"),
             ep_reward=np.array(qoe_history), n_episodes=n_episodes, seed=cfg.seed)
    ma = np.convolve(qoe_history, np.ones(200) / 200, mode="valid")
    if len(ma) > 500:
        rel = abs(ma[-1] - ma[-500]) / max(abs(ma[-500]), 1e-9)
        print(f"  200-episode moving-average reward: {ma[-500]:+.3f} -> {ma[-1]:+.3f} over the last 500 episodes (rel. change {rel*100:.1f}%)")
    if verbose:
        print(f"  saved -> {save_path}")
    return agent


# ---------------------------------------------------------------------------
#  Public scheme entry point
# ---------------------------------------------------------------------------
_cached_agent: TD3Agent | None = None


def _load_agent() -> TD3Agent:
    global _cached_agent
    if _cached_agent is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise RuntimeError(
                f"TD3 checkpoint not found at {CHECKPOINT_PATH}. "
                f"Run `python -m src.baselines.td3_rl --train` first."
            )
        agent = TD3Agent()
        agent.load(CHECKPOINT_PATH)
        _cached_agent = agent
    return _cached_agent


def solve_per_slot_td3(slot: SlotData,
                        w_q: float = C.W_Q, w_1: float = C.W_1, w_2: float = C.W_2,
                        B_up_budget: float = C.B_UP_MAX,
                        B_dn_budget: float = C.B_DN_MAX,
                        F_budget: float = C.F_SAT_MAX,
                        **_) -> SlotResult:
    t0 = time.perf_counter()
    agent = _load_agent()
    decisions = _evaluate_slot(slot, agent, w_q, w_1, w_2,
                                deterministic=True,
                                B_up_budget=B_up_budget,
                                B_dn_budget=B_dn_budget,
                                F_budget=F_budget)
    S = slot.topology.S
    return SlotResult(
        decisions=decisions,
        iters=1,
        primal_dual_gap=0.0,
        sum_QoE=float(sum(d.QoE for d in decisions)),
        wall_time_s=time.perf_counter() - t0,
        duals={"mu_up": np.zeros(S), "mu_dn": np.zeros(S), "nu": np.zeros(S)},
        gap_history=[],
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=3000)
    args = parser.parse_args()
    if args.train:
        train(n_episodes=args.n_episodes)
    else:
        print("Use --train to start TD3 training.")


__all__ = ["TD3Agent", "TD3Config", "train", "solve_per_slot_td3", "CHECKPOINT_PATH"]
