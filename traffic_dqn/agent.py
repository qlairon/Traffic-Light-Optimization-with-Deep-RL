"""Q-network, replay buffer and DQN agent."""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─────────────────────────────────────────────
#  2. Q-NETWORK
# ─────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Fully-connected DQN: 8 → 64 → 64 → 4

    Architecture choices:
        • Two hidden layers (64 units each) — enough capacity for this task.
        • ReLU activations throughout.
        • No activation on the output layer (raw Q-values).
    """

    def __init__(self, state_size: int, action_size: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  3. REPLAY BUFFER
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Uniform experience replay.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
#  4. DQN AGENT
# ─────────────────────────────────────────────

class DQNAgent:
    """
    DQN with:
        • ε-greedy exploration (multiplicative decay per learning step)
        • Double-DQN targets (online net picks a', target net evaluates it)
        • Target network (hard update every C steps)
        • Experience replay
        • Huber loss (smooth L1) for stability
    """

    def __init__(
        self,
        state_size:     int   = 8,
        action_size:    int   = 4,
        lr:             float = 1e-3,
        gamma:          float = 0.95,
        epsilon:        float = 1.0,
        epsilon_min:    float = 0.05,
        epsilon_decay:  float = 0.990,
        buffer_size:    int   = 10_000,
        batch_size:     int   = 64,
        target_update:  int   = 100,   # steps between target-net hard updates
    ):
        self.action_size    = action_size
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.batch_size     = batch_size
        self.target_update  = target_update
        self.step_count     = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network (trained every step)
        self.q_net = QNetwork(state_size, action_size).to(self.device)
        # Target network (frozen, updated periodically)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

    # ── Action selection ──────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax().item())

    # ── Learning ──────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """
        Sample a mini-batch, compute TD targets, update online network.
        Returns the loss value (or None if buffer too small).
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current Q(s, a)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double-DQN target: r + γ · Q_target(s', argmax_a' Q_online(s', a'))  (0 if terminal)
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(1, keepdim=True)
            max_next_q   = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            td_targets = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_values, td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)  # gradient clipping
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodically sync target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str = "models/dqn_traffic.pth"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net":   self.q_net.state_dict(),
            "epsilon": self.epsilon,
        }, path)
        print(f"Model saved → {path}")

    def load(self, path: str = "models/dqn_traffic.pth"):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.epsilon = ckpt["epsilon"]
        print(f"Model loaded ← {path}")
