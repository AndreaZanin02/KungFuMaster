import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Four core components of DQN (Mnih et al., 2015):
- Q-Network (Policy Network):
    Neural network that maps game frames to Q-values for every possible action.
- Target Network:
    A periodically-updated copy of the Q-Network, used to stabilize TD target computation.
- Replay Buffer (Experience Replay):
    Stores past transitions (s, a, r, s', done). Random mini-batches are drawn to break
    temporal correlation between consecutive frames.
- Epsilon-Greedy Strategy:
    Balances exploration (random actions with probability epsilon) and exploitation
    (greedy w.r.t. Q-values). Epsilon decays over time.
"""


class ReplayBuffer:
    """Fixed-size replay buffer storing transitions as uint8 numpy arrays."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Store a transition. States should be uint8 to save RAM."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample a random mini-batch to break temporal correlation."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)


class DQN_CNN(nn.Module):
    """Nature CNN (Mnih et al., 2015) — 3 conv layers + 2 FC layers."""

    def __init__(self, input_channels: int, num_actions: int) -> None:
        super().__init__()
        # Input: (Batch, input_channels, 84, 84)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # After convolutions: (Batch, 64, 7, 7) → flatten → 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize uint8 images [0, 255] -> [0.0, 1.0] on device
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    """DQN agent with target network and epsilon-greedy action selection."""

    def __init__(self, action_dim: int, device: torch.device, input_channels: int = 4,
                 lr: float = 1e-4, gamma: float = 0.99, batch_size: int = 32) -> None:
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        # Policy and target networks
        self.policy_net = DQN_CNN(input_channels=input_channels, num_actions=action_dim).to(device)
        self.target_net = DQN_CNN(input_channels=input_channels, num_actions=action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if random.random() > epsilon:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_t)
                return q_values.max(1)[1].item()
        return random.randrange(self.action_dim)

    def learn(self, memory: ReplayBuffer) -> Optional[float]:
        """Run one gradient step on a mini-batch from the replay buffer."""
        if len(memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)

        # Convert to tensors on device
        states = torch.tensor(states, dtype=torch.uint8, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.uint8, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q-values for the actions taken
        curr_q = self.policy_net(states).gather(1, actions)

        # TD target using the target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Huber loss (more stable than MSE for outliers)
        loss = F.smooth_l1_loss(curr_q, target_q)

        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Path) -> None:
        """Save agent state to disk."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: Path) -> None:
        """Load agent state from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])