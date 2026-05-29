import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zlib

# ---------------------- Original replay buffer --------------------------
"""
# Fixed-size replay buffer storing transitions as uint8 numpy arrays
class ReplayBuffer:

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    # Store a transition. States should be uint8 to save RAM
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    # Sample a random mini-batch to break temporal correlation
    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)
"""

# -------------------- Optimized replay buffer with compression ----------------------
"""
class ReplayBuffer:
    def __init__(self, capacity: int, frame_stack: int = 4) -> None:
        self.capacity = capacity
        self.frame_stack = frame_stack

        # max_frames is incremented because every reset gives us 4 extra frames
        self.max_frames = int(capacity * 1.1) + frame_stack
        self.frames = [None] * self.max_frames
        
        # frame_ptr is the absolute index, the wrap is managed by the module operation
        self.frame_ptr = 0          

        # Transaction structure: (absolute_state_last_idx, action, reward, done)
        self.transitions = deque()

        self._compress = lambda f: zlib.compress(f.tobytes(), level=1)
        self._decompress = lambda b, shape: np.frombuffer(
            zlib.decompress(b), dtype=np.uint8
        ).reshape(shape)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        
        # Saving the starting frames 
        if len(self.transitions) == 0 or (len(self.transitions) > 0 and self.transitions[-1][3]):
            for i in range(self.frame_stack):
                self._write_frame(state[i])

        # Save only the new frame (not all the 4 new frames) and the index
        new_frame_idx = self._write_frame(next_state[-1])
        state_last_idx = new_frame_idx - 1

        self.transitions.append((state_last_idx, action, reward, done))

        # Circular memory management logic
        while self.transitions:
            oldest_state_last_idx = self.transitions[0][0]
            
            # The oldest frame of the transaction
            oldest_required_frame = oldest_state_last_idx - (self.frame_stack - 1)
            
            # If the current index and the oldest frame required from the transaction are farer than
            # the max dimension of the buffer, il frame has been overwritten
            is_overwritten = (self.frame_ptr - oldest_required_frame) >= self.max_frames
            
            if is_overwritten or len(self.transitions) > self.capacity:
                self.transitions.popleft()
            else:
                break

    def _write_frame(self, frame: np.ndarray) -> int:
        idx = self.frame_ptr
        self.frames[idx % self.max_frames] = self._compress(frame)
        self.frame_ptr += 1
        return idx

    def _get_state(self, last_frame_idx: int) -> np.ndarray:
        frames = []
        for i in range(self.frame_stack - 1, -1, -1):
            idx = (last_frame_idx - i) % self.max_frames
            frames.append(self._decompress(self.frames[idx], (84, 84)))
        return np.stack(frames, axis=0)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        batch = random.sample(self.transitions, batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for frame_idx, action, reward, done in batch:
            states.append(self._get_state(frame_idx))
            next_states.append(self._get_state(frame_idx + 1))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.transitions)
"""

# --------------------- Pre-allocated circular buffer no compression ----------------------
class ReplayBuffer:
    def __init__(self, capacity: int, frame_stack: int = 4) -> None:
        self.capacity = capacity
        self.frame_stack = frame_stack

        # max_frames is incremented because every reset gives us 4 extra frames
        self.max_frames = int(capacity * 1.1) + frame_stack
        
        # Pre-allocation of the memory
        self.frames = np.empty((self.max_frames, 84, 84), dtype=np.uint8)
        
        # frame_ptr is the absolute index, the wrap is managed by the module operation
        self.frame_ptr = 0          

        # Transaction structure: (absolute_state_last_idx, action, reward, done)
        self.transitions = deque()

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        
        # Saving the starting frames 
        if len(self.transitions) == 0 or (len(self.transitions) > 0 and self.transitions[-1][3]):
            for i in range(self.frame_stack):
                self._write_frame(state[i])

        # Save only the new frame (not all the 4 new frames) and the index
        new_frame_idx = self._write_frame(next_state[-1])
        state_last_idx = new_frame_idx - 1

        self.transitions.append((state_last_idx, action, reward, done))

        # Circular memory management logic
        while self.transitions:
            oldest_state_last_idx = self.transitions[0][0]
            
            # The oldest frame of the transaction
            oldest_required_frame = oldest_state_last_idx - (self.frame_stack - 1)
            
            # If the current index and the oldest frame required from the transaction are farer than
            # the max dimension of the buffer, il frame has been overwritten
            is_overwritten = (self.frame_ptr - oldest_required_frame) >= self.max_frames
            
            if is_overwritten or len(self.transitions) > self.capacity:
                self.transitions.popleft()
            else:
                break

    def _write_frame(self, frame: np.ndarray) -> int:
        idx = self.frame_ptr
        self.frames[idx % self.max_frames] = frame
        self.frame_ptr += 1
        return idx

    def _get_state(self, last_frame_idx: int) -> np.ndarray:
        # Advanced numpy indexes
        indices = [(last_frame_idx - i) % self.max_frames for i in range(self.frame_stack - 1, -1, -1)]
        return self.frames[indices]

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        batch = random.sample(self.transitions, batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for frame_idx, action, reward, done in batch:
            states.append(self._get_state(frame_idx))
            next_states.append(self._get_state(frame_idx + 1))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.transitions)

class DQN_CNN(nn.Module):

    def __init__(self, input_channels: int, num_actions: int) -> None:
        super().__init__()
        # Input: (Batch, input_channels, 84, 84)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # After convolutions: (Batch, 64, 7, 7) -> flatten -> 3136
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

# DQN agent with target network and epsilon-greedy action selection
class DQNAgent:
    

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

    # Epsilon-greedy action selection
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() > epsilon:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_t)
                return q_values.max(1)[1].item()
        return random.randrange(self.action_dim)

    # Run one gradient step on a mini-batch from the replay buffer
    # type hint aggiornato: restituisce una tupla (loss, q_mean)
    def learn(self, memory: ReplayBuffer) -> Optional[tuple[float, float]]:
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

        # Log informations
        return loss.item(), curr_q.mean().item()

    # Copy weights from policy network to target network
    def update_target_network(self) -> None:
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # Save agent state to disk
    def save(self, path: Path) -> None:
        
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    # Load agent state from disk
    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])