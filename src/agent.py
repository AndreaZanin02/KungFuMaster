import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

"""
Four core components:
- Q-Network (Policy Network): 
    The primary neural network that takes game frames as input and outputs the Q-Values 
    for every possible action
- Target Network: An exact copy of the Q-Network that is updated less frequently
    (e.g., every 1000 steps). It serves to stabilize training by providing a fixed "target" 
    for the error calculation
- Replay Buffer (Experience Replay): A memory buffer that stores past transitions
    (State, Action, Reward, Next State). The agent draws random mini-batches from this buffer
    to train, breaking the temporal correlation between consecutive frames
- Epsilon-Greedy Strategy: The logic that decides whether to explore random moves
    (with probability eps) or exploit the network's current knowledge. The value of eps decays over time
"""

# THE REPLAY BUFFER --> (CurrentState, Acrion, Reward, NextState)
class ReplayBuffer:
    def __init__(self, capacity):
        # We use a deque that automatically removes the oldest elements when full
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state and next_state are uint8 numpy arrays (to save RAM)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample a random mini-batch to break temporal correlation
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# THE NEURAL NETWORK (NATURE CNN)
class DQN_CNN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN_CNN, self).__init__()
        # Input: (Batch, 4, 84, 84)  --> ALE standards
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # After convolutions, flatten and pass to fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # Normalize images [0, 255] -> [0.0, 1.0] directly on the GPU
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) # Flatten (Batch, 3136)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# THE DQN AGENT
class DQNAgent:
    def __init__(self, action_dim, device, lr=1e-4, gamma=0.99, batch_size=32):
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        # Initialize Policy Network and Target Network
        self.policy_net = DQN_CNN(input_channels=4, num_actions=action_dim).to(self.device)
        self.target_net = DQN_CNN(input_channels=4, num_actions=action_dim).to(self.device)
        
        # The target network starts with the same weights as the policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # The target network does not perform backpropagation

        # Adam optimizer (otherwise we can also try RMSprop)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, epsilon):
        # Epsilon-Greedy: exploration vs exploitation
        if random.random() > epsilon:
            with torch.no_grad():
                # Convert state to tensor, add batch dimension, and move to GPU
                state_tensor = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item() # Return the index of the action with the highest Q-value
        else:
            return random.randrange(self.action_dim) # Random action

    def learn(self, memory):
        # Do not learn if the buffer does not have enough elements
        if len(memory) < self.batch_size:
            return None

        # 1. Sample from the Replay Buffer
        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)

        # 2. Convert to Tensors and move to GPU (heavy lifting happens here)
        states = torch.tensor(states, dtype=torch.uint8, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.uint8, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3. Calculate current Q-values predicted by the Policy Network
        # gather() extracts the specific Q-value for the action the agent actually took
        curr_q = self.policy_net(states).gather(1, actions)

        # 4. Calculate future Q-values (max) using the Target Network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # If the next state is terminal (done=1), the target is just the reward
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # 5. Calculate Loss (Huber Loss is more stable than MSE for outliers)
        loss = F.smooth_l1_loss(curr_q, target_q)

        # 6. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        # Copy weights from Policy to Target network
        self.target_net.load_state_dict(self.policy_net.state_dict())