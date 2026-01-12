import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from model.cnn_model import FlappyCNN


class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=1e-4, gamma=0.99, buffer_size=50000, batch_size=32,
                 device="cpu"):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        # Policy Network (The one we train)
        self.policy_net = FlappyCNN(input_channels=state_shape[0], num_actions=action_size).to(self.device)

        # Target Network (The stable one used for calculating labels)
        self.target_net = FlappyCNN(input_channels=state_shape[0], num_actions=action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set to evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Huber Loss is often more stable for DQN than MSE
        self.criterion = nn.SmoothL1Loss()

    def act(self, state, epsilon=0.0):
        """
        Strategy Pattern: Chooses between Exploration (Random) and Exploitation (Model)
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                # Prepare state for the GPU/Model: (1, 4, 84, 84)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def learn(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return None

        # 1. Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)

        # 2. Convert to Tensors and move to GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # (B, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # (B, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # (B, 1)

        # 3. Compute Current Q values: Q(s, a)
        # We gather the Q-value corresponding to the action actually taken
        current_q = self.policy_net(states).gather(1, actions)

        # 4. Compute Target Q values: r + gamma * max(Q_target(s', a'))
        # We use the Target Network for this to ensure stability (Course 8, Slide 63)
        with torch.no_grad():
            # Double DQN:
            # 1. Policy net chooses the action
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # 2. Target net evaluates that action
            next_q_values = self.target_net(next_states).gather(1, next_actions)

            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # 5. Compute Loss and Optimize
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        """
        Copies weights from Policy Net to Target Net.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())