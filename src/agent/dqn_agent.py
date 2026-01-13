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

        self.policy_net = FlappyCNN(input_channels=state_shape[0], num_actions=action_size).to(self.device)

        self.target_net = FlappyCNN(input_channels=state_shape[0], num_actions=action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.criterion = nn.SmoothL1Loss()

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def learn(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_state_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)

            next_q_values = self.target_net(next_states).gather(1, next_state_actions)

            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())