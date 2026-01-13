import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import cv2
from collections import deque


class FlappyWrapper:
    def __init__(self, render=True, frame_stack=4, frame_skip=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.frame_skip = frame_skip

        self.env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else "rgb_array"
        )

    def reset(self):
        obs, info = self.env.reset()

        frame = self._preprocess(obs)

        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)

        return self._get_state()

    def step(self, action):
        total_reward = 0.0
        done = False

        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)

            shaped_reward = reward
            if terminated:
                shaped_reward = -1.0
            elif reward >= 1.0:
                shaped_reward = 10.0
            else:
                shaped_reward = 0.1

            total_reward += shaped_reward
            done = terminated or truncated
            if done:
                break

        frame = self._preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0), total_reward, done, info

    def close(self):
        self.env.close()

    def _get_state(self):
        # Stack pe canal: (C, H, W)
        return np.stack(self.frames, axis=0)

    def _preprocess(self, frame):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        frame = frame.astype(np.float32) / 255.0

        return frame