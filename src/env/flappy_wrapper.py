import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import cv2
from collections import deque


class FlappyWrapper:
    """
    Wrapper peste FlappyBird Gymnasium care:
    - returnează imagini
    - face resize la 84x84
    - grayscale
    - normalize
    - face stack de ultimele N frame-uri
    """

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

        # Frame Skipping Logic
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)

            # --- REWARD SHAPING ---
            # Standard rewards are +0.1 alive, +1.0 pipe, -1.0 death
            # We add a small bonus for staying alive to encourage horizontal progress
            shaped_reward = reward
            if terminated:
                # Heavy penalty to discourage "giving up" early
                shaped_reward = -5.0
            elif reward >= 1.0:
                # Massively increase the pipe reward so the CNN values the gap
                shaped_reward = 10.0
            else:
                # Small survival bonus to keep the bird in the air
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
        # frame poate fi:
        # - (H, W, 3) RGB
        # - (H, W) grayscale

        # Dacă are 3 canale, convertim la grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize la 84x84
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize [0,255] -> [0,1]
        frame = frame.astype(np.float32) / 255.0

        return frame
