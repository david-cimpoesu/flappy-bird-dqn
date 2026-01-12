import collections
import numpy as np
import random


class ReplayBuffer:
    """
    Implements a circular buffer to store transitions.
    """

    def __init__(self, capacity):
        # Uses collections.deque which is efficient for appending and popping from ends
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition.
        """
        # We store the experience as a tuple
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions.
        """
        transitions = random.sample(self.buffer, batch_size)

        # Unzip the transitions into separate arrays
        state, action, reward, next_state, done = zip(*transitions)

        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)