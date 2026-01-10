import numpy as np
import torch

from env.flappy_wrapper import FlappyWrapper
from model.cnn_model import FlappyCNN


def main():
    print("Creating environment...")
    env = FlappyWrapper(render=True, frame_stack=4)

    print("Resetting environment...")
    state = env.reset()

    print("State shape:", state.shape)  # (4, 84, 84)

    # Convertim la tensor PyTorch
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    print("State tensor shape:", state_tensor.shape)  # (1, 4, 84, 84)

    print("Creating model...")
    model = FlappyCNN(input_channels=4, num_actions=2)

    print("Running forward pass...")
    with torch.no_grad():
        q_values = model(state_tensor)

    print("Q values shape:", q_values.shape)  # (1, 2)
    print("Q values:", q_values)

    print("Running random agent for 1000 steps...")
    for i in range(1000):
        action = np.random.randint(0, 2)
        next_state, reward, done, info = env.step(action)
        if done:
            print("Game over, resetting...")
            env.reset()

    print("Next state shape:", next_state.shape)
    print("Reward:", reward, "Done:", done)

    env.close()
    print("TEST FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
