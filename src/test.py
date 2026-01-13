import torch
import time
from env.flappy_wrapper import FlappyWrapper
from agent.dqn_agent import DQNAgent


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    env = FlappyWrapper(render=True, frame_stack=4)

    agent = DQNAgent(state_shape=(4, 84, 84), action_size=2, device=device)

    try:
        agent.load("checkpoints/best_model.pth")
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: checkpoints/best_model.pth not found. Train the model first.")
        return

    for episode in range(5):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state, epsilon=0.0)
            state, reward, done, info = env.step(action)
            total_reward += reward

        print(f"Test Episode {episode + 1} Score: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()