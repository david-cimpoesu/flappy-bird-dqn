import torch
import time
from env.flappy_wrapper import FlappyWrapper
from agent.dqn_agent import DQNAgent


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    # Create Environment with Render ON
    env = FlappyWrapper(render=True, frame_stack=4)

    # Initialize Agent (State shape and action size must match training)
    agent = DQNAgent(state_shape=(4, 84, 84), action_size=2, device=device)

    # Load the best model
    try:
        agent.load("checkpoints/best_model.pth")
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: checkpoints/best_model.pth not found. Train the model first.")
        return

    # Run for a few episodes
    for episode in range(5):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Act with epsilon=0 (Pure Exploitation)
            action = agent.act(state, epsilon=0.0)
            state, reward, done, info = env.step(action)
            total_reward += reward

            # Optional: Slow down slightly to watch
            # time.sleep(0.03)

        print(f"Test Episode {episode + 1} Score: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()