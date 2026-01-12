import torch
import numpy as np
import os
from env.flappy_wrapper import FlappyWrapper
from agent.dqn_agent import DQNAgent
from agent.replay_buffer import ReplayBuffer

# --- Hyperparameters ---
MAX_EPISODES = 10000  # Total games to play
MAX_STEPS = 2000  # Max steps per game (to prevent infinite loops)
BATCH_SIZE = 64
BUFFER_SIZE = 50000  # Course suggests 50k, but 30k is fine for 16GB RAM
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 150000  # Frames over which epsilon decays
SYNC_TARGET_FRAMES = 1000  # How often to update target net
LEARNING_STARTS = 5000  # Fill buffer before training


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores_history = []
    print(f"Training on: {device}")

    # Initialize Environment
    env = FlappyWrapper(render=False, frame_stack=4, frame_skip=2)

    # Initialize Agent and Buffer
    agent = DQNAgent(
        state_shape=(4, 84, 84),
        action_size=2,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        device=device
    )
    buffer = ReplayBuffer(BUFFER_SIZE)

    # Variables for tracking
    epsilon = EPSILON_START
    frame_idx = 0
    best_score = -float('inf')

    # Ensure directories exist
    os.makedirs("checkpoints", exist_ok=True)

    print("Starting training loop...")

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        episode_reward = 0
        loss_val = 0

        for step in range(MAX_STEPS):
            frame_idx += 1

            # 1. Select Action (Epsilon Greedy)
            # Calculate current epsilon
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                      np.exp(-1. * frame_idx / EPSILON_DECAY)

            action = agent.act(state, epsilon)

            # 2. Step Environment
            next_state, reward, done, info = env.step(action)

            # 3. Store in Buffer
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # 4. Train Agent
            if len(buffer) > LEARNING_STARTS:
                loss = agent.learn(buffer)
                if loss is not None:
                    loss_val = loss

            # 5. Sync Target Network
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                agent.sync_target()
                print(f"[{frame_idx}] Target Network Synced")

            if done:
                break

        # Logging
        scores_history.append(episode_reward)
        pipes_passed = info.get('score', 0)
        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f} | Loss: {loss_val:.4f} | Pipes: {pipes_passed}")

        # Save Best Model
        if episode_reward > best_score and frame_idx > LEARNING_STARTS:
            best_score = episode_reward
            agent.save("checkpoints/best_model.pth")
            print(f"--> New Best Model Saved! Score: {best_score}")

    env.close()
    os.makedirs("results", exist_ok=True)
    np.save("results/training_scores.npy", np.array(scores_history))
    print("Scores saved to results/training_scores.npy")
    print("Training Complete.")


if __name__ == "__main__":
    main()