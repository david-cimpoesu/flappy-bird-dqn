import matplotlib.pyplot as plt
import numpy as np
import os


def plot_learning_curve(scores_file="results/training_scores.npy", output_file="results/learning_curve.png"):
    if not os.path.exists(scores_file):
        print(f"File {scores_file} not found. Run training first.")
        return

    scores = np.load(scores_file)

    # Calculate Moving Average (to smooth the jittery graph)
    window_size = 100
    moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Raw Score', alpha=0.3, color='gray')
    plt.plot(range(window_size - 1, len(scores)), moving_avg, label=f'{window_size}-Episode Moving Avg', color='blue')

    plt.title('Flappy Bird DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score (Reward)')
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    plot_learning_curve()