"""
Train and test Monte Carlo Control on FrozenLake.
"""

from env.frozenlake_env import create_frozenlake_env
from algorithms.monte_carlo import MonteCarloAgent
import numpy as np


def train_mc(num_episodes=5000):
    env = create_frozenlake_env(
        grid_size=6,
        hole_density=0.0,       # start easy
        is_slippery=False,
        seed=42
    )

    agent = MonteCarloAgent(
        env=env,
        epsilon=0.2,
        alpha=0.5,
        gamma=1
    )

    successes = []

    for episode_idx in range(num_episodes):
        episode = agent.generate_episode()
        agent.update_q(episode)

        # Track success (goal reached)
        episode_return = sum([r for (_, _, r) in episode])
        successes.append(episode_return)

        if (episode_idx + 1) % 500 == 0:
            success_rate = np.mean(successes[-500:])
            print(f"Episode {episode_idx + 1}, success rate (last 500): {success_rate:.2f}")

    env.close()
    return agent


if __name__ == "__main__":
    trained_agent = train_mc()
