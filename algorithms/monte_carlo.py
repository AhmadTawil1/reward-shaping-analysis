"""
Every-Visit Monte Carlo Control implementation.

This module implements:
- ε-greedy action selection
- Episode generation
- Every-Visit Monte Carlo Q-value updates
"""

import numpy as np
from collections import defaultdict


class MonteCarloAgent:
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=1.0):
        """
        Initialize the Monte Carlo agent.

        Args:
            env: Gymnasium environment
            epsilon (float): Exploration probability
            alpha (float): Learning rate
            gamma (float): Discount factor (must be 1.0 for this project)
        """
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        # Q-table: Q[state][action]
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def select_action(self, state):
        """
        Select an action using ε-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        q_values = self.q_table[state]
        max_q = np.max(q_values)

        # Random tie-breaking
        best_actions = np.flatnonzero(q_values == max_q)
        return np.random.choice(best_actions)

    def generate_episode(self):
        """
        Generate a single episode using the current policy.

        Returns:
            episode (list): List of (state, action, reward)
        """
        episode = []
        state, _ = self.env.reset()

        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            episode.append((state, action, reward))

            state = next_state
            done = terminated or truncated

        return episode

    def update_q(self, episode):
        """
        Update Q-values using Every-Visit Monte Carlo.
        """
        G = 0.0

        # Compute returns backwards
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G

            # Incremental MC update
            self.q_table[state][action] += self.alpha * (G - self.q_table[state][action])
