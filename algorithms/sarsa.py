"""
SARSA(0) implementation.

On-policy TD control using ε-greedy action selection.
"""

import numpy as np
from collections import defaultdict


class SARSAAgent:
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=1.0):
        """
        Initialize the SARSA agent.

        Args:
            env: Gymnasium environment
            epsilon (float): Exploration probability
            alpha (float): Learning rate
            gamma (float): Discount factor
        """
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def select_action(self, state):
        """
        ε-greedy action selection.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return np.random.choice(best_actions)

    def train(self, num_episodes):
        """
        Train the agent using SARSA.

        Returns:
            success_log (list): Binary success per episode
            real_returns (list): Real (unshaped) return per episode
            episode_lengths (list): Number of steps per episode
        """
        success_log = []
        episode_lengths = []
        real_returns = []

        for _ in range(num_episodes):
            ep_success = 0
            step_count = 0
            ep_real_return = 0.0

            # Initialize S, A
            state, _ = self.env.reset()
            action = self.select_action(state)

            done = False

            while not done:
                step_count += 1

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Track success
                if info.get("is_success", False):
                    ep_success = 1

                # Track real return (original reward)
                ep_real_return += info.get("orig_reward", reward)

                # Choose A' from S' using policy derived from Q (ε-greedy)
                if not done:
                    next_action = self.select_action(next_state)
                    td_target = reward + self.gamma * self.q_table[next_state][next_action]
                else:
                    # Terminal state: no next action needed
                    next_action = None
                    td_target = reward

                # Update Q(S, A)
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error

                # S ← S'; A ← A'
                state = next_state
                action = next_action

            success_log.append(ep_success)
            episode_lengths.append(step_count)
            real_returns.append(ep_real_return)

        return success_log, real_returns, episode_lengths

    def get_policy(self):
        """
        Extract greedy policy from Q-table.

        Returns:
            dict: {state: best_action}
        """
        policy = {}
        for state in self.q_table.keys():
            q_values = self.q_table[state]
            policy[state] = np.argmax(q_values)
        return policy