"""
Every-Visit Monte Carlo Control implementation.

This module implements:
- ε-greedy action selection
- Episode generation
- Every-Visit Monte Carlo Q-value updates
- Optional optimistic initialization for better exploration

UPDATED: Added optimistic initialization support
"""

import numpy as np
from collections import defaultdict


class MonteCarloAgent:
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=1.0,
                 epsilon_decay=1.0, epsilon_min=0.0,
                 initial_q_value=0.0):
        """
        Initialize the Monte Carlo agent.

        Args:
            env: Gymnasium environment
            epsilon (float): Exploration probability
            alpha (float): Learning rate
            gamma (float): Discount factor (must be 1.0 for this project)
            epsilon_decay (float): Multiplicative decay factor per episode (default: 1.0 = no decay)
            epsilon_min (float): Minimum epsilon value (default: 0.0)
            initial_q_value (float): Initial Q-value for all state-action pairs.
                                     Use 0.0 for neutral, >0 for optimistic initialization.
                                     Optimistic values (e.g., 0.5, 1.0) encourage exploration.
        """
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # Epsilon decay parameters
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Optimistic initialization
        self.initial_q_value = initial_q_value

        # Q-table: Q[state][action]
        # Use optimistic initialization if specified
        if initial_q_value == 0.0:
            self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        else:
            self.q_table = defaultdict(
                lambda: np.full(self.env.action_space.n, self.initial_q_value)
            )

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
            episode (list): List of (state, action, shaped_reward, orig_reward, is_success)
        """
        episode = []
        state, _ = self.env.reset()

        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Extract original reward from info (for metrics)
            orig_reward = info.get("orig_reward", reward)
            is_success = info.get("is_success", False)

            # Store: (state, action, shaped_reward, orig_reward, is_success)
            episode.append((state, action, reward, orig_reward, is_success))

            state = next_state
            done = terminated or truncated

        return episode

    def update_q(self, episode):
        """
        Update Q-values using Every-Visit Monte Carlo.

        Uses SHAPED reward for learning (constant-α incremental update).
        """
        G = 0.0

        # Walk episode backwards
        for t in reversed(range(len(episode))):
            state, action, shaped_reward, _, _ = episode[t]

            # Compute return
            G = shaped_reward + self.gamma * G

            # Incremental Every-Visit MC update
            self.q_table[state][action] += self.alpha * (
                G - self.q_table[state][action]
            )

    def train(self, num_episodes):
        """
        Train the agent for a given number of episodes.

        Returns:
            success_log (list): Binary success per episode
            real_returns (list): Real (unshaped) return per episode
            episode_lengths (list): Number of steps per episode
        """
        success_log = []
        real_returns = []
        episode_lengths = []

        for ep in range(num_episodes):
            # Generate episode
            episode = self.generate_episode()

            # Update Q-values
            self.update_q(episode)

            # Track metrics (using ORIGINAL rewards)
            ep_length = len(episode)
            ep_real_return = sum(step[3] for step in episode)  # orig_reward
            ep_success = int(any(step[4] for step in episode))  # is_success

            success_log.append(ep_success)
            real_returns.append(ep_real_return)
            episode_lengths.append(ep_length)
            
            # Apply epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
    
    def get_value_function(self):
        """
        Extract state value function V(s) from Q-table.
        
        Returns:
            dict: {state: max_q_value}
        """
        value_func = {}
        for state in self.q_table.keys():
            value_func[state] = np.max(self.q_table[state])
        return value_func