"""
Reward Shaping Wrappers for FrozenLake.

This module implements reward shaping using Gymnasium wrappers.
It allows modifying the reward signal while keeping the underlying
MDP and algorithms unchanged.

Implemented shaping methods:
1. Baseline (no shaping)         -> implicit (no wrapper)
2. Step-cost shaping             -> encourages shorter paths
3. Potential-based distance shaping -> policy-invariant shaping

These wrappers are designed to work with:
- Monte Carlo Control
- SARSA(0)

"""

import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):
    """
    Base class for reward shaping wrappers.

    This wrapper intercepts the environment's step() method,
    applies a shaping term F(s, a, s'), and returns a modified reward:
        R'(s,a,s') = R(s,a,s') + F(s,a,s')

    Subclasses must implement:
        compute_shaping_reward(...)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_state = None

    def reset(self, **kwargs):
        """
        Reset the environment and track the initial state.
        """
        state, info = self.env.reset(**kwargs)
        self.last_state = state
        return state, info

    def step(self, action):
        # Step in the original environment
        next_state, orig_reward, terminated, truncated, info = self.env.step(action)

        # Compute shaping term F(s,a,s')
        shaping_reward = self.compute_shaping_reward(
            self.last_state, action, next_state
        )

        # Shaped reward used for LEARNING
        shaped_reward = orig_reward + shaping_reward

        # Make sure info exists and is mutable
        info = dict(info)

        # --- Preserve original reward & metrics ---
        info["orig_reward"] = orig_reward              # real FrozenLake reward
        info["shaping_reward"] = shaping_reward        # F(s,a,s')
        info["shaped_reward"] = shaped_reward           # R'
        info["is_success"] = bool(terminated and orig_reward == 1)

        self.last_state = next_state

        return next_state, shaped_reward, terminated, truncated, info

    def compute_shaping_reward(self, state, action, next_state):
        """
        Compute the shaping reward F(s, a, s').

        Default: no shaping.
        """
        return 0.0


# -------------------------------------------------------------------
# 1. Step-Cost Reward Shaping
# -------------------------------------------------------------------

class StepCostShaping(RewardShapingWrapper):
    """
    Step-cost reward shaping.

    Adds a small negative reward at every step to encourage
    shorter and more efficient paths.

    Mathematical form:
        R'(s,a,s') = R(s,a,s') + c
    where c < 0
    """

    def __init__(self, env: gym.Env, step_cost: float = -0.01):
        super().__init__(env)
        self.step_cost = step_cost

    def compute_shaping_reward(self, state, action, next_state):
        return self.step_cost


# -------------------------------------------------------------------
# 2. Potential-Based Distance Reward Shaping
# -------------------------------------------------------------------

class PotentialBasedDistanceShaping(RewardShapingWrapper):
    """
    Potential-based reward shaping using distance to the goal.

    Potential function:
        Φ(s) = -d(s)
    where d(s) is the Manhattan distance to the goal.

    Shaping term:
        F(s,a,s') = γΦ(s') - Φ(s)

    Final reward:
        R'(s,a,s') = R(s,a,s') + βF(s,a,s')

    This shaping is policy-invariant in theory.
    """

    def __init__(self, env: gym.Env, beta: float = 1.0, gamma: float = 1.0):
        super().__init__(env)
        self.beta = beta
        self.gamma = gamma

        # Extract grid size from FrozenLake description
        desc = env.unwrapped.desc
        self.grid_size = len(desc)

        # Locate goal position
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = desc[i][j]
                if cell == b"G" or cell == "G":
                    self.goal_pos = (i, j)

    def state_to_pos(self, state: int):
        """
        Convert discrete state index to (row, col).
        """
        row = state // self.grid_size
        col = state % self.grid_size
        return row, col

    def potential(self, state: int):
        """
        Compute Φ(s) = -ManhattanDistance(s, goal)
        """
        r, c = self.state_to_pos(state)
        gr, gc = self.goal_pos
        return -abs(r - gr) - abs(c - gc)

    def compute_shaping_reward(self, state, action, next_state):
        phi_s = self.potential(state)
        phi_next = self.potential(next_state)

        shaping = self.gamma * phi_next - phi_s
        return self.beta * shaping





