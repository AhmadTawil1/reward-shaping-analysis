"""
Reward Shaping Wrappers for FrozenLake.

This module implements reward shaping using Gymnasium wrappers.
It allows modifying the reward signal while keeping the underlying
MDP and algorithms unchanged.

Implemented shaping methods:
1. Baseline (no shaping)         -> implicit (no wrapper)
2. Step-cost shaping             -> encourages shorter paths
3. Potential-based distance shaping -> policy-invariant shaping
4. Safety-based shaping          -> encourages safer paths away from holes
5. Exploration bonus shaping     -> encourages diverse exploration

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
    
    NOTE: Project requires γ = 1.0, so this is hardcoded.
    """

    def __init__(self, env: gym.Env, beta: float = 1.0):
        super().__init__(env)
        self.beta = beta
        # Project requires gamma = 1.0
        self.gamma = 1.0

        # Extract grid size from FrozenLake description (more robust)
        if hasattr(env, 'unwrapped'):
            desc = env.unwrapped.desc
        else:
            desc = env.desc
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

        # Use gamma = 1.0 as per project requirements
        shaping = self.gamma * phi_next - phi_s
        return self.beta * shaping


# -------------------------------------------------------------------
# 3. CUSTOM: Safety-Based Reward Shaping
# -------------------------------------------------------------------

class SafetyBasedShaping(RewardShapingWrapper):
    """
    Custom Shaping #1: Safety-Based Shaping
    
    Penalizes being near holes to encourage safer paths.
    Uses potential-based formulation for policy invariance.
    
    Mathematical form:
        Φ(s) = min_distance_to_hole(s)
        F(s,a,s') = γΦ(s') - Φ(s)
        R'(s,a,s') = R(s,a,s') + βF(s,a,s')
    
    Intuition: 
        - Agent receives positive shaping when moving away from holes
        - Agent receives negative shaping when moving toward holes
        - Encourages exploration of safer paths
        - Potential-based, so preserves optimal policy in theory
    """
    
    def __init__(self, env: gym.Env, beta: float = 0.1):
        super().__init__(env)
        self.beta = beta
        self.gamma = 1.0  # Project requirement
        
        # Extract grid size (more robust)
        if hasattr(env, 'unwrapped'):
            desc = env.unwrapped.desc
        else:
            desc = env.desc
        self.grid_size = len(desc)
        
        # Find all hole positions
        self.hole_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = desc[i][j]
                if cell == b"H" or cell == "H":
                    self.hole_positions.append((i, j))
        
        # OPTIMIZATION: Pre-compute all potentials for faster access
        self.potential_cache = {}
        for state in range(self.grid_size ** 2):
            r, c = self.state_to_pos(state)
            if not self.hole_positions:
                self.potential_cache[state] = 10.0
            else:
                distances = [abs(r - hr) + abs(c - hc) 
                           for hr, hc in self.hole_positions]
                self.potential_cache[state] = float(min(distances))
    
    def state_to_pos(self, state: int):
        """Convert state index to (row, col)."""
        row = state // self.grid_size
        col = state % self.grid_size
        return row, col
    
    def potential(self, state: int):
        """
        Φ(s) = min_distance_to_hole(s)
        
        Returns the Manhattan distance to the nearest hole.
        Pre-computed for efficiency.
        """
        return self.potential_cache[state]
    
    def compute_shaping_reward(self, state, action, next_state):
        phi_s = self.potential(state)
        phi_next = self.potential(next_state)
        
        # Potential-based shaping with gamma = 1.0
        shaping = self.gamma * phi_next - phi_s
        return self.beta * shaping


# -------------------------------------------------------------------
# 4. CUSTOM: Exploration Bonus Shaping (FIXED)
# -------------------------------------------------------------------

class ExplorationBonusShaping(RewardShapingWrapper):
    """
    Custom Shaping #2: Exploration Bonus (State Visitation Counting)
    
    Provides bonus rewards for visiting less-explored states.
    Uses potential-based formulation to preserve policy invariance.
    
    Mathematical form:
        Φ(s) = -sqrt(visit_count[s] + 1)
        F(s,a,s') = γΦ(s') - Φ(s)
        R'(s,a,s') = R(s,a,s') + βF(s,a,s')
    
    Intuition:
        - Novel states have higher potential (less negative)
        - Moving to a novel state gives positive shaping reward
        - Encourages diverse exploration across training
        - As states are visited, their "novelty bonus" decreases
        - sqrt() provides diminishing returns (1st visit matters most)
        - Potential-based, so preserves optimal policy
    
    IMPORTANT FIX:
        Visit counts are now maintained ACROSS episodes by default.
        This is crucial for Monte Carlo, which learns from complete episodes
        and benefits from knowing which states have been explored globally.
        
        Set reset_per_episode=True to reset counts each episode
        (for within-episode exploration only).
    
    Arguments:
        beta (float): Shaping strength (default: 0.05)
        reset_per_episode (bool): If True, reset counts each episode.
                                  If False (default), counts persist across episodes.
    """
    
    def __init__(self, env: gym.Env, beta: float = 0.05, 
                 reset_per_episode: bool = False):
        super().__init__(env)
        self.beta = beta
        self.gamma = 1.0  # Project requirement
        self.visit_counts = {}
        self.reset_per_episode = reset_per_episode
    
    def reset(self, **kwargs):
        """
        Reset environment. 
        
        By default, visit counts are PRESERVED across episodes
        to benefit episodic learners like Monte Carlo.
        """
        state, info = super().reset(**kwargs)
        
        # Only reset counts if explicitly requested
        if self.reset_per_episode:
            self.visit_counts = {}
        
        return state, info
    
    def potential(self, state: int):
        """
        Φ(s) = -sqrt(visit_count[s] + 1)
        
        Novel states have higher (less negative) potential.
        The +1 ensures we never take sqrt(0).
        """
        count = self.visit_counts.get(state, 0)
        return -np.sqrt(count + 1)
    
    def compute_shaping_reward(self, state, action, next_state):
        # Compute potentials BEFORE updating count
        phi_s = self.potential(state)
        phi_next = self.potential(next_state)
        
        # Update visit count for next state
        self.visit_counts[next_state] = self.visit_counts.get(next_state, 0) + 1
        
        # Potential-based shaping with gamma = 1.0
        shaping = self.gamma * phi_next - phi_s
        return self.beta * shaping