"""
FrozenLake Environment Generator

This module provides a configurable FrozenLake environment generator
that supports:
- Arbitrary grid sizes (>= 6x6)
- Controlled hole density (10%–20%)
- Exactly one start and one goal
- Slippery or non-slippery dynamics
- Reproducibility via random seed

Used by:
- Monte Carlo Control
- SARSA
- Reward shaping wrappers
"""

import numpy as np
import gymnasium as gym
import time
from collections import deque
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv


class CustomSlipperinessFrozenLake(gym.Wrapper):
    """
    Wrapper to modify FrozenLake's transition probabilities.
    
    By default, FrozenLake with is_slippery=True has:
    - 1/3 probability for intended direction
    - 1/3 probability for perpendicular left
    - 1/3 probability for perpendicular right
    
    This wrapper allows you to set a custom success_rate (e.g., 0.7)
    for the intended action, with the remaining probability split
    equally between the two perpendicular directions.
    
    Args:
        env: The base FrozenLake environment
        success_rate: Probability of moving in intended direction (0.0 to 1.0)
    """
    def __init__(self, env, success_rate=0.7):
        super().__init__(env)
        self.success_rate = success_rate
        self.fail_rate = (1.0 - success_rate) / 2.0
        
        # Modify the transition probability dictionary
        self._modify_transitions()
    
    def _modify_transitions(self):
        """Modify the P dictionary to use custom transition probabilities."""
        # Access the underlying FrozenLake environment
        base_env = self.env.unwrapped
        
        # Get the original P dictionary
        original_P = base_env.P
        
        # Create new transition probabilities
        new_P = {}
        
        for state in original_P:
            new_P[state] = {}
            for action in original_P[state]:
                transitions = original_P[state][action]
                
                # If there are 3 transitions (slippery case), modify probabilities
                if len(transitions) == 3 and abs(transitions[0][0] - 1/3) < 0.01:
                    # This is a slippery transition with 1/3 probabilities
                    # Modify to use custom success_rate
                    new_transitions = []
                    for i, (prob, next_state, reward, done) in enumerate(transitions):
                        # First transition gets success_rate, others get fail_rate
                        if i == 0:
                            new_prob = self.success_rate
                        else:
                            new_prob = self.fail_rate
                        new_transitions.append((new_prob, next_state, reward, done))
                    new_P[state][action] = new_transitions
                else:
                    # Keep non-slippery or already modified transitions unchanged
                    new_P[state][action] = transitions
        
        # Replace the P dictionary
        base_env.P = new_P


def has_valid_path(grid_map: list[str]) -> bool:
    """
    Check whether a valid path exists from Start (S) to Goal (G)
    while avoiding holes (H), using Breadth-First Search (BFS).

    Args:
        grid_map (list[str]): FrozenLake map representation.

    Returns:
        bool: True if a path exists, False otherwise.
    """
    rows = len(grid_map)
    cols = len(grid_map[0]) if rows > 0 else 0

    # Locate start and goal positions
    start = None
    goal = None

    for i in range(rows):
        for j in range(cols):
            if grid_map[i][j] == "S":
                start = (i, j)
            elif grid_map[i][j] == "G":
                goal = (i, j)

    if start is None or goal is None:
        return False

    queue = deque([start])
    visited = {start}

    # Possible movements: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()

        if (x, y) == goal:
            return True

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if (nx, ny) not in visited and grid_map[nx][ny] != "H":
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return False


def get_adjacent_positions(idx: int, rows: int, cols: int) -> list[int]:
    """
    Get all adjacent positions (including diagonals) for a given cell index.
    
    Args:
        idx (int): Flat index of the cell in the grid.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    
    Returns:
        list[int]: List of adjacent cell indices.
    """
    row = idx // cols
    col = idx % cols
    
    adjacent = []
    # Check all 8 directions: up, down, left, right, and 4 diagonals
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # Skip the cell itself
            
            new_row = row + dr
            new_col = col + dc
            
            # Check if the new position is within bounds
            if 0 <= new_row < rows and 0 <= new_col < cols:
                adjacent.append(new_row * cols + new_col)
    
    return adjacent


def generate_frozenlake_map(
    rows: int,
    cols: int,
    hole_density: float,
    seed: int = 42
) -> list[str]:
    """
    Generate a random FrozenLake map with guaranteed reachability
    from Start (S) to Goal (G).

    The map is regenerated until at least one valid path exists.
    Holes are not placed adjacent to the start or goal positions.

    Args:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        hole_density (float): Fraction of holes (between 0.1 and 0.2).
        seed (int): Random seed for reproducibility.

    Returns:
        list[str]: Valid FrozenLake map layout.
    """
    assert rows >= 2 and cols >= 2, "Grid must be at least 2x2"
    assert 0.1 <= hole_density <= 0.2, "Hole density must be between 0.1 and 0.2"

    rng = np.random.default_rng(seed)

    num_cells = rows * cols
    num_holes = round(num_cells * hole_density)

    while True:
        # Initialize all cells as frozen
        grid = np.full(num_cells, "F")

        # Fixed start (top-left) and goal (bottom-right)
        start_idx = 0
        goal_idx = num_cells - 1
        grid[start_idx] = "S"
        grid[goal_idx] = "G"

        # Get positions adjacent to start and goal
        start_adjacent = get_adjacent_positions(start_idx, rows, cols)
        goal_adjacent = get_adjacent_positions(goal_idx, rows, cols)
        
        # Combine all forbidden positions (start, goal, and their adjacent cells)
        forbidden_indices = set([start_idx, goal_idx] + start_adjacent + goal_adjacent)

        # Candidate positions for holes (exclude start, goal, and adjacent cells)
        available_indices = [
            i for i in range(num_cells)
            if i not in forbidden_indices
        ]

        # Ensure we have enough available positions for the requested number of holes
        if len(available_indices) < num_holes:
            # If we can't place the requested number of holes, place as many as possible
            num_holes_to_place = len(available_indices)
        else:
            num_holes_to_place = num_holes

        hole_indices = rng.choice(
            available_indices,
            size=num_holes_to_place,
            replace=False
        )

        for idx in hole_indices:
            grid[idx] = "H"

        # Convert flat grid to list of strings
        grid_map = [
            "".join(grid[i * cols:(i + 1) * cols])
            for i in range(rows)
        ]

        # Accept map only if a valid path exists
        if has_valid_path(grid_map):
            return grid_map



def create_frozenlake_env(
    rows: int,
    cols: int,
    hole_density: float,
    is_slippery: bool = True,
    success_rate: float = 0.7,
    seed: int = 42,
    render_mode: str | None = None
) -> gym.Env:
    """
    Create a Gymnasium FrozenLake environment.

    Args:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        hole_density (float): Fraction of holes.
        is_slippery (bool): Whether transitions are stochastic.
        success_rate (float): Probability of successful action when slippery (default: 0.7).
                             Only used when is_slippery=True.
                             Default FrozenLake uses 0.33 (1/3).
        seed (int): Random seed.
        render_mode (str | None): Optional render mode ("human", "ansi").

    Returns:
        gym.Env: Configured FrozenLake environment.
    """
    desc = generate_frozenlake_map(
        rows=rows,
        cols=cols,
        hole_density=hole_density,
        seed=seed
    )

    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=is_slippery,
        max_episode_steps=500,
        render_mode=render_mode
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Apply custom slipperiness wrapper if needed
    if is_slippery and success_rate != 1.0/3.0:
        env = CustomSlipperinessFrozenLake(env, success_rate=success_rate)

    return env



if __name__ == "__main__":
    ## Test the environment
    env = create_frozenlake_env(
        rows=6,
        cols=6,
        hole_density=0.15,
        is_slippery=False,
        render_mode="human"
    )

    state, _ = env.reset()

    done = False

    while not done:
        action = env.action_space.sample()  # random action
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(0.5)  # slow down so you can see it

    env.close()