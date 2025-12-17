"""
FrozenLake Environment Generator

This module provides a configurable FrozenLake environment generator
that supports:
- Arbitrary grid sizes (>= 2x2)
- Controlled hole density (10%–20%)
- Exactly one start and one goal
- Slippery or non-slippery dynamics
- Reproducibility via random seed

Used by:
- Monte Carlo Control
- SARSA
- Reward shaping wrappers (later)
"""

import numpy as np
import gymnasium as gym
import time


def generate_frozenlake_map(
    grid_size: int,
    hole_density: float,
    seed: int = 42
) -> list[str]:
    """
    Generate a random FrozenLake map.

    Args:
        grid_size (int): Size of the grid (grid_size x grid_size).
        hole_density (float): Fraction of holes (between 0.1 and 0.2).
        seed (int): Random seed for reproducibility.

    Returns:
        List[str]: FrozenLake map layout.
    """
    assert grid_size >= 2, "Grid size must be at least 2"
    assert 0.0 <= hole_density <= 0.2, "Hole density must be between 0.1 and 0.2"

    rng = np.random.default_rng(seed)

    num_cells = grid_size * grid_size
    num_holes = int(num_cells * hole_density)

    # Initialize all cells as frozen
    grid = np.full(num_cells, "F")

    # Define fixed start and goal positions
    start_idx = 0
    goal_idx = num_cells - 1

    grid[start_idx] = "S"
    grid[goal_idx] = "G"

    # Available positions for holes (exclude start & goal)
    available_indices = [
        i for i in range(num_cells)
        if i not in (start_idx, goal_idx)
    ]

    hole_indices = rng.choice(
        available_indices,
        size=num_holes,
        replace=False
    )

    for idx in hole_indices:
        grid[idx] = "H"

    # Convert to FrozenLake string format
    grid_map = [
        "".join(grid[i * grid_size:(i + 1) * grid_size])
        for i in range(grid_size)
    ]

    return grid_map


def create_frozenlake_env(
    grid_size: int,
    hole_density: float,
    is_slippery: bool = True,
    seed: int = 42,
    render_mode: str | None = None
) -> gym.Env:
    """
    Create a Gymnasium FrozenLake environment.

    Args:
        grid_size (int): Size of the grid.
        hole_density (float): Fraction of holes.
        is_slippery (bool): Whether transitions are stochastic.
        seed (int): Random seed.
        render_mode (str | None): Optional render mode ("human", "ansi").

    Returns:
        gym.Env: Configured FrozenLake environment.
    """
    desc = generate_frozenlake_map(
        grid_size=grid_size,
        hole_density=hole_density,
        seed=seed
    )

    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=is_slippery,
        render_mode=render_mode
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env



if __name__ == "__main__":
    ## Test the environment
    env = create_frozenlake_env(
        grid_size=6,
        hole_density=0.17,
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
