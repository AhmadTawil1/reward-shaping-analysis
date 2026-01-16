"""
Test script to visualize FrozenLake environments with 20 different seeds.

This script demonstrates that holes are not placed adjacent to the start
or goal positions across multiple random map generations.
"""

import time
import numpy as np
from env.frozenlake_env import create_frozenlake_env


def print_map_ascii(env):
    """
    Print the map in ASCII format for better visualization.
    
    Args:
        env: The FrozenLake environment
    """
    # Get the map description from the environment
    desc = env.unwrapped.desc
    grid_size = len(desc)
    
    print("\n" + "┌" + "─" * (grid_size * 2 + 1) + "┐")
    for row in desc:
        row_str = "│ "
        for cell in row:
            cell_char = cell.decode('utf-8') if isinstance(cell, bytes) else cell
            row_str += cell_char + " "
        row_str += "│"
        print(row_str)
    print("└" + "─" * (grid_size * 2 + 1) + "┘")


def verify_protected_zones(env, seed):
    """
    Verify that no holes are adjacent to start or goal.
    
    Args:
        env: The FrozenLake environment
        seed: The random seed used
    
    Returns:
        bool: True if protected zones are respected, False otherwise
    """
    desc = env.unwrapped.desc
    grid_size = len(desc)
    
    # Start is at (0, 0), Goal is at (grid_size-1, grid_size-1)
    start_pos = (0, 0)
    goal_pos = (grid_size - 1, grid_size - 1)
    
    # Check all adjacent positions (including diagonals)
    violations = []
    
    for pos, name in [(start_pos, "Start"), (goal_pos, "Goal")]:
        row, col = pos
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                    cell = desc[new_row][new_col]
                    cell_char = cell.decode('utf-8') if isinstance(cell, bytes) else cell
                    if cell_char == 'H':
                        violations.append(f"Hole found adjacent to {name} at ({new_row}, {new_col})")
    
    if violations:
        print(f"  ❌ VIOLATION (Seed {seed}):")
        for v in violations:
            print(f"     {v}")
        return False
    else:
        print(f"  ✓ Protected zones verified (Seed {seed})")
        return True


def visualize_multiple_seeds(num_seeds=20, grid_size=6, hole_density=0.15):
    """
    Create and visualize FrozenLake environments with multiple seeds.
    
    Args:
        num_seeds (int): Number of different seeds to test
        grid_size (int): Size of the grid
        hole_density (float): Density of holes (0.1 to 0.2)
    """
    print(f"\n{'='*70}")
    print(f"FrozenLake Environment Visualization - Multiple Seeds Test")
    print(f"{'='*70}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Hole Density: {hole_density}")
    print(f"Number of Seeds: {num_seeds}")
    print(f"{'='*70}\n")
    
    print("Legend:")
    print("  S = Start (top-left, position 0,0)")
    print("  G = Goal (bottom-right)")
    print("  F = Frozen (safe to walk)")
    print("  H = Hole (game over if you fall in)")
    print("\n✓ = No holes adjacent to S or G (including diagonals)")
    print("❌ = Violation detected (should not happen!)\n")
    print(f"{'='*70}\n")
    
    # Generate seeds
    seeds = [42 + i * 100 for i in range(num_seeds)]
    
    all_passed = True
    
    for i, seed in enumerate(seeds, 1):
        print(f"Map {i}/{num_seeds} (Seed: {seed})")
        print("-" * 70)
        
        # Create environment without rendering mode for faster processing
        env = create_frozenlake_env(
            rows=grid_size,
            cols=grid_size,
            hole_density=hole_density,
            is_slippery=False,
            seed=seed,
            render_mode=None
        )
        
        # Reset the environment
        env.reset()
        
        # Print the map
        print_map_ascii(env)
        
        # Verify protected zones
        passed = verify_protected_zones(env, seed)
        if not passed:
            all_passed = False
        
        env.close()
        print()
    
    print(f"{'='*70}")
    if all_passed:
        print(f"✓ ALL {num_seeds} MAPS PASSED - No holes adjacent to Start or Goal!")
    else:
        print(f"❌ SOME MAPS FAILED - Violations detected!")
    print(f"{'='*70}\n")


def visualize_single_with_gui(seed=42, grid_size=6, hole_density=0.15, duration=5):
    """
    Visualize a single environment with GUI rendering.
    
    Args:
        seed (int): Random seed
        grid_size (int): Size of the grid
        hole_density (float): Density of holes
        duration (int): How long to display (seconds)
    """
    print(f"\n{'='*70}")
    print(f"Opening GUI Visualization (Seed: {seed})")
    print(f"{'='*70}\n")
    
    env = create_frozenlake_env(
        rows=grid_size,
        cols=grid_size,
        hole_density=hole_density,
        is_slippery=False,
        seed=seed,
        render_mode="human"
    )
    
    env.reset()
    env.render()
    
    print(f"GUI window opened. Displaying for {duration} seconds...")
    print("(Close the window manually or wait for auto-close)\n")
    
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    
    env.close()
    print("GUI closed.\n")


if __name__ == "__main__":
    num_seeds = 20
    grid_size = 6
    hole_density = 0.15
    duration_per_map = 9  # seconds per map (20 maps × 9 seconds = 180 seconds = 3 minutes)
    
    # Generate seeds
    seeds = [42 + i * 100 for i in range(num_seeds)]
    
    print(f"\n{'='*70}")
    print(f"FrozenLake Environment GUI Visualization - {num_seeds} Maps")
    print(f"{'='*70}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Hole Density: {hole_density}")
    print(f"Duration per map: {duration_per_map} seconds")
    print(f"Total duration: {num_seeds * duration_per_map} seconds ({num_seeds * duration_per_map / 60:.1f} minutes)")
    print(f"{'='*70}\n")
    
    print("Legend:")
    print("  S = Start (top-left, position 0,0)")
    print("  G = Goal (bottom-right)")
    print("  F = Frozen (safe to walk)")
    print("  H = Hole (game over if you fall in)")
    print("\nNote: No holes should appear adjacent to S or G (including diagonals)\n")
    print(f"{'='*70}\n")
    
    # Visualize each map with GUI
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'─'*70}")
        print(f"Map {i}/{num_seeds} (Seed: {seed})")
        print(f"{'─'*70}")
        
        # Create environment with GUI rendering
        env = create_frozenlake_env(
            rows=grid_size,
            cols=grid_size,
            hole_density=hole_density,
            is_slippery=False,
            seed=seed,
            render_mode="human"
        )
        
        # Reset and render
        env.reset()
        
        # Print ASCII version in console
        print_map_ascii(env)
        
        # Verify protected zones
        verify_protected_zones(env, seed)
        
        # Render GUI
        env.render()
        
        print(f"\nDisplaying map {i}/{num_seeds} for {duration_per_map} seconds...")
        print(f"Time remaining: {(num_seeds - i) * duration_per_map + duration_per_map} seconds")
        
        try:
            time.sleep(duration_per_map)
        except KeyboardInterrupt:
            print("\n\nVisualization interrupted by user.")
            env.close()
            break
        
        env.close()
    
    print(f"\n{'='*70}")
    print(f"Visualization complete! All {num_seeds} maps displayed.")
    print(f"{'='*70}\n")
