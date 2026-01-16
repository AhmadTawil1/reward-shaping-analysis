"""
Unit tests for FrozenLake environment generation.

Tests:
- Map generation with valid parameters
- Protected zones (no holes adjacent to start/goal)
- Path reachability (BFS validation)
- Reproducibility with seeds
- Custom slipperiness wrapper
"""

import unittest
import numpy as np
from env.frozenlake_env import (
    generate_frozenlake_map,
    has_valid_path,
    get_adjacent_positions,
    create_frozenlake_env,
    CustomSlipperinessFrozenLake
)


class TestFrozenLakeEnvironment(unittest.TestCase):
    """Test suite for FrozenLake environment generation."""
    
    def test_map_generation_basic(self):
        """Test that map generation produces valid maps."""
        grid_size = 6
        hole_density = 0.15
        seed = 42
        
        map_layout = generate_frozenlake_map(grid_size, hole_density, seed)
        
        # Check map size
        self.assertEqual(len(map_layout), grid_size)
        for row in map_layout:
            self.assertEqual(len(row), grid_size)
        
        # Check start and goal exist
        flat_map = ''.join(map_layout)
        self.assertIn('S', flat_map)
        self.assertIn('G', flat_map)
        self.assertEqual(flat_map.count('S'), 1)
        self.assertEqual(flat_map.count('G'), 1)
    
    def test_protected_zones_start(self):
        """Test that no holes are placed adjacent to start position."""
        grid_size = 6
        hole_density = 0.15
        
        # Test multiple seeds
        for seed in [42, 123, 456, 789]:
            map_layout = generate_frozenlake_map(grid_size, hole_density, seed)
            
            # Start is at (0, 0)
            # Check all adjacent positions (including diagonals)
            adjacent_positions = [(0, 1), (1, 0), (1, 1)]
            
            for row, col in adjacent_positions:
                cell = map_layout[row][col]
                self.assertNotEqual(cell, 'H', 
                    f"Hole found adjacent to start at ({row}, {col}) with seed {seed}")
    
    def test_protected_zones_goal(self):
        """Test that no holes are placed adjacent to goal position."""
        grid_size = 6
        hole_density = 0.15
        
        # Test multiple seeds
        for seed in [42, 123, 456, 789]:
            map_layout = generate_frozenlake_map(grid_size, hole_density, seed)
            
            # Goal is at (grid_size-1, grid_size-1)
            goal_row, goal_col = grid_size - 1, grid_size - 1
            
            # Check all adjacent positions
            adjacent_positions = [
                (goal_row - 1, goal_col),
                (goal_row, goal_col - 1),
                (goal_row - 1, goal_col - 1)
            ]
            
            for row, col in adjacent_positions:
                cell = map_layout[row][col]
                self.assertNotEqual(cell, 'H',
                    f"Hole found adjacent to goal at ({row}, {col}) with seed {seed}")
    
    def test_get_adjacent_positions(self):
        """Test the get_adjacent_positions helper function."""
        grid_size = 6
        
        # Test corner position (0, 0) - start
        adjacent = get_adjacent_positions(0, grid_size)
        expected = [1, 6, 7]  # (0,1), (1,0), (1,1) in flat indices
        self.assertEqual(sorted(adjacent), sorted(expected))
        
        # Test corner position (5, 5) - goal for 6x6 grid
        adjacent = get_adjacent_positions(35, grid_size)  # 5*6 + 5 = 35
        expected = [28, 29, 34]  # (4,4), (4,5), (5,4) in flat indices
        self.assertEqual(sorted(adjacent), sorted(expected))
        
        # Test middle position (2, 2)
        adjacent = get_adjacent_positions(14, grid_size)  # 2*6 + 2 = 14
        # Should have 8 neighbors
        self.assertEqual(len(adjacent), 8)
    
    def test_path_reachability(self):
        """Test that generated maps always have a valid path."""
        grid_size = 6
        hole_density = 0.15
        
        # Test multiple seeds
        for seed in range(10):
            map_layout = generate_frozenlake_map(grid_size, hole_density, seed)
            self.assertTrue(has_valid_path(map_layout),
                f"No valid path found for seed {seed}")
    
    def test_reproducibility(self):
        """Test that same seed produces same map."""
        grid_size = 6
        hole_density = 0.15
        seed = 42
        
        map1 = generate_frozenlake_map(grid_size, hole_density, seed)
        map2 = generate_frozenlake_map(grid_size, hole_density, seed)
        
        self.assertEqual(map1, map2)
    
    def test_different_seeds_produce_different_maps(self):
        """Test that different seeds produce different maps."""
        grid_size = 6
        hole_density = 0.15
        
        map1 = generate_frozenlake_map(grid_size, hole_density, 42)
        map2 = generate_frozenlake_map(grid_size, hole_density, 123)
        
        # Maps should be different (very high probability)
        self.assertNotEqual(map1, map2)
    
    def test_hole_density_approximate(self):
        """Test that hole density is approximately as specified."""
        grid_size = 6
        hole_density = 0.15
        seed = 42
        
        map_layout = generate_frozenlake_map(grid_size, hole_density, seed)
        flat_map = ''.join(map_layout)
        
        num_holes = flat_map.count('H')
        total_cells = grid_size * grid_size
        
        # Account for protected zones reducing available positions
        # Density should be close but may be slightly lower due to constraints
        actual_density = num_holes / total_cells
        
        # Allow some tolerance (within 5% of target)
        self.assertLessEqual(actual_density, hole_density + 0.05)
    
    def test_environment_creation(self):
        """Test that environment can be created successfully."""
        env = create_frozenlake_env(
            grid_size=6,
            hole_density=0.15,
            is_slippery=False,
            seed=42
        )
        
        # Test reset
        state, info = env.reset()
        self.assertIsInstance(state, (int, np.integer))
        
        # Test step
        action = 0  # Move left
        next_state, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(next_state, (int, np.integer))
        
        env.close()
    
    def test_custom_slipperiness(self):
        """Test custom slipperiness wrapper."""
        env = create_frozenlake_env(
            grid_size=6,
            hole_density=0.15,
            is_slippery=True,
            success_rate=0.7,
            seed=42
        )
        
        # Check that wrapper is applied
        self.assertIsInstance(env, CustomSlipperinessFrozenLake)
        
        # Check success rate is set correctly
        self.assertAlmostEqual(env.success_rate, 0.7)
        self.assertAlmostEqual(env.fail_rate, 0.15)
        
        env.close()
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Grid size too small
        with self.assertRaises(AssertionError):
            generate_frozenlake_map(grid_size=1, hole_density=0.15, seed=42)
        
        # Hole density too low
        with self.assertRaises(AssertionError):
            generate_frozenlake_map(grid_size=6, hole_density=0.05, seed=42)
        
        # Hole density too high
        with self.assertRaises(AssertionError):
            generate_frozenlake_map(grid_size=6, hole_density=0.25, seed=42)


class TestBFSPathValidation(unittest.TestCase):
    """Test suite for BFS path validation."""
    
    def test_simple_valid_path(self):
        """Test BFS with a simple valid path."""
        # Simple 3x3 map with clear path
        map_layout = [
            "SFF",
            "FHF",
            "FFG"
        ]
        self.assertTrue(has_valid_path(map_layout))
    
    def test_no_path(self):
        """Test BFS with no valid path."""
        # Map with holes blocking all paths
        map_layout = [
            "SHF",
            "HHF",
            "FFG"
        ]
        self.assertFalse(has_valid_path(map_layout))
    
    def test_diagonal_not_allowed(self):
        """Test that BFS doesn't use diagonal movements."""
        # Path only exists diagonally
        map_layout = [
            "SHH",
            "HFH",
            "HHG"
        ]
        self.assertFalse(has_valid_path(map_layout))


if __name__ == '__main__':
    unittest.main()
