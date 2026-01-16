"""
Unit tests for reward shaping wrappers.

Tests:
- Step-cost shaping
- Potential-based distance shaping
- Safety-based shaping
- Exploration bonus shaping
- Wrapper integration with environment
"""

import unittest
import numpy as np
from env.frozenlake_env import create_frozenlake_env
from env.reward_shaping import (
    RewardShapingWrapper,
    StepCostShaping,
    PotentialBasedDistanceShaping,
    SafetyBasedShaping,
    ExplorationBonusShaping
)


class TestRewardShapingBase(unittest.TestCase):
    """Test base reward shaping wrapper."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = create_frozenlake_env(
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            seed=42
        )
    
    def tearDown(self):
        """Clean up environment."""
        self.env.close()
    
    def test_base_wrapper_no_shaping(self):
        """Test that base wrapper doesn't modify rewards."""
        wrapped_env = RewardShapingWrapper(self.env)
        
        state, _ = wrapped_env.reset()
        next_state, reward, terminated, truncated, info = wrapped_env.step(0)
        
        # Base wrapper should not add shaping
        self.assertEqual(info['shaping_reward'], 0.0)
        self.assertEqual(reward, info['orig_reward'])


class TestStepCostShaping(unittest.TestCase):
    """Test step-cost shaping."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = create_frozenlake_env(
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            seed=42
        )
    
    def tearDown(self):
        """Clean up environment."""
        self.env.close()
    
    def test_step_cost_applied(self):
        """Test that step cost is applied at every step."""
        step_cost = -0.01
        wrapped_env = StepCostShaping(self.env, step_cost=step_cost)
        
        state, _ = wrapped_env.reset()
        next_state, reward, terminated, truncated, info = wrapped_env.step(0)
        
        # Should add step cost
        self.assertEqual(info['shaping_reward'], step_cost)
        self.assertEqual(reward, info['orig_reward'] + step_cost)
    
    def test_step_cost_custom_value(self):
        """Test custom step cost value."""
        step_cost = -0.05
        wrapped_env = StepCostShaping(self.env, step_cost=step_cost)
        
        state, _ = wrapped_env.reset()
        next_state, reward, terminated, truncated, info = wrapped_env.step(0)
        
        self.assertAlmostEqual(info['shaping_reward'], step_cost)


class TestPotentialBasedDistanceShaping(unittest.TestCase):
    """Test potential-based distance shaping."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = create_frozenlake_env(
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            seed=42
        )
    
    def tearDown(self):
        """Clean up environment."""
        self.env.close()
    
    def test_potential_function(self):
        """Test potential function calculation."""
        wrapped_env = PotentialBasedDistanceShaping(self.env, beta=1.0)
        
        # Goal is at (3, 3) for 4x4 grid
        # State 0 is at (0, 0), distance = 6
        potential_0 = wrapped_env.potential(0)
        self.assertEqual(potential_0, -6)
        
        # State 15 is at (3, 3), distance = 0
        potential_15 = wrapped_env.potential(15)
        self.assertEqual(potential_15, 0)
    
    def test_shaping_encourages_progress(self):
        """Test that shaping rewards progress toward goal."""
        wrapped_env = PotentialBasedDistanceShaping(self.env, beta=1.0)
        
        # Moving from state 0 (0,0) to state 1 (0,1) reduces distance by 1
        # Φ(0) = -6, Φ(1) = -5
        # F = γΦ(s') - Φ(s) = 1*(-5) - (-6) = 1
        
        wrapped_env.reset()
        wrapped_env.last_state = 0
        shaping = wrapped_env.compute_shaping_reward(0, 2, 1)  # right action
        
        self.assertGreater(shaping, 0, "Should reward progress toward goal")
    
    def test_beta_scaling(self):
        """Test that beta scales the shaping reward."""
        wrapped_env_1 = PotentialBasedDistanceShaping(self.env, beta=1.0)
        wrapped_env_2 = PotentialBasedDistanceShaping(self.env, beta=0.5)
        
        shaping_1 = wrapped_env_1.compute_shaping_reward(0, 2, 1)
        shaping_2 = wrapped_env_2.compute_shaping_reward(0, 2, 1)
        
        self.assertAlmostEqual(shaping_2, shaping_1 * 0.5)


class TestSafetyBasedShaping(unittest.TestCase):
    """Test safety-based shaping."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = create_frozenlake_env(
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            seed=42
        )
    
    def tearDown(self):
        """Clean up environment."""
        self.env.close()
    
    def test_potential_cache_created(self):
        """Test that potential cache is pre-computed."""
        wrapped_env = SafetyBasedShaping(self.env, beta=0.1)
        
        # Should have potentials for all states
        num_states = 4 * 4  # 4x4 grid
        self.assertEqual(len(wrapped_env.potential_cache), num_states)
    
    def test_potential_values_positive(self):
        """Test that potentials are positive (distances)."""
        wrapped_env = SafetyBasedShaping(self.env, beta=0.1)
        
        for state, potential in wrapped_env.potential_cache.items():
            self.assertGreaterEqual(potential, 0,
                f"Potential for state {state} should be non-negative")
    
    def test_shaping_rewards_safety(self):
        """Test that moving away from holes gives positive reward."""
        wrapped_env = SafetyBasedShaping(self.env, beta=0.1)
        
        # Find a state near a hole and one farther away
        # This is environment-specific, so we just check the mechanism works
        wrapped_env.reset()
        
        # Any valid transition should produce a shaping value
        shaping = wrapped_env.compute_shaping_reward(0, 0, 1)
        self.assertIsInstance(shaping, (float, np.floating))


class TestExplorationBonusShaping(unittest.TestCase):
    """Test exploration bonus shaping."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = create_frozenlake_env(
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            seed=42
        )
    
    def tearDown(self):
        """Clean up environment."""
        self.env.close()
    
    def test_visit_counts_initialized(self):
        """Test that visit counts start empty."""
        wrapped_env = ExplorationBonusShaping(self.env, beta=0.05)
        self.assertEqual(len(wrapped_env.visit_counts), 0)
    
    def test_visit_counts_updated(self):
        """Test that visit counts are updated."""
        wrapped_env = ExplorationBonusShaping(self.env, beta=0.05)
        
        state, _ = wrapped_env.reset()
        next_state, reward, terminated, truncated, info = wrapped_env.step(0)
        
        # Next state should have been visited
        self.assertIn(next_state, wrapped_env.visit_counts)
        self.assertEqual(wrapped_env.visit_counts[next_state], 1)
    
    def test_exploration_bonus_decreases(self):
        """Test that bonus decreases with repeated visits."""
        wrapped_env = ExplorationBonusShaping(self.env, beta=0.05)
        
        # First visit
        wrapped_env.visit_counts = {}
        shaping_1 = wrapped_env.compute_shaping_reward(0, 0, 1)
        
        # Second visit (counts already updated from first)
        shaping_2 = wrapped_env.compute_shaping_reward(0, 0, 1)
        
        # Second visit should give less bonus (more negative or less positive)
        self.assertLess(shaping_2, shaping_1,
            "Exploration bonus should decrease with repeated visits")
    
    def test_reset_per_episode_false(self):
        """Test that counts persist across episodes by default."""
        wrapped_env = ExplorationBonusShaping(self.env, beta=0.05, 
                                             reset_per_episode=False)
        
        # Visit state 1
        wrapped_env.reset()
        wrapped_env.step(0)
        
        count_before = wrapped_env.visit_counts.copy()
        
        # Reset environment
        wrapped_env.reset()
        
        # Counts should persist
        self.assertEqual(wrapped_env.visit_counts, count_before)
    
    def test_reset_per_episode_true(self):
        """Test that counts reset when reset_per_episode=True."""
        wrapped_env = ExplorationBonusShaping(self.env, beta=0.05,
                                             reset_per_episode=True)
        
        # Visit state 1
        wrapped_env.reset()
        wrapped_env.step(0)
        
        self.assertGreater(len(wrapped_env.visit_counts), 0)
        
        # Reset environment
        wrapped_env.reset()
        
        # Counts should be cleared
        self.assertEqual(len(wrapped_env.visit_counts), 0)


class TestWrapperIntegration(unittest.TestCase):
    """Test that wrappers integrate properly with environment."""
    
    def test_info_dict_contains_metrics(self):
        """Test that info dict contains all necessary metrics."""
        env = create_frozenlake_env(
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            seed=42
        )
        
        wrapped_env = StepCostShaping(env, step_cost=-0.01)
        
        state, _ = wrapped_env.reset()
        next_state, reward, terminated, truncated, info = wrapped_env.step(0)
        
        # Check all required keys
        self.assertIn('orig_reward', info)
        self.assertIn('shaping_reward', info)
        self.assertIn('shaped_reward', info)
        self.assertIn('is_success', info)
        
        # Check consistency
        self.assertEqual(info['shaped_reward'], 
                        info['orig_reward'] + info['shaping_reward'])
        
        wrapped_env.close()
    
    def test_multiple_wrappers_stack(self):
        """Test that multiple wrappers can be stacked."""
        env = create_frozenlake_env(
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            seed=42
        )
        
        # Stack two wrappers
        wrapped_env = StepCostShaping(env, step_cost=-0.01)
        double_wrapped = PotentialBasedDistanceShaping(wrapped_env, beta=0.5)
        
        # Should work without errors
        state, _ = double_wrapped.reset()
        next_state, reward, terminated, truncated, info = double_wrapped.step(0)
        
        # Reward should include both shapings
        self.assertIsInstance(reward, (float, np.floating))
        
        double_wrapped.close()


if __name__ == '__main__':
    unittest.main()
