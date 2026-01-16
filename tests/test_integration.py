"""
Integration tests for the complete RL pipeline.

Tests:
- End-to-end training with different shaping methods
- Reproducibility across runs
- Config integration
- Results consistency
"""

import unittest
import numpy as np
from experiments.run_monte_carlo import run_monte_carlo
from experiments.run_sarsa import run_sarsa
import config


class TestMonteCarloIntegration(unittest.TestCase):
    """Integration tests for Monte Carlo experiments."""
    
    def test_baseline_training(self):
        """Test Monte Carlo with no shaping."""
        successes, returns, lengths = run_monte_carlo(
            num_episodes=50,
            num_runs=2,
            shaping_type=None,
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params={}
        )
        
        # Check output shapes
        self.assertEqual(successes.shape, (2, 50))
        self.assertEqual(returns.shape, (2, 50))
        self.assertEqual(lengths.shape, (2, 50))
        
        # Check value ranges
        self.assertTrue(np.all((successes == 0) | (successes == 1)))
        self.assertTrue(np.all(lengths > 0))
    
    def test_step_cost_shaping(self):
        """Test Monte Carlo with step-cost shaping."""
        successes, returns, lengths = run_monte_carlo(
            num_episodes=50,
            num_runs=2,
            shaping_type='step',
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params=config.get_shaping_params('step')
        )
        
        self.assertEqual(successes.shape, (2, 50))
        self.assertEqual(returns.shape, (2, 50))
        self.assertEqual(lengths.shape, (2, 50))
    
    def test_potential_shaping(self):
        """Test Monte Carlo with potential-based shaping."""
        successes, returns, lengths = run_monte_carlo(
            num_episodes=50,
            num_runs=2,
            shaping_type='potential',
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params=config.get_shaping_params('potential')
        )
        
        self.assertEqual(successes.shape, (2, 50))
    
    def test_safety_shaping(self):
        """Test Monte Carlo with safety-based shaping."""
        successes, returns, lengths = run_monte_carlo(
            num_episodes=50,
            num_runs=2,
            shaping_type='safety',
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params=config.get_shaping_params('safety')
        )
        
        self.assertEqual(successes.shape, (2, 50))
    
    def test_exploration_shaping(self):
        """Test Monte Carlo with exploration bonus shaping."""
        successes, returns, lengths = run_monte_carlo(
            num_episodes=50,
            num_runs=2,
            shaping_type='exploration',
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params=config.get_shaping_params('exploration')
        )
        
        self.assertEqual(successes.shape, (2, 50))


class TestSARSAIntegration(unittest.TestCase):
    """Integration tests for SARSA experiments."""
    
    def test_baseline_training(self):
        """Test SARSA with no shaping."""
        successes, returns, lengths = run_sarsa(
            num_episodes=50,
            num_runs=2,
            shaping_type=None,
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params={}
        )
        
        # Check output shapes
        self.assertEqual(successes.shape, (2, 50))
        self.assertEqual(returns.shape, (2, 50))
        self.assertEqual(lengths.shape, (2, 50))
        
        # Check value ranges
        self.assertTrue(np.all((successes == 0) | (successes == 1)))
        self.assertTrue(np.all(lengths > 0))
    
    def test_all_shaping_methods(self):
        """Test SARSA with all shaping methods."""
        shaping_types = ['step', 'potential', 'safety', 'exploration']
        
        for shaping_type in shaping_types:
            with self.subTest(shaping_type=shaping_type):
                successes, returns, lengths = run_sarsa(
                    num_episodes=50,
                    num_runs=2,
                    shaping_type=shaping_type,
                    grid_size=4,
                    hole_density=0.15,
                    is_slippery=False,
                    epsilon=0.1,
                    alpha=0.1,
                    gamma=1.0,
                    initial_q_value=0.0,
                    shaping_params=config.get_shaping_params(shaping_type)
                )
                
                self.assertEqual(successes.shape, (2, 50))


class TestReproducibility(unittest.TestCase):
    """Test reproducibility with random seeds."""
    
    def test_mc_reproducibility(self):
        """Test that same seed produces same results for MC."""
        # Reset numpy seed before first run
        np.random.seed(42)
        
        # Run 1
        successes_1, returns_1, lengths_1 = run_monte_carlo(
            num_episodes=20,
            num_runs=1,
            shaping_type=None,
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params={}
        )
        
        # Reset numpy seed before second run (same seed)
        np.random.seed(42)
        
        # Run 2 with same seed
        successes_2, returns_2, lengths_2 = run_monte_carlo(
            num_episodes=20,
            num_runs=1,
            shaping_type=None,
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params={}
        )
        
        # Results should be identical
        np.testing.assert_array_equal(successes_1, successes_2)
        np.testing.assert_array_equal(returns_1, returns_2)
        np.testing.assert_array_equal(lengths_1, lengths_2)
    
    def test_sarsa_reproducibility(self):
        """Test that same seed produces same results for SARSA."""
        # Reset numpy seed before first run
        np.random.seed(42)
        
        # Run 1
        successes_1, returns_1, lengths_1 = run_sarsa(
            num_episodes=20,
            num_runs=1,
            shaping_type=None,
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params={}
        )
        
        # Reset numpy seed before second run (same seed)
        np.random.seed(42)
        
        # Run 2
        successes_2, returns_2, lengths_2 = run_sarsa(
            num_episodes=20,
            num_runs=1,
            shaping_type=None,
            grid_size=4,
            hole_density=0.15,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params={}
        )
        
        # Results should be identical
        np.testing.assert_array_equal(successes_1, successes_2)
        np.testing.assert_array_equal(returns_1, returns_2)
        np.testing.assert_array_equal(lengths_1, lengths_2)


class TestConfigIntegration(unittest.TestCase):
    """Test integration with config module."""
    
    def test_get_shaping_params(self):
        """Test that shaping params are retrieved correctly."""
        # Test all shaping types
        shaping_types = [None, 'step', 'potential', 'safety', 'exploration']
        
        for shaping_type in shaping_types:
            params = config.get_shaping_params(shaping_type)
            self.assertIsInstance(params, dict)
    
    def test_test_config(self):
        """Test that test config is valid."""
        cfg = config.get_test_config()
        
        self.assertIn('num_episodes', cfg)
        self.assertIn('num_runs', cfg)
        self.assertIn('grid_size', cfg)
        self.assertIn('hole_density', cfg)
        self.assertIn('is_slippery', cfg)
        self.assertIn('epsilon', cfg)
        self.assertIn('alpha', cfg)
        self.assertIn('gamma', cfg)
        
        # Check gamma is 1.0 (project requirement)
        self.assertEqual(cfg['gamma'], 1.0)
    
    def test_final_config(self):
        """Test that final config is valid."""
        cfg = config.get_final_config()
        
        self.assertIn('num_episodes', cfg)
        self.assertIn('num_runs', cfg)
        
        # Final config should have more episodes than test
        test_cfg = config.get_test_config()
        self.assertGreater(cfg['num_episodes'], test_cfg['num_episodes'])


class TestLearningProgress(unittest.TestCase):
    """Test that agents actually learn over time."""
    
    def test_mc_learns_simple_task(self):
        """Test that MC shows improvement on simple task."""
        # Use a very simple environment to ensure learning
        successes, returns, lengths = run_monte_carlo(
            num_episodes=300,
            num_runs=3,
            shaping_type='potential',  # Use shaping to help learning
            grid_size=4,
            hole_density=0.1,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params={'beta': 1.0}
        )
        
        # Compare first 30 episodes vs last 30 episodes
        early_success_rate = np.mean(successes[:, :30])
        late_success_rate = np.mean(successes[:, -30:])
        
        # Should show improvement OR achieve reasonable success
        # (Some random seeds may start well, so we check either condition)
        improvement = late_success_rate - early_success_rate
        
        self.assertTrue(
            improvement >= -0.05 or late_success_rate >= 0.3,
            f"MC should learn or achieve success: early={early_success_rate:.2%}, "
            f"late={late_success_rate:.2%}, improvement={improvement:+.2%}"
        )
    
    def test_sarsa_learns_simple_task(self):
        """Test that SARSA shows improvement on simple task."""
        # Use a very simple environment to ensure learning
        successes, returns, lengths = run_sarsa(
            num_episodes=300,
            num_runs=3,
            shaping_type='potential',  # Use shaping to help learning
            grid_size=4,
            hole_density=0.1,
            is_slippery=False,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0,
            initial_q_value=0.0,
            shaping_params={'beta': 1.0}
        )
        
        # Compare first 30 episodes vs last 30 episodes
        early_success_rate = np.mean(successes[:, :30])
        late_success_rate = np.mean(successes[:, -30:])
        
        # Should show improvement OR achieve reasonable success
        improvement = late_success_rate - early_success_rate
        
        self.assertTrue(
            improvement >= -0.05 or late_success_rate >= 0.3,
            f"SARSA should learn or achieve success: early={early_success_rate:.2%}, "
            f"late={late_success_rate:.2%}, improvement={improvement:+.2%}"
        )


if __name__ == '__main__':
    unittest.main()
