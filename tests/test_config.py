"""
Unit tests for configuration module.

Tests:
- Config parameter retrieval
- Shaping parameter generation
- Test vs final config differences
- Parameter validation
"""

import unittest
import config


class TestConfigModule(unittest.TestCase):
    """Test suite for config module."""
    
    def test_random_seed_exists(self):
        """Test that random seed is defined."""
        self.assertIsNotNone(config.RANDOM_SEED)
        self.assertIsInstance(config.RANDOM_SEED, int)
    
    def test_test_mode_flag(self):
        """Test that TEST_MODE flag exists."""
        self.assertIsInstance(config.TEST_MODE, bool)
    
    def test_results_dir_exists(self):
        """Test that results directory is defined."""
        self.assertIsInstance(config.RESULTS_DIR, str)
        self.assertGreater(len(config.RESULTS_DIR), 0)
    
    def test_get_test_config(self):
        """Test test configuration retrieval."""
        cfg = config.get_test_config()
        
        # Check all required keys
        required_keys = [
            'num_episodes', 'num_runs', 'grid_rows', 'grid_cols', 'hole_density',
            'is_slippery', 'epsilon', 'alpha', 'gamma'
        ]
        for key in required_keys:
            self.assertIn(key, cfg, f"Missing key: {key}")
        
        # Check gamma is 1.0 (project requirement)
        self.assertEqual(cfg['gamma'], 1.0)
        
        # Check reasonable values
        self.assertGreater(cfg['num_episodes'], 0)
        self.assertGreater(cfg['num_runs'], 0)
        self.assertGreaterEqual(cfg['grid_rows'], 2)
        self.assertGreaterEqual(cfg['grid_cols'], 2)
        self.assertGreaterEqual(cfg['hole_density'], 0.1)
        self.assertLessEqual(cfg['hole_density'], 0.2)
        self.assertGreater(cfg['epsilon'], 0)
        self.assertLess(cfg['epsilon'], 1)
        self.assertGreater(cfg['alpha'], 0)
        self.assertLessEqual(cfg['alpha'], 1)
    
    def test_get_final_config(self):
        """Test final configuration retrieval."""
        cfg = config.get_final_config()
        
        # Check all required keys
        required_keys = [
            'num_episodes', 'num_runs', 'grid_rows', 'grid_cols', 'hole_density',
            'is_slippery', 'epsilon', 'alpha', 'gamma'
        ]
        for key in required_keys:
            self.assertIn(key, cfg, f"Missing key: {key}")
        
        # Check gamma is 1.0
        self.assertEqual(cfg['gamma'], 1.0)
    
    def test_final_vs_test_config(self):
        """Test that final config has more episodes than test config."""
        test_cfg = config.get_test_config()
        final_cfg = config.get_final_config()
        
        # Final should have more episodes for thorough evaluation
        self.assertGreater(final_cfg['num_episodes'], test_cfg['num_episodes'])
    
    def test_get_shaping_params_none(self):
        """Test shaping params for baseline (no shaping)."""
        params = config.get_shaping_params(None)
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)
    
    def test_get_shaping_params_step(self):
        """Test shaping params for step-cost shaping."""
        params = config.get_shaping_params('step')
        self.assertIsInstance(params, dict)
        self.assertIn('step_cost', params)
        self.assertLess(params['step_cost'], 0)  # Should be negative
    
    def test_get_shaping_params_potential(self):
        """Test shaping params for potential-based shaping."""
        params = config.get_shaping_params('potential')
        self.assertIsInstance(params, dict)
        self.assertIn('beta', params)
        self.assertGreater(params['beta'], 0)
    
    def test_get_shaping_params_safety(self):
        """Test shaping params for safety-based shaping."""
        params = config.get_shaping_params('safety')
        self.assertIsInstance(params, dict)
        self.assertIn('beta', params)
        self.assertGreater(params['beta'], 0)
    
    def test_get_shaping_params_exploration(self):
        """Test shaping params for exploration bonus shaping."""
        params = config.get_shaping_params('exploration')
        self.assertIsInstance(params, dict)
        self.assertIn('beta', params)
        self.assertGreater(params['beta'], 0)
    
    def test_initial_q_value(self):
        """Test that initial Q-value is defined."""
        self.assertIsInstance(config.INITIAL_Q_VALUE, (int, float))
    
    def test_hyperparameters_defined(self):
        """Test that key hyperparameters are defined."""
        # Check epsilon
        self.assertTrue(hasattr(config, 'EPSILON'))
        self.assertGreater(config.EPSILON, 0)
        self.assertLess(config.EPSILON, 1)
        
        # Check alpha
        self.assertTrue(hasattr(config, 'ALPHA'))
        self.assertGreater(config.ALPHA, 0)
        self.assertLessEqual(config.ALPHA, 1)
        
        # Check gamma
        self.assertTrue(hasattr(config, 'GAMMA'))
        self.assertEqual(config.GAMMA, 1.0)  # Project requirement
    
    def test_environment_params_defined(self):
        """Test that environment parameters are defined."""
        self.assertTrue(hasattr(config, 'GRID_ROWS'))
        self.assertTrue(hasattr(config, 'GRID_COLS'))
        self.assertGreaterEqual(config.GRID_ROWS, 2)
        self.assertGreaterEqual(config.GRID_COLS, 2)
        
        self.assertTrue(hasattr(config, 'HOLE_DENSITY'))
        self.assertGreaterEqual(config.HOLE_DENSITY, 0.1)
        self.assertLessEqual(config.HOLE_DENSITY, 0.2)
        
        self.assertTrue(hasattr(config, 'IS_SLIPPERY'))
        self.assertIsInstance(config.IS_SLIPPERY, bool)
    
    def test_shaping_params_consistency(self):
        """Test that shaping params are consistent across calls."""
        params1 = config.get_shaping_params('step')
        params2 = config.get_shaping_params('step')
        
        self.assertEqual(params1, params2)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""
    
    def test_gamma_is_one(self):
        """Test that gamma is always 1.0 (project requirement)."""
        test_cfg = config.get_test_config()
        final_cfg = config.get_final_config()
        
        self.assertEqual(test_cfg['gamma'], 1.0)
        self.assertEqual(final_cfg['gamma'], 1.0)
        self.assertEqual(config.GAMMA, 1.0)
    
    def test_hole_density_in_range(self):
        """Test that hole density is in valid range."""
        test_cfg = config.get_test_config()
        final_cfg = config.get_final_config()
        
        for cfg in [test_cfg, final_cfg]:
            self.assertGreaterEqual(cfg['hole_density'], 0.1)
            self.assertLessEqual(cfg['hole_density'], 0.2)
    
    def test_epsilon_in_range(self):
        """Test that epsilon is in valid range."""
        test_cfg = config.get_test_config()
        final_cfg = config.get_final_config()
        
        for cfg in [test_cfg, final_cfg]:
            self.assertGreater(cfg['epsilon'], 0)
            self.assertLess(cfg['epsilon'], 1)
    
    def test_alpha_in_range(self):
        """Test that alpha is in valid range."""
        test_cfg = config.get_test_config()
        final_cfg = config.get_final_config()
        
        for cfg in [test_cfg, final_cfg]:
            self.assertGreater(cfg['alpha'], 0)
            self.assertLessEqual(cfg['alpha'], 1)


if __name__ == '__main__':
    unittest.main()
