# Test Suite Documentation

This directory contains comprehensive tests for the Reward Shaping Analysis project.

## Test Structure

```
tests/
├── __init__.py                  # Package initialization
├── test_environment.py          # Environment generation tests
├── test_algorithms.py           # Algorithm tests (MC, SARSA)
├── test_reward_shaping.py       # Reward shaping wrapper tests
├── test_integration.py          # End-to-end integration tests
├── run_tests.py                 # Test runner script
└── README.md                    # This file
```

## Test Coverage

### 1. Environment Tests (`test_environment.py`)
Tests for FrozenLake environment generation:
- ✅ Map generation with valid parameters
- ✅ Protected zones (no holes adjacent to start/goal)
- ✅ Path reachability (BFS validation)
- ✅ Reproducibility with seeds
- ✅ Custom slipperiness wrapper
- ✅ Invalid parameter handling
- ✅ Helper function correctness

**Key Tests:**
- `test_protected_zones_start()` - Ensures no holes near start
- `test_protected_zones_goal()` - Ensures no holes near goal
- `test_path_reachability()` - Validates BFS path finding
- `test_reproducibility()` - Checks seed consistency

### 2. Algorithm Tests (`test_algorithms.py`)
Tests for Monte Carlo and SARSA agents:
- ✅ Agent initialization
- ✅ Optimistic Q-value initialization
- ✅ ε-greedy action selection (exploration/exploitation)
- ✅ Episode generation
- ✅ Q-value updates
- ✅ Training execution
- ✅ Policy extraction
- ✅ Value function extraction
- ✅ Learning on simple tasks

**Key Tests:**
- `test_action_selection_exploration()` - Validates exploration
- `test_action_selection_exploitation()` - Validates exploitation
- `test_both_algorithms_learn()` - Ensures both MC and SARSA learn

### 3. Reward Shaping Tests (`test_reward_shaping.py`)
Tests for all reward shaping wrappers:
- ✅ Base wrapper functionality
- ✅ Step-cost shaping
- ✅ Potential-based distance shaping
- ✅ Safety-based shaping (custom)
- ✅ Exploration bonus shaping (custom)
- ✅ Wrapper stacking
- ✅ Info dict metrics

**Key Tests:**
- `test_potential_function()` - Validates distance calculations
- `test_shaping_encourages_progress()` - Checks goal-directed shaping
- `test_exploration_bonus_decreases()` - Validates novelty decay
- `test_multiple_wrappers_stack()` - Tests wrapper composition

### 4. Integration Tests (`test_integration.py`)
End-to-end tests for the complete pipeline:
- ✅ Monte Carlo with all shaping methods
- ✅ SARSA with all shaping methods
- ✅ Reproducibility across runs
- ✅ Config integration
- ✅ Learning progress verification

**Key Tests:**
- `test_mc_reproducibility()` - Ensures deterministic results
- `test_sarsa_reproducibility()` - Ensures deterministic results
- `test_mc_learns_simple_task()` - Validates learning improvement
- `test_sarsa_learns_simple_task()` - Validates learning improvement

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Test Suite
```bash
python tests/run_tests.py --suite test_environment
python tests/run_tests.py --suite test_algorithms
python tests/run_tests.py --suite test_reward_shaping
python tests/run_tests.py --suite test_integration
```

### Run Fast Tests Only (Skip Integration)
```bash
python tests/run_tests.py --fast
```

### Adjust Verbosity
```bash
python tests/run_tests.py --verbosity 0  # Quiet
python tests/run_tests.py --verbosity 1  # Normal
python tests/run_tests.py --verbosity 2  # Verbose (default)
```

### Run Individual Test File
```bash
python -m unittest tests.test_environment
python -m unittest tests.test_algorithms
python -m unittest tests.test_reward_shaping
python -m unittest tests.test_integration
```

### Run Specific Test Class
```bash
python -m unittest tests.test_environment.TestFrozenLakeEnvironment
python -m unittest tests.test_algorithms.TestMonteCarloAgent
```

### Run Specific Test Method
```bash
python -m unittest tests.test_environment.TestFrozenLakeEnvironment.test_protected_zones_start
```

## Test Statistics

| Test Suite | Number of Tests | Estimated Runtime |
|------------|----------------|-------------------|
| Environment | 15+ | ~5 seconds |
| Algorithms | 15+ | ~10 seconds |
| Reward Shaping | 20+ | ~8 seconds |
| Integration | 15+ | ~60 seconds |
| **Total** | **65+** | **~90 seconds** |

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python tests/run_tests.py
```

## Writing New Tests

When adding new features, follow these guidelines:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test components working together
3. **Use setUp/tearDown**: Clean up resources (especially environments)
4. **Test Edge Cases**: Invalid inputs, boundary conditions
5. **Test Reproducibility**: Use seeds to ensure deterministic behavior
6. **Descriptive Names**: Use clear, descriptive test method names
7. **Assertions**: Use specific assertions (assertEqual, assertGreater, etc.)

### Example Test Template

```python
import unittest
from your_module import YourClass

class TestYourClass(unittest.TestCase):
    """Test suite for YourClass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.instance = YourClass()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up resources
        pass
    
    def test_feature(self):
        """Test a specific feature."""
        result = self.instance.method()
        self.assertEqual(result, expected_value)

if __name__ == '__main__':
    unittest.main()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root
2. **Random Failures**: Check that seeds are set properly
3. **Slow Tests**: Use `--fast` flag to skip integration tests during development
4. **Environment Not Closing**: Ensure `tearDown()` calls `env.close()`

### Debug Mode

Run tests with Python's debugger:
```bash
python -m pdb tests/run_tests.py
```

### Coverage Report

Generate test coverage report:
```bash
pip install coverage
coverage run -m unittest discover tests
coverage report
coverage html  # Generate HTML report
```

## Best Practices

- ✅ Run tests before committing
- ✅ Write tests for new features
- ✅ Keep tests fast and focused
- ✅ Use meaningful test names
- ✅ Test both success and failure cases
- ✅ Mock external dependencies when appropriate
- ✅ Maintain test independence (no test should depend on another)

## Contact

For questions about the test suite, please refer to the main project documentation or open an issue.
