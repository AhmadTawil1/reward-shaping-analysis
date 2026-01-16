# Quick Test Reference

## Run All Tests
```bash
python tests/run_tests.py
```

## Run Specific Suite
```bash
python tests/run_tests.py --suite test_environment
python tests/run_tests.py --suite test_algorithms
python tests/run_tests.py --suite test_reward_shaping
python tests/run_tests.py --suite test_integration
python tests/run_tests.py --suite test_config
```

## Fast Mode (Skip Integration)
```bash
python tests/run_tests.py --fast
```

## Verbosity Levels
```bash
python tests/run_tests.py --verbosity 0  # Quiet
python tests/run_tests.py --verbosity 1  # Normal
python tests/run_tests.py --verbosity 2  # Verbose
```

## Individual Test Files
```bash
python -m unittest tests.test_environment
python -m unittest tests.test_algorithms
python -m unittest tests.test_reward_shaping
python -m unittest tests.test_integration
python -m unittest tests.test_config
```

## Test Coverage
- **Environment**: 15+ tests (protected zones, BFS, reproducibility)
- **Algorithms**: 15+ tests (MC, SARSA, learning)
- **Reward Shaping**: 20+ tests (all 4 shaping methods)
- **Integration**: 15+ tests (end-to-end workflows)
- **Config**: 15+ tests (parameter validation)
- **Total**: 75+ tests

## Expected Runtime
- Fast mode: ~20 seconds
- Full suite: ~90 seconds

## Exit Codes
- `0` = All tests passed ✅
- `1` = Some tests failed ❌
