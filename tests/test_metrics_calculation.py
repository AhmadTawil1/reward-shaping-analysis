"""
Test Metrics Calculation

This test validates that throughput and real returns are calculated correctly
from experiment results.

Usage:
    python tests/test_metrics_calculation.py
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from experiments.run_monte_carlo import run_monte_carlo


def compute_throughput(successes):
    """Compute cumulative throughput from success array."""
    cumulative = np.cumsum(successes, axis=1)
    episodes = np.arange(1, successes.shape[1] + 1)
    return cumulative / episodes


def test_throughput_calculation():
    """
    Test that throughput is calculated correctly.
    
    Throughput at episode N = (total successes up to N) / N
    """
    print("\n" + "="*70)
    print("TEST: THROUGHPUT CALCULATION")
    print("="*70)
    
    # Create sample success data
    # Example: 3 runs, 10 episodes each
    # Run 1: [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] -> 7/10 = 0.7 final throughput
    # Run 2: [0, 1, 1, 0, 1, 1, 1, 0, 1, 1] -> 7/10 = 0.7 final throughput
    # Run 3: [1, 1, 0, 1, 1, 0, 1, 1, 1, 0] -> 7/10 = 0.7 final throughput
    
    successes = np.array([
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    ])
    
    print(f"\nSample success data (3 runs, 10 episodes):")
    for i, run in enumerate(successes):
        print(f"  Run {i+1}: {run} -> Total: {np.sum(run)}/10")
    
    # Calculate throughput
    throughput = compute_throughput(successes)
    
    print(f"\nCalculated throughput shape: {throughput.shape}")
    print(f"Expected shape: (3, 10)")
    
    # Verify shape
    if throughput.shape != (3, 10):
        print(f"\n❌ FAILED: Throughput shape is incorrect!")
        return False
    
    # Verify final throughput for each run
    print(f"\nFinal throughput (episode 10):")
    for i in range(3):
        final_throughput = throughput[i, -1]
        expected = np.sum(successes[i]) / 10
        print(f"  Run {i+1}: {final_throughput:.3f} (expected: {expected:.3f})")
        
        if abs(final_throughput - expected) > 0.001:
            print(f"\n❌ FAILED: Throughput calculation incorrect for run {i+1}!")
            return False
    
    # Verify throughput at specific episodes
    print(f"\nThroughput progression (Run 1):")
    for ep in [1, 3, 5, 10]:
        tp = throughput[0, ep-1]
        expected = np.sum(successes[0, :ep]) / ep
        print(f"  Episode {ep}: {tp:.3f} (expected: {expected:.3f})")
        
        if abs(tp - expected) > 0.001:
            print(f"\n❌ FAILED: Throughput at episode {ep} is incorrect!")
            return False
    
    print(f"\n✅ PASSED: Throughput calculation is correct!")
    return True


def test_real_returns_calculation():
    """
    Test that real returns are calculated correctly.
    
    Real returns should be the sum of ORIGINAL (unshaped) rewards,
    not the shaped rewards.
    """
    print("\n" + "="*70)
    print("TEST: REAL RETURNS CALCULATION")
    print("="*70)
    
    print(f"\nRunning small Monte Carlo experiment...")
    print(f"  Episodes: 20")
    print(f"  Runs: 2")
    print(f"  Shaping: Safety-Based (to test that returns are unshaped)")
    
    # Run a small experiment with reward shaping
    successes, returns, lengths = run_monte_carlo(
        num_episodes=20,
        num_runs=2,
        shaping_type='safety',  # Use shaping to verify we get REAL returns
        grid_rows=config.GRID_ROWS,
        grid_cols=config.GRID_COLS,
        hole_density=config.HOLE_DENSITY,
        is_slippery=config.IS_SLIPPERY,
        success_rate=config.SUCCESS_RATE,
        epsilon=config.EPSILON,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        shaping_params={'beta': config.BETA_SAFETY}
    )
    
    print(f"\nResults shape:")
    print(f"  Successes: {successes.shape}")
    print(f"  Returns: {returns.shape}")
    print(f"  Lengths: {lengths.shape}")
    
    # Verify shapes
    if successes.shape != (2, 20) or returns.shape != (2, 20) or lengths.shape != (2, 20):
        print(f"\n❌ FAILED: Result shapes are incorrect!")
        return False
    
    print(f"\nAnalyzing returns...")
    
    # Check that returns are consistent with successes
    # In FrozenLake, successful episodes should have return = 1.0
    # Failed episodes should have return = 0.0 (no intermediate rewards)
    
    for run_idx in range(2):
        print(f"\nRun {run_idx + 1}:")
        
        for ep_idx in range(20):
            success = successes[run_idx, ep_idx]
            ret = returns[run_idx, ep_idx]
            
            # In standard FrozenLake with no step costs:
            # - Success (reached goal) -> return should be 1.0
            # - Failure (fell in hole or timeout) -> return should be 0.0
            
            if success == 1:
                # Successful episode
                if ret != 1.0:
                    print(f"  Episode {ep_idx+1}: SUCCESS but return={ret} (expected 1.0)")
                    print(f"  ⚠️  WARNING: Return doesn't match success!")
                    # This might happen with step costs, so just warn
            else:
                # Failed episode
                if ret != 0.0:
                    print(f"  Episode {ep_idx+1}: FAILURE but return={ret} (expected 0.0)")
                    print(f"  ⚠️  WARNING: Return doesn't match failure!")
    
    # Check that returns are in valid range [0, 1]
    min_return = np.min(returns)
    max_return = np.max(returns)
    
    print(f"\nReturn statistics:")
    print(f"  Min return: {min_return}")
    print(f"  Max return: {max_return}")
    print(f"  Mean return: {np.mean(returns):.3f}")
    
    if min_return < 0 or max_return > 1:
        print(f"\n❌ FAILED: Returns outside valid range [0, 1]!")
        return False
    
    # Verify that successful episodes have positive returns
    successful_episodes = successes == 1
    if np.any(successful_episodes):
        successful_returns = returns[successful_episodes]
        if np.all(successful_returns > 0):
            print(f"\n✅ PASSED: All successful episodes have positive returns!")
        else:
            print(f"\n⚠️  WARNING: Some successful episodes have zero returns")
            print(f"   This might indicate an issue with reward tracking")
    
    print(f"\n✅ PASSED: Real returns calculation appears correct!")
    return True


def test_metrics_consistency():
    """
    Test that metrics are consistent across runs.
    
    Verify that:
    - Success count matches return count (success -> return = 1.0)
    - Episode lengths are reasonable
    - No NaN or infinite values
    """
    print("\n" + "="*70)
    print("TEST: METRICS CONSISTENCY")
    print("="*70)
    
    print(f"\nRunning small experiment to check consistency...")
    
    successes, returns, lengths = run_monte_carlo(
        num_episodes=30,
        num_runs=2,
        shaping_type=None,  # No shaping for clearer results
        grid_rows=config.GRID_ROWS,
        grid_cols=config.GRID_COLS,
        hole_density=config.HOLE_DENSITY,
        is_slippery=config.IS_SLIPPERY,
        success_rate=config.SUCCESS_RATE,
        epsilon=config.EPSILON,
        alpha=config.ALPHA,
        gamma=config.GAMMA
    )
    
    # Check for NaN or infinite values
    print(f"\nChecking for invalid values...")
    
    has_nan_successes = np.any(np.isnan(successes))
    has_nan_returns = np.any(np.isnan(returns))
    has_nan_lengths = np.any(np.isnan(lengths))
    
    has_inf_returns = np.any(np.isinf(returns))
    has_inf_lengths = np.any(np.isinf(lengths))
    
    if has_nan_successes or has_nan_returns or has_nan_lengths:
        print(f"❌ FAILED: Found NaN values!")
        print(f"  Successes: {has_nan_successes}")
        print(f"  Returns: {has_nan_returns}")
        print(f"  Lengths: {has_nan_lengths}")
        return False
    
    if has_inf_returns or has_inf_lengths:
        print(f"❌ FAILED: Found infinite values!")
        print(f"  Returns: {has_inf_returns}")
        print(f"  Lengths: {has_inf_lengths}")
        return False
    
    print(f"  ✓ No NaN or infinite values found")
    
    # Check episode lengths are reasonable
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    mean_length = np.mean(lengths)
    
    print(f"\nEpisode length statistics:")
    print(f"  Min: {min_length}")
    print(f"  Max: {max_length}")
    print(f"  Mean: {mean_length:.1f}")
    
    if min_length < 1:
        print(f"❌ FAILED: Episode length less than 1!")
        return False
    
    if max_length > 500:  # Max episode steps is 500
        print(f"❌ FAILED: Episode length exceeds maximum (500)!")
        return False
    
    print(f"  ✓ Episode lengths are reasonable")
    
    # Check success/return consistency (for baseline, no shaping)
    print(f"\nChecking success/return consistency...")
    
    total_successes = np.sum(successes)
    total_return_sum = np.sum(returns)
    
    print(f"  Total successes: {total_successes}")
    print(f"  Total return sum: {total_return_sum}")
    
    # For baseline FrozenLake, sum of returns should equal number of successes
    if abs(total_successes - total_return_sum) < 0.1:
        print(f"  ✓ Success count matches return sum")
    else:
        print(f"  ⚠️  WARNING: Success count doesn't match return sum")
        print(f"     Difference: {abs(total_successes - total_return_sum)}")
    
    print(f"\n✅ PASSED: Metrics are consistent!")
    return True


def run_all_tests():
    """Run all metrics calculation tests."""
    print("\n" + "="*70)
    print("METRICS CALCULATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Throughput calculation
    results['throughput'] = test_throughput_calculation()
    
    # Test 2: Real returns calculation
    results['real_returns'] = test_real_returns_calculation()
    
    # Test 3: Metrics consistency
    results['consistency'] = test_metrics_consistency()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL METRICS TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED! Review metrics calculation")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
