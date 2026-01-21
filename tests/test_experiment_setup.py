"""
Pre-Experiment Validation Tests

Tests to run BEFORE executing compare_methods.py to ensure:
1. All runs use the same map (seed consistency)
2. Success rate is correctly set to 0.7 (stochastic behavior)
3. Results are reproducible (same seed = same results)

Usage:
    python tests\test_experiment_setup.py
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from env.frozenlake_env import create_frozenlake_env
from experiments.run_monte_carlo import run_monte_carlo
from experiments.run_sarsa import run_sarsa


def test_map_consistency():
    """
    Test 1: Verify all runs generate the same map.
    
    This test creates multiple environments and checks that they all
    have identical map layouts when using config.RANDOM_SEED.
    """
    print("\n" + "="*70)
    print("TEST 1: MAP CONSISTENCY ACROSS RUNS")
    print("="*70)
    
    num_test_runs = 5
    maps = []
    
    print(f"\nGenerating {num_test_runs} environments with seed={config.RANDOM_SEED}...")
    
    for i in range(num_test_runs):
        env = create_frozenlake_env(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            hole_density=config.HOLE_DENSITY,
            is_slippery=config.IS_SLIPPERY,
            seed=config.RANDOM_SEED,
            success_rate=config.SUCCESS_RATE
        )
        
        # Access the underlying FrozenLake environment's desc attribute
        # If wrapped, unwrap to get the base environment
        unwrapped_env = env.unwrapped
        maps.append(unwrapped_env.desc.copy())
        env.close()
        print(f"  Run {i+1}: Map generated")
    
    # Check if all maps are identical
    all_identical = True
    for i in range(1, num_test_runs):
        if not np.array_equal(maps[0], maps[i]):
            all_identical = False
            print(f"\n❌ FAILED: Map {i+1} differs from Map 1")
            print(f"Map 1:\n{maps[0]}")
            print(f"Map {i+1}:\n{maps[i]}")
            break
    
    if all_identical:
        print(f"\n✅ PASSED: All {num_test_runs} runs generated identical maps!")
        print(f"\nMap layout (seed={config.RANDOM_SEED}):")
        print(maps[0])
        print(f"\nLegend: S=Start, F=Frozen, H=Hole, G=Goal")
        return True
    else:
        print(f"\n❌ FAILED: Maps are not consistent across runs!")
        return False


def test_success_rate():
    """
    Test 2: Verify success rate is 0.7 (stochastic behavior).
    
    This test checks the transition probability dictionary to verify
    that the success rate is correctly set to 0.7.
    """
    print("\n" + "="*70)
    print("TEST 2: SUCCESS RATE VERIFICATION (0.7 expected)")
    print("="*70)
    
    env = create_frozenlake_env(
        rows=config.GRID_ROWS,
        cols=config.GRID_COLS,
        hole_density=config.HOLE_DENSITY,
        is_slippery=config.IS_SLIPPERY,
        seed=config.RANDOM_SEED,
        success_rate=config.SUCCESS_RATE
    )
    
    print(f"\nVerifying transition probabilities in slippery environment...")
    print(f"Expected success rate: {config.SUCCESS_RATE}")
    print(f"Expected fail rate (per side): {(1.0 - config.SUCCESS_RATE) / 2.0}")
    
    # Access the transition probability dictionary
    # For wrapped environments, we need to unwrap
    unwrapped_env = env.unwrapped
    P = unwrapped_env.P
    
    # Check a few states to verify probabilities
    # State 0 (start position), action 1 (RIGHT)
    test_state = 1  # Use state 1 (not start, not hole, not goal)
    test_action = 1  # RIGHT
    
    if test_state in P and test_action in P[test_state]:
        transitions = P[test_state][test_action]
        
        print(f"\nChecking transition probabilities for state {test_state}, action {test_action} (RIGHT):")
        print(f"Number of possible outcomes: {len(transitions)}")
        
        # Find the intended direction probability
        # In FrozenLake, transitions are: [(prob, next_state, reward, done), ...]
        probabilities = [t[0] for t in transitions]
        
        print(f"Transition probabilities: {probabilities}")
        
        # Check if we have the expected probabilities
        expected_success = config.SUCCESS_RATE
        expected_fail = (1.0 - config.SUCCESS_RATE) / 2.0
        
        # Allow small floating point tolerance
        tolerance = 0.01
        
        # Check if probabilities match expected values
        has_success_prob = any(abs(p - expected_success) < tolerance for p in probabilities)
        has_fail_prob = any(abs(p - expected_fail) < tolerance for p in probabilities)
        
        print(f"\nExpected probabilities:")
        print(f"  Success (intended direction): {expected_success}")
        print(f"  Fail (each perpendicular): {expected_fail}")
        
        if has_success_prob and has_fail_prob:
            print(f"\n✅ PASSED: Transition probabilities match expected success rate!")
            print(f"   Found success probability: {expected_success}")
            print(f"   Found fail probabilities: {expected_fail}")
            env.close()
            return True
        else:
            print(f"\n❌ FAILED: Transition probabilities don't match!")
            print(f"   Expected: {expected_success}, {expected_fail}, {expected_fail}")
            print(f"   Got: {probabilities}")
            env.close()
            return False
    else:
        print(f"\n⚠️  WARNING: Could not access transition probabilities")
        print(f"   State {test_state} or action {test_action} not found in P dictionary")
        env.close()
        return False


def test_reproducibility():
    """
    Test 3: Verify experiments produce identical results.
    
    This test runs the same experiment twice and checks that
    all results (successes, returns, lengths) are identical.
    """
    print("\n" + "="*70)
    print("TEST 3: REPRODUCIBILITY (Same seed = Same results)")
    print("="*70)
    
    # Use small parameters for quick testing
    test_episodes = 50
    test_runs = 3
    
    print(f"\nRunning Monte Carlo experiment twice with same configuration...")
    print(f"  Episodes: {test_episodes}")
    print(f"  Runs: {test_runs}")
    print(f"  Seed: {config.RANDOM_SEED}")
    print(f"  Shaping: Safety-Based")
    
    # First run
    print(f"\n[1/2] First execution...")
    np.random.seed(config.RANDOM_SEED)  # Reset global seed
    successes1, returns1, lengths1 = run_monte_carlo(
        num_episodes=test_episodes,
        num_runs=test_runs,
        shaping_type='safety',
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
    
    # Second run
    print(f"\n[2/2] Second execution...")
    np.random.seed(config.RANDOM_SEED)  # Reset global seed again
    successes2, returns2, lengths2 = run_monte_carlo(
        num_episodes=test_episodes,
        num_runs=test_runs,
        shaping_type='safety',
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
    
    # Compare results
    print(f"\nComparing results...")
    
    successes_match = np.array_equal(successes1, successes2)
    returns_match = np.array_equal(returns1, returns2)
    lengths_match = np.array_equal(lengths1, lengths2)
    
    print(f"  Successes identical: {successes_match}")
    print(f"  Returns identical: {returns_match}")
    print(f"  Lengths identical: {lengths_match}")
    
    if successes_match and returns_match and lengths_match:
        print(f"\n✅ PASSED: Both runs produced identical results!")
        print(f"\nSample comparison (Run 1, Last 5 episodes):")
        print(f"  Successes: {successes1[0, -5:]}")
        print(f"  Returns:   {returns1[0, -5:]}")
        print(f"  Lengths:   {lengths1[0, -5:]}")
        return True
    else:
        print(f"\n❌ FAILED: Results differ between runs!")
        if not successes_match:
            print(f"\nSuccesses mismatch (first difference):")
            diff_idx = np.where(successes1 != successes2)
            print(f"  Run 1: {successes1[diff_idx[0][0], diff_idx[1][0]]}")
            print(f"  Run 2: {successes2[diff_idx[0][0], diff_idx[1][0]]}")
        return False


def test_run_independence():
    """
    Test 4: Verify runs are independent despite using the same map.
    
    This test ensures that even though all runs use the same map,
    each run produces different learning trajectories due to:
    - Stochastic action outcomes (slippery environment)
    - Epsilon-greedy exploration
    - Different random number generator states
    """
    print("\n" + "="*70)
    print("TEST 4: RUN INDEPENDENCE")
    print("="*70)
    
    # Use larger parameters for comprehensive testing
    test_episodes = 1000  # Increased for better statistical power
    test_runs = 100  # More runs for robust statistics
    
    print(f"\nRunning Monte Carlo experiment with multiple runs...")
    print(f"  Episodes: {test_episodes}")
    print(f"  Runs: {test_runs}")
    print(f"  Seed: {config.RANDOM_SEED} (same map for all runs)")
    print(f"  Shaping: None (baseline)")
    
    # Run experiment
    np.random.seed(config.RANDOM_SEED)
    successes, returns, lengths = run_monte_carlo(
        num_episodes=test_episodes,
        num_runs=test_runs,
        shaping_type=None,
        grid_rows=config.GRID_ROWS,
        grid_cols=config.GRID_COLS,
        hole_density=config.HOLE_DENSITY,
        is_slippery=config.IS_SLIPPERY,
        success_rate=config.SUCCESS_RATE,
        epsilon=config.EPSILON,
        alpha=config.ALPHA,
        gamma=config.GAMMA
    )
    
    print(f"\nAnalyzing run independence...")
    
    # Check 1: Runs should NOT be identical
    all_runs_identical = True
    for i in range(1, test_runs):
        if not np.array_equal(successes[0], successes[i]):
            all_runs_identical = False
            break
    
    if all_runs_identical:
        print(f"\n❌ FAILED: All runs produced identical results!")
        print(f"   Runs should be independent despite using the same map")
        return False
    else:
        print(f"  ✓ Runs are NOT identical (as expected)")
    
    # Check 2: Compare pairwise differences
    print(f"\nPairwise run comparisons:")
    differences = []
    
    for i in range(test_runs):
        for j in range(i + 1, test_runs):
            # Calculate how different the runs are
            diff_successes = np.sum(successes[i] != successes[j])
            diff_percentage = (diff_successes / test_episodes) * 100
            differences.append(diff_percentage)
            
            if i < 2 and j < 3:  # Only print first few comparisons
                print(f"  Run {i+1} vs Run {j+1}: {diff_percentage:.1f}% different episodes")
    
    avg_difference = np.mean(differences)
    print(f"\nAverage difference between runs: {avg_difference:.1f}%")
    
    # Runs should be reasonably different (at least 20% different on average)
    if avg_difference < 20:
        print(f"\n⚠️  WARNING: Runs are very similar ({avg_difference:.1f}% different)")
        print(f"   Expected at least 20% difference due to stochasticity")
        print(f"   This might indicate insufficient randomness")
    else:
        print(f"  ✓ Runs show sufficient variation ({avg_difference:.1f}% different)")
    
    # Check 3: Verify different final performance across runs
    final_throughputs = []
    for i in range(test_runs):
        final_tp = np.sum(successes[i, -20:]) / 20  # Last 20 episodes
        final_throughputs.append(final_tp)
    
    print(f"\nFinal throughput (last 20 episodes) per run:")
    for i, tp in enumerate(final_throughputs):
        print(f"  Run {i+1}: {tp*100:.1f}%")
    
    # Check if there's variation in final performance
    throughput_std = np.std(final_throughputs)
    print(f"\nStandard deviation of final throughputs: {throughput_std:.3f}")
    
    if throughput_std < 0.01:
        print(f"\n⚠️  WARNING: Very low variation in final performance")
        print(f"   This might indicate runs are too similar")
    else:
        print(f"  ✓ Runs show variation in final performance")
    
    # Check 4: Verify episode-by-episode differences
    print(f"\nEpisode-by-episode analysis (first 10 episodes):")
    for ep in range(min(10, test_episodes)):
        ep_successes = successes[:, ep]
        unique_count = len(np.unique(ep_successes))
        success_rate = np.mean(ep_successes)
        
        if ep < 5:  # Print first 5 episodes
            print(f"  Episode {ep+1}: {ep_successes} -> {unique_count} unique outcomes, {success_rate*100:.0f}% success")
    
    # At least some episodes should have different outcomes across runs
    episodes_with_variation = 0
    for ep in range(test_episodes):
        if len(np.unique(successes[:, ep])) > 1:
            episodes_with_variation += 1
    
    variation_percentage = (episodes_with_variation / test_episodes) * 100
    print(f"\nEpisodes with varying outcomes: {episodes_with_variation}/{test_episodes} ({variation_percentage:.1f}%)")
    
    if variation_percentage < 30:
        print(f"\n❌ FAILED: Too few episodes show variation ({variation_percentage:.1f}%)")
        print(f"   Expected at least 30% of episodes to have different outcomes across runs")
        return False
    else:
        print(f"  ✓ Sufficient episode-level variation ({variation_percentage:.1f}%)")
    
    print(f"\n✅ PASSED: Runs are independent!")
    print(f"\nKey findings:")
    print(f"  • Runs use the same map (seed {config.RANDOM_SEED})")
    print(f"  • But produce different learning trajectories")
    print(f"  • Average {avg_difference:.1f}% difference between runs")
    print(f"  • {variation_percentage:.1f}% of episodes vary across runs")
    
    return True


def run_all_tests():
    """Run all pre-experiment validation tests."""
    print("\n" + "="*70)
    print("PRE-EXPERIMENT VALIDATION TEST SUITE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Random Seed: {config.RANDOM_SEED}")
    print(f"  Grid Size: {config.GRID_ROWS}x{config.GRID_COLS}")
    print(f"  Hole Density: {config.HOLE_DENSITY}")
    print(f"  Slippery: {config.IS_SLIPPERY}")
    print(f"  Success Rate: {config.SUCCESS_RATE}")
    print("="*70)
    
    results = {}
    
    # Test 1: Map Consistency
    results['map_consistency'] = test_map_consistency()
    
    # Test 2: Success Rate
    results['success_rate'] = test_success_rate()
    
    # Test 3: Reproducibility
    results['reproducibility'] = test_reproducibility()
    
    # Test 4: Run Independence
    results['run_independence'] = test_run_independence()
    
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
        print("🎉 ALL TESTS PASSED! Safe to run compare_methods.py")
    else:
        print("⚠️  SOME TESTS FAILED! Fix issues before running experiments")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
