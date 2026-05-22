"""
Comprehensive Hyperparameter Tuning for All Methods

This script finds the best hyperparameters for each algorithm + shaping combination.

Strategy:
- 5000 episodes, 5 runs per configuration
- Tests all combinations of ε, α, and shaping-specific parameters
- Saves best hyperparameters to best_hyperparameters.json
- Uses these in final comparison

Settings:
- Episodes: 5000 (same as final experiments)
- Runs: 5 (vs 20 for final, for speed)
- This is a good balance between accuracy and speed

Expected Runtime: ~1-2 hours total

Usage:
    python find_best_hyperparameters.py
    
Output:
    best_hyperparameters.json - Use this in compare_methods.py

Author: Ahmad & Yosef
Date: January 2026
"""

import numpy as np
import json
import os
from tqdm import tqdm
import itertools

import config
from experiments.run_monte_carlo import run_monte_carlo
from experiments.run_sarsa import run_sarsa

# Tuning settings
TUNING_EPISODES = 5000  # Same as final experiments
TUNING_RUNS = 5         # Fewer runs for speed (vs 20 for final)

# Hyperparameter ranges to test
EPSILON_VALUES = [0.05, 0.1, 0.2]
ALPHA_VALUES = [0.05, 0.1, 0.2]

# Shaping-specific parameter ranges
STEP_COST_VALUES = [-0.005, -0.01, -0.02]
BETA_POTENTIAL_VALUES = [0.5, 1.0, 2.0]
BETA_SAFETY_VALUES = [0.05, 0.1, 0.2]
BETA_EXPLORATION_VALUES = [0.02, 0.05, 0.1]


def find_best_hyperparameters(algorithm, shaping_type, param_ranges):
    """
    Find best hyperparameters for a specific algorithm + shaping combination.
    
    Tests all combinations and returns the one with highest throughput.
    
    Args:
        algorithm (str): 'MC' or 'SARSA'
        shaping_type (str): None, 'step', 'potential', 'safety', 'exploration'
        param_ranges (dict): Dictionary of parameter names to lists of values
    
    Returns:
        dict: {
            'params': best hyperparameters found,
            'throughput': final throughput achieved,
            'all_results': list of all tested combinations with results
        }
    """
    print(f"\n{'='*70}")
    print(f"TUNING: {algorithm} + {shaping_type if shaping_type else 'Baseline'}")
    print(f"{'='*70}")
    
    # Generate all combinations to test
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    print(f"Testing {len(combinations)} combinations...")
    print(f"  Episodes: {TUNING_EPISODES}")
    print(f"  Runs: {TUNING_RUNS}")
    print(f"  Parameters: {', '.join(param_names)}")
    
    all_results = []
    best_params = {}
    best_throughput = -1
    
    # Test each combination
    for combo in tqdm(combinations, desc=f"{algorithm}+{shaping_type or 'baseline'}"):
        # Build parameter dict
        params = dict(zip(param_names, combo))
        
        # Extract algorithm parameters
        epsilon = params.get('epsilon', config.EPSILON)
        alpha = params.get('alpha', config.ALPHA)
        
        # Extract shaping parameters
        shaping_params = {}
        if 'step_cost' in params:
            shaping_params['step_cost'] = params['step_cost']
        if 'beta' in params:
            shaping_params['beta'] = params['beta']
        
        # Run experiment
        try:
            if algorithm == 'MC':
                successes, returns, lengths = run_monte_carlo(
                    num_episodes=TUNING_EPISODES,
                    num_runs=TUNING_RUNS,
                    shaping_type=shaping_type,
                    grid_rows=config.GRID_ROWS,
                    grid_cols=config.GRID_COLS,
                    hole_density=config.HOLE_DENSITY,
                    is_slippery=config.IS_SLIPPERY,
                    success_rate=config.SUCCESS_RATE,
                    epsilon=epsilon,
                    alpha=alpha,
                    gamma=config.GAMMA,
                    initial_q_value=config.INITIAL_Q_VALUE,
                    shaping_params=shaping_params if shaping_params else None
                )
            else:  # SARSA
                successes, returns, lengths = run_sarsa(
                    num_episodes=TUNING_EPISODES,
                    num_runs=TUNING_RUNS,
                    shaping_type=shaping_type,
                    grid_rows=config.GRID_ROWS,
                    grid_cols=config.GRID_COLS,
                    hole_density=config.HOLE_DENSITY,
                    is_slippery=config.IS_SLIPPERY,
                    success_rate=config.SUCCESS_RATE,
                    epsilon=epsilon,
                    alpha=alpha,
                    gamma=config.GAMMA,
                    initial_q_value=config.INITIAL_Q_VALUE,
                    shaping_params=shaping_params if shaping_params else None
                )
            
            # Compute final throughput (last 100 episodes, averaged across runs)
            throughput = np.mean(successes[:, -100:])
            
            # Store result
            result = {
                'params': params.copy(),
                'throughput': float(throughput),
                'avg_return': float(np.mean(returns[:, -100:])),
                'avg_length': float(np.mean(lengths[:, -100:]))
            }
            all_results.append(result)
            
            # Update best
            if throughput > best_throughput:
                best_throughput = throughput
                best_params = params.copy()
                
        except Exception as e:
            print(f"\n❌ Error with params {params}: {e}")
            continue
    
    # Sort results by throughput
    all_results.sort(key=lambda x: x['throughput'], reverse=True)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {algorithm} + {shaping_type if shaping_type else 'Baseline'}")
    print(f"{'='*70}")
    print(f"\n🏆 Best hyperparameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"\n📊 Performance:")
    print(f"   Throughput: {best_throughput*100:.2f}%")
    print(f"   Avg Return: {all_results[0]['avg_return']:.3f}")
    print(f"   Avg Length: {all_results[0]['avg_length']:.1f}")
    
    # Show top 3 configurations
    print(f"\n📈 Top 3 configurations:")
    for i, result in enumerate(all_results[:3], 1):
        print(f"\n   #{i}: Throughput = {result['throughput']*100:.2f}%")
        for param, value in result['params'].items():
            print(f"       {param}: {value}")
    
    return {
        'params': best_params,
        'throughput': float(best_throughput),
        'all_results': all_results
    }


def main():
    """Main function to run hyperparameter tuning for all methods."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE HYPERPARAMETER TUNING")
    print("="*70)
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"  Grid: {config.GRID_ROWS}×{config.GRID_COLS}")
    print(f"  Hole Density: {config.HOLE_DENSITY}")
    print(f"  Slippery: {config.IS_SLIPPERY}")
    print(f"  Success Rate: {config.SUCCESS_RATE}")
    print(f"  Episodes per test: {TUNING_EPISODES}")
    print(f"  Runs per test: {TUNING_RUNS}")
    
    # Calculate total experiments
    total_configs = (
        len(EPSILON_VALUES) * len(ALPHA_VALUES) * 2 +  # 2 baselines (MC, SARSA)
        len(EPSILON_VALUES) * len(ALPHA_VALUES) * len(STEP_COST_VALUES) * 2 +  # 2 step-cost
        len(EPSILON_VALUES) * len(ALPHA_VALUES) * len(BETA_POTENTIAL_VALUES) * 2 +  # 2 potential
        len(EPSILON_VALUES) * len(ALPHA_VALUES) * len(BETA_SAFETY_VALUES) * 2 +  # 2 safety
        len(EPSILON_VALUES) * len(ALPHA_VALUES) * len(BETA_EXPLORATION_VALUES) * 2  # 2 exploration
    )
    
    print(f"\n📊 This will test {total_configs} different configurations")
    print(f"   Total episodes: {total_configs * TUNING_RUNS * TUNING_EPISODES:,}")
    
    # Estimate runtime
    estimated_time_hours = total_configs * 1.5 / 60  # ~5 min per config
    print(f"\n⏱️  Estimated runtime: {estimated_time_hours:.1f} hours")
    print(f"   (This is a LOT - consider running overnight!)")
    
    print("\n" + "="*70)
    print("⚠️  WARNING: This will take 6-8 hours!")
    print("="*70)
    print("\nYou can:")
    print("  1. Run this overnight")
    print("  2. Run in background: nohup python find_best_hyperparameters.py &")
    print("  3. Use screen/tmux if on remote server")
    
    response = input("\n👉 Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Define experiments to run
    experiments = [
        # ===== MONTE CARLO =====
        ('MC', None, {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES
        }),
        ('MC', 'step', {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES,
            'step_cost': STEP_COST_VALUES
        }),
        ('MC', 'potential', {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES,
            'beta': BETA_POTENTIAL_VALUES
        }),
        ('MC', 'safety', {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES,
            'beta': BETA_SAFETY_VALUES
        }),
        ('MC', 'exploration', {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES,
            'beta': BETA_EXPLORATION_VALUES
        }),
        
        # ===== SARSA =====
        ('SARSA', None, {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES
        }),
        ('SARSA', 'step', {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES,
            'step_cost': STEP_COST_VALUES
        }),
        ('SARSA', 'potential', {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES,
            'beta': BETA_POTENTIAL_VALUES
        }),
        ('SARSA', 'safety', {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES,
            'beta': BETA_SAFETY_VALUES
        }),
        ('SARSA', 'exploration', {
            'epsilon': EPSILON_VALUES,
            'alpha': ALPHA_VALUES,
            'beta': BETA_EXPLORATION_VALUES
        }),
    ]
    
    # Run all experiments
    results = {}
    
    print("\n" + "="*70)
    print(f"STARTING TUNING: {len(experiments)} methods")
    print("="*70)
    
    for idx, (algorithm, shaping_type, param_ranges) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] Tuning {algorithm} + {shaping_type or 'baseline'}...")
        
        key = f"{algorithm}_{shaping_type if shaping_type else 'baseline'}"
        result = find_best_hyperparameters(algorithm, shaping_type, param_ranges)
        results[key] = result
    
    # Save results
    output_file = 'best_hyperparameters.json'
    
    # Prepare data for JSON (remove full results to keep file small)
    results_for_json = {}
    for key, data in results.items():
        results_for_json[key] = {
            'params': data['params'],
            'throughput': data['throughput']
        }
    
    with open(output_file, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    # Also save detailed results
    detailed_output_file = 'best_hyperparameters_detailed.json'
    with open(detailed_output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("✅ HYPERPARAMETER TUNING COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to:")
    print(f"   {output_file} (for use in compare_methods.py)")
    print(f"   {detailed_output_file} (detailed results)")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: BEST HYPERPARAMETERS FOR ALL METHODS")
    print("="*80)
    print(f"\n{'Method':<30} | {'ε':<6} | {'α':<6} | {'Other':<15} | {'Throughput'}")
    print("-" * 80)
    
    for key in sorted(results.keys()):
        data = results[key]
        params = data['params']
        throughput = data['throughput']
        
        epsilon = params.get('epsilon', '-')
        alpha = params.get('alpha', '-')
        
        # Format other parameters
        other = []
        if 'step_cost' in params:
            other.append(f"c={params['step_cost']}")
        if 'beta' in params:
            other.append(f"β={params['beta']}")
        other_str = ', '.join(other) if other else '-'
        
        print(f"{key:<30} | {epsilon:<6} | {alpha:<6} | {other_str:<15} | {throughput*100:>5.1f}%")
    
    print("-" * 80)
    
    # Find overall best
    best_method = max(results.items(), key=lambda x: x[1]['throughput'])
    print(f"\n🏆 Overall Best Method: {best_method[0]}")
    print(f"   Throughput: {best_method[1]['throughput']*100:.2f}%")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review the hyperparameters above")
    print("2. Modify compare_methods.py to load and use these values")
    print("3. Run final comparison with 20 runs:")
    print("   python compare_methods.py")
    print("="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    if config.RANDOM_SEED is not None:
        np.random.seed(config.RANDOM_SEED)
        print(f"\n🔒 Random seed set to {config.RANDOM_SEED} for reproducibility")
    
    main()