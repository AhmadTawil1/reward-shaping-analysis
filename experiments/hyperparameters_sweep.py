"""
Hyperparameter Sensitivity Study: Epsilon and Alpha

This script performs sensitivity analysis on the two fundamental
RL hyperparameters: epsilon (exploration rate) and alpha (learning rate).

We test SARSA with Safety-Based shaping to understand how these
parameters affect learning dynamics and final performance.

Usage:
    python hyperparameter_sweep.py

Configuration:
    - Edit config.py to change sweep settings
    - Uses config.SWEEP_EPISODES and config.SWEEP_RUNS
    - Tests config.EPSILON_SWEEP and config.ALPHA_SWEEP values

Results:
    - Saves plots to results/ directory
    - Prints summary tables to console
    - Generates plots: 
      * epsilon_sensitivity_subplots.png (N subplots, one per epsilon value)
      * alpha_sensitivity_subplots.png (N subplots, one per alpha value)
      * epsilon_sensitivity_throughput.png (combined throughput comparison)
      * alpha_sensitivity_throughput.png (combined throughput comparison)

Author: Ahmad
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import config
from experiments.run_sarsa import run_sarsa

# Set random seed for reproducibility
np.random.seed(config.RANDOM_SEED)
print(f"🎲 Random seed set to: {config.RANDOM_SEED}")


def plot_epsilon_sensitivity_subplots(epsilon_dict, throughput_dict, lengths_dict, save_path):
    """
    Plot epsilon sensitivity with separate subplots for each epsilon value's returns.
    
    Args:
        epsilon_dict (dict): {epsilon_value: returns_array (runs, episodes)}
        throughput_dict (dict): {epsilon_value: throughput_array (runs, episodes)}
        lengths_dict (dict): {epsilon_value: lengths_array (runs, episodes)}
        save_path (str): Where to save the plot
    """
    sorted_items = sorted(epsilon_dict.items(), key=lambda x: x[0])
    num_values = len(sorted_items)
    
    # Create subplots - one for each epsilon value
    fig, axes = plt.subplots(num_values, 1, figsize=(14, 5 * num_values))
    
    # Handle case where there's only one epsilon value
    if num_values == 1:
        axes = [axes]
    
    # Plot each epsilon value in its own subplot
    for idx, (epsilon_value, data) in enumerate(sorted_items):
        ax = axes[idx]
        
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        n = data.shape[0]
        ci = 1.96 * std / np.sqrt(n)
        
        episodes = np.arange(1, data.shape[1] + 1)
        
        color = f'C{idx}'
        
        ax.plot(episodes, mean, linewidth=2.5, color=color, label=f'ε = {epsilon_value}')
        ax.fill_between(episodes, mean - ci, mean + ci, alpha=0.2, color=color)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Real Return', fontsize=12)
        ax.set_title(f'({chr(97+idx)}) Returns with ε = {epsilon_value} (α = 0.1 fixed)', 
                      fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('SARSA + Safety: Epsilon (ε) Sensitivity Study - Returns', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()


def plot_alpha_sensitivity_subplots(alpha_dict, throughput_dict, lengths_dict, save_path):
    """
    Plot alpha sensitivity with separate subplots for each alpha value's returns.
    
    Args:
        alpha_dict (dict): {alpha_value: returns_array (runs, episodes)}
        throughput_dict (dict): {alpha_value: throughput_array (runs, episodes)}
        lengths_dict (dict): {alpha_value: lengths_array (runs, episodes)}
        save_path (str): Where to save the plot
    """
    sorted_items = sorted(alpha_dict.items(), key=lambda x: x[0])
    num_values = len(sorted_items)
    
    # Create subplots - one for each alpha value
    fig, axes = plt.subplots(num_values, 1, figsize=(14, 5 * num_values))
    
    # Handle case where there's only one alpha value
    if num_values == 1:
        axes = [axes]
    
    # Plot each alpha value in its own subplot
    for idx, (alpha_value, data) in enumerate(sorted_items):
        ax = axes[idx]
        
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        n = data.shape[0]
        ci = 1.96 * std / np.sqrt(n)
        
        episodes = np.arange(1, data.shape[1] + 1)
        
        color = f'C{idx}'
        
        ax.plot(episodes, mean, linewidth=2.5, color=color, label=f'α = {alpha_value}')
        ax.fill_between(episodes, mean - ci, mean + ci, alpha=0.2, color=color)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Real Return', fontsize=12)
        ax.set_title(f'({chr(97+idx)}) Returns with α = {alpha_value} (ε = 0.1 fixed)', 
                      fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('SARSA + Safety: Alpha (α) Sensitivity Study - Returns', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()



def plot_sensitivity(results_dict, param_name, title, ylabel, save_path):
    """
    Plot learning curves for different parameter values with 95% CI.
    
    Args:
        results_dict (dict): {param_value: data_array (runs, episodes)}
        param_name (str): 'ε' or 'α' for axis labels
        title (str): Plot title
        ylabel (str): Y-axis label ('Real Return' or 'Throughput')
        save_path (str): Where to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    # Sort by parameter value for consistent colors
    sorted_items = sorted(results_dict.items(), key=lambda x: x[0])
    
    for idx, (param_value, data) in enumerate(sorted_items):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        n = data.shape[0]
        ci = 1.96 * std / np.sqrt(n)  # 95% confidence interval
        
        episodes = np.arange(1, data.shape[1] + 1)
        
        label = f'{param_name} = {param_value}'
        color = f'C{idx}'
        
        plt.plot(episodes, mean, label=label, linewidth=2.5, color=color)
        plt.fill_between(episodes, mean - ci, mean + ci, alpha=0.2, color=color)
    
    plt.xlabel('Episode', fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()


def print_sensitivity_table(results_dict, throughput_dict, param_name):
    """
    Print formatted summary table of sensitivity results.
    
    Args:
        results_dict (dict): {param_value: returns_array}
        throughput_dict (dict): {param_value: throughput_array}
        param_name (str): Parameter name for display
    """
    print("\n" + "="*90)
    print(f"{param_name.upper()} SENSITIVITY - RESULTS SUMMARY")
    print("="*90)
    print(f"{param_name:<15} | {'Throughput (last 100)':<25} | {'Avg Return (last 100)':<25}")
    print("-" * 90)
    
    # Calculate statistics
    stats = []
    for param_value in sorted(results_dict.keys()):
        returns = results_dict[param_value]
        throughput = throughput_dict[param_value]
        
        avg_throughput = np.mean(throughput[:, -100:])
        std_throughput = np.std(throughput[:, -100:])
        
        avg_return = np.mean(returns[:, -100:])
        std_return = np.std(returns[:, -100:])
        
        stats.append((param_value, avg_throughput, std_throughput, avg_return, std_return))
    
    # Print table
    for param_value, avg_tp, std_tp, avg_ret, std_ret in stats:
        marker = "⭐" if avg_tp == max(s[1] for s in stats) else "  "
        print(f"{marker} {param_value:<13.3f} | "
              f"{avg_tp*100:>7.2f}% ± {std_tp*100:>5.2f}% | "
              f"{avg_ret:>8.3f} ± {std_ret:>6.3f}")
    
    print("-" * 90)
    
    # Find best
    best_param = max(stats, key=lambda x: x[1])[0]
    best_throughput = max(s[1] for s in stats)
    print(f"\n🎯 Best {param_name}: {best_param:.3f}")
    print(f"   Throughput: {best_throughput*100:.2f}%")
    print("="*90)


def compute_throughput(successes):
    """
    Compute cumulative throughput from success array.
    
    Args:
        successes (np.ndarray): Binary array (runs, episodes)
    
    Returns:
        np.ndarray: Cumulative throughput (runs, episodes)
    """
    cumulative = np.cumsum(successes, axis=1)
    episodes = np.arange(1, successes.shape[1] + 1)
    return cumulative / episodes


def epsilon_sensitivity_study():
    """
    Study the effect of different epsilon values on learning.
    
    Epsilon controls the exploration-exploitation tradeoff:
    - Low epsilon (e.g., 0.05): More exploitation (greedy)
    - High epsilon (e.g., 0.2): More exploration (random)
    
    Returns:
        tuple: (returns_dict, throughput_dict)
    """
    print("\n" + "="*70)
    print("EPSILON SENSITIVITY STUDY")
    print("="*70)
    print(f"Testing epsilon values: {config.EPSILON_SWEEP}")
    print(f"Fixed alpha: {config.ALPHA_SWEEP[0]}")
    print(f"Algorithm: SARSA + Safety-Based Shaping")
    print(f"Episodes per run: {config.SWEEP_EPISODES}")
    print(f"Number of runs: {config.SWEEP_RUNS}")
    print(f"Grid: {config.GRID_ROWS}×{config.GRID_COLS}")
    print(f"Hole density: {config.HOLE_DENSITY}")
    print("="*70)
    
    results = {}
    throughput_results = {}
    lengths_results = {}
    
    for epsilon in tqdm(config.EPSILON_SWEEP, desc="Epsilon values", unit="value"):
        print(f"\n🔄 Running with ε = {epsilon}")
        
        successes, returns, lengths = run_sarsa(
            num_episodes=config.SWEEP_EPISODES,
            num_runs=config.SWEEP_RUNS,
            shaping_type='safety',
            grid_rows=config.GRID_ROWS,
            grid_cols=config.GRID_COLS,
            hole_density=config.HOLE_DENSITY,
            is_slippery=config.IS_SLIPPERY,
            epsilon=epsilon,            # VARY THIS
            alpha=config.ALPHA_SWEEP[0],  # FIXED (first sweep value)
            gamma=config.GAMMA,
            shaping_params={'beta': config.BETA_SAFETY_SARSA}
        )
        
        results[epsilon] = returns
        throughput_results[epsilon] = compute_throughput(successes)
        lengths_results[epsilon] = lengths
        
        # Print quick summary
        final_throughput = np.mean(throughput_results[epsilon][:, -100:])
        avg_return = np.mean(returns[:, -100:])
        print(f"   Final throughput: {final_throughput*100:.2f}%")
        print(f"   Avg return (last 100): {avg_return:.3f}")
    
    # Plot throughput (individual)
    plot_sensitivity(
        throughput_results,
        param_name='ε',
        title='SARSA + Safety: Epsilon Sensitivity Study (Throughput)',
        ylabel='Throughput',
        save_path=f'{config.RESULTS_DIR}/epsilon_sensitivity_throughput.png'
    )
    
    # Print summary table
    print_sensitivity_table(results, throughput_results, 'Epsilon (ε)')
    
    return results, throughput_results, lengths_results


def alpha_sensitivity_study():
    """
    Study the effect of different alpha values on learning.
    
    Alpha controls the learning rate:
    - Low alpha (e.g., 0.05): Slow, stable learning
    - High alpha (e.g., 0.2): Fast but potentially unstable learning
    
    Returns:
        tuple: (returns_dict, throughput_dict)
    """
    print("\n" + "="*70)
    print("ALPHA SENSITIVITY STUDY")
    print("="*70)
    print(f"Testing alpha values: {config.ALPHA_SWEEP}")
    print(f"Fixed epsilon: {config.EPSILON_SWEEP[0]}")
    print(f"Algorithm: SARSA + Safety-Based Shaping")
    print(f"Episodes per run: {config.SWEEP_EPISODES}")
    print(f"Number of runs: {config.SWEEP_RUNS}")
    print(f"Grid: {config.GRID_ROWS}×{config.GRID_COLS}")
    print(f"Hole density: {config.HOLE_DENSITY}")
    print("="*70)
    
    results = {}
    throughput_results = {}
    lengths_results = {}
    
    for alpha in tqdm(config.ALPHA_SWEEP, desc="Alpha values", unit="value"):
        print(f"\n🔄 Running with α = {alpha}")
        
        successes, returns, lengths = run_sarsa(
            num_episodes=config.SWEEP_EPISODES,
            num_runs=config.SWEEP_RUNS,
            shaping_type='safety',
            grid_rows=config.GRID_ROWS,
            grid_cols=config.GRID_COLS,
            hole_density=config.HOLE_DENSITY,
            is_slippery=config.IS_SLIPPERY,
            epsilon=config.EPSILON_SWEEP[0],  # FIXED (first sweep value)
            alpha=alpha,                # VARY THIS
            gamma=config.GAMMA,
            shaping_params={'beta': config.BETA_SAFETY_SARSA}
        )
        
        results[alpha] = returns
        throughput_results[alpha] = compute_throughput(successes)
        lengths_results[alpha] = lengths
        
        # Print quick summary
        final_throughput = np.mean(throughput_results[alpha][:, -100:])
        avg_return = np.mean(returns[:, -100:])
        print(f"   Final throughput: {final_throughput*100:.2f}%")
        print(f"   Avg return (last 100): {avg_return:.3f}")
    
    # Plot throughput (individual)
    plot_sensitivity(
        throughput_results,
        param_name='α',
        title='SARSA + Safety: Alpha Sensitivity Study (Throughput)',
        ylabel='Throughput',
        save_path=f'{config.RESULTS_DIR}/alpha_sensitivity_throughput.png'
    )
    
    # Print summary table
    print_sensitivity_table(results, throughput_results, 'Alpha (α)')
    
    return results, throughput_results, lengths_results


if __name__ == "__main__":
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("HYPERPARAMETER SENSITIVITY STUDY")
    print("="*70)
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"  Episodes per run: {config.SWEEP_EPISODES}")
    print(f"  Number of runs: {config.SWEEP_RUNS}")
    print(f"  Total episodes per value: {config.SWEEP_EPISODES * config.SWEEP_RUNS:,}")
    print(f"  Algorithm: SARSA + Safety-Based Shaping")
    print(f"  Grid: {config.GRID_ROWS}×{config.GRID_COLS}")
    print(f"  Hole Density: {config.HOLE_DENSITY}")
    print(f"  Slippery: {config.IS_SLIPPERY}")
    if config.IS_SLIPPERY:
        print(f"  Success Rate: {config.SUCCESS_RATE}")
    
    print(f"\n📊 This study will generate 4 plots:")
    print(f"  1. Epsilon sensitivity (3 subplots: returns, throughput, lengths)")
    print(f"  2. Alpha sensitivity (3 subplots: returns, throughput, lengths)")
    print(f"  3. Epsilon sensitivity (throughput only)")
    print(f"  4. Alpha sensitivity (throughput only)")
    
    # Estimate runtime
    total_experiments = (len(config.EPSILON_SWEEP) + len(config.ALPHA_SWEEP))
    total_episodes = total_experiments * config.SWEEP_RUNS * config.SWEEP_EPISODES
    estimated_time_min = total_experiments * 1
    estimated_time_max = total_experiments * 1.5
    
    print(f"\n⏱️  Estimated runtime: {estimated_time_min}-{estimated_time_max} minutes")
    print(f"   ({total_episodes:,} total episodes across all experiments)")
    
    print("\n" + "="*70)
    
    # Confirm before running
    print("\n⚠️  This will take some time. Make sure:")
    print("  ✓ You've edited config.py with desired settings")
    print("  ✓ You have enough disk space for results")
    print("  ✓ You're ready to wait for completion")
    
    input("\n👉 Press ENTER to start, or Ctrl+C to cancel...")
    
    # Run studies
    print("\n[1/2] Running epsilon sensitivity study...")
    epsilon_results, epsilon_throughput, epsilon_lengths = epsilon_sensitivity_study()
    
    print("\n[2/2] Running alpha sensitivity study...")
    alpha_results, alpha_throughput, alpha_lengths = alpha_sensitivity_study()
    
    # ========= EPSILON SENSITIVITY SUBPLOTS =========
    print("\n📊 Generating epsilon sensitivity figure (3 subplots)...")
    plot_epsilon_sensitivity_subplots(
        epsilon_results,
        epsilon_throughput,
        epsilon_lengths,
        save_path=f'{config.RESULTS_DIR}/epsilon_sensitivity_subplots.png'
    )
    
    # ========= ALPHA SENSITIVITY SUBPLOTS =========
    print("\n📊 Generating alpha sensitivity figure (3 subplots)...")
    plot_alpha_sensitivity_subplots(
        alpha_results,
        alpha_throughput,
        alpha_lengths,
        save_path=f'{config.RESULTS_DIR}/alpha_sensitivity_subplots.png'
    )
    
    # Final summary
    print("\n" + "="*70)
    print("✅ SENSITIVITY STUDIES COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to '{config.RESULTS_DIR}/':")
    print("  📈 epsilon_sensitivity_subplots.png     ⭐ (3 subplots: returns, throughput, lengths)")
    print("  📈 alpha_sensitivity_subplots.png       ⭐ (3 subplots: returns, throughput, lengths)")
    print("  📈 epsilon_sensitivity_throughput.png   (throughput only)")
    print("  📈 alpha_sensitivity_throughput.png     (throughput only)")
    
    # Best values summary
    print("\n" + "="*70)
    print("🏆 OPTIMAL HYPERPARAMETERS FOUND")
    print("="*70)
    
    best_epsilon = max(epsilon_throughput.items(), 
                      key=lambda x: np.mean(x[1][:, -100:]))[0]
    best_epsilon_throughput = np.mean(epsilon_throughput[best_epsilon][:, -100:])
    
    best_alpha = max(alpha_throughput.items(), 
                    key=lambda x: np.mean(x[1][:, -100:]))[0]
    best_alpha_throughput = np.mean(alpha_throughput[best_alpha][:, -100:])
    
    print(f"\n  Best ε: {best_epsilon:.3f} (throughput: {best_epsilon_throughput*100:.2f}%)")
    print(f"  Best α: {best_alpha:.3f} (throughput: {best_alpha_throughput*100:.2f}%)")
    
    if best_epsilon == config.EPSILON_SWEEP[0] and best_alpha == config.ALPHA_SWEEP[0]:
        print("\n  ✅ Your default config values are optimal!")
    else:
        print(f"\n  💡 Consider updating config.py:")
        print(f"     EPSILON = {best_epsilon}")
        print(f"     ALPHA = {best_alpha}")
    
    print("\n" + "="*70)
    print("📝 NEXT STEPS")
    print("="*70)
    print("  1. Review the epsilon and alpha 3-subplot figures")
    print("  2. Include epsilon_sensitivity_subplots.png in your report")
    print("  3. Include alpha_sensitivity_subplots.png in your report")
    print("  4. Write analysis section discussing:")
    print("     • Why certain values work better")
    print("     • Connection to exploration-exploitation")
    print("     • Impact on learning dynamics")
    print("  5. Prepare to explain plots in presentation")
    print("="*70)