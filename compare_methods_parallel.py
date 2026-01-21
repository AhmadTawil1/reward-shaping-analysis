"""
Comprehensive Reward Shaping Comparison - PARALLEL EXECUTION VERSION

This script compares all reward shaping methods (Baseline, Step-Cost, Potential-Based,
Safety-Based, and Exploration Bonus) for both Monte Carlo and SARSA algorithms using
PARALLEL execution for maximum performance.

=== WHAT THIS SCRIPT DOES ===

1. MONTE CARLO COMPARISON (Parallel):
   - Runs 5 reward shaping methods
   - Each method: 20 independent runs × 10,000 episodes
   - Generates throughput and returns plots

2. SARSA COMPARISON (Parallel):
   - Runs 5 reward shaping methods
   - Each method: 20 independent runs × 10,000 episodes
   - Generates throughput and returns plots

3. FINAL COMPARISON:
   - Compares best MC method vs best SARSA method
   - Generates comparison plots

4. POLICY VISUALIZATION:
   - Visualizes learned policies for best methods
   - Shows action arrows and Q-values on grid

=== HOW PARALLELIZATION WORKS ===

- For each method: 20 runs execute SIMULTANEOUSLY across CPU cores
- Uses joblib.Parallel with n_jobs=-1 (all available cores)
- Each run is completely independent (different random seed)
- Results are aggregated after all runs complete
- Expected speedup: 6-9x faster than sequential execution

=== OUTPUTS ===

PLOTS GENERATED:
  • mc_all_methods_throughput_parallel.png
  • mc_all_methods_returns_parallel.png
  • sarsa_all_methods_throughput_parallel.png
  • sarsa_all_methods_returns_parallel.png
  • final_mc_vs_sarsa_throughput_parallel.png
  • final_mc_vs_sarsa_returns_parallel.png
  • best_policies_comparison_parallel.png

CONSOLE OUTPUT:
  • Summary tables for each algorithm
  • Final throughput and return statistics
  • Best method identification
  • Total execution time

=== THROUGHPUT CALCULATION ===

Throughput = Cumulative success rate over episodes
Formula: throughput[t] = (total_successes_up_to_t) / t

For each episode:
  1. Calculate throughput for each of 20 runs
  2. Compute mean across runs (solid line in plot)
  3. Compute 95% CI: mean ± 1.96 × (std / √20)
  4. Plot mean line with shaded confidence interval

=== STATISTICAL VALIDITY ===

- 20 independent runs per method ensures statistical robustness
- 95% confidence intervals show reliability of results
- Non-overlapping CIs indicate statistically significant differences
- All experiments use same environment map (controlled by RANDOM_SEED)
- Each run uses different agent seed for independence

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
import os
from tqdm import tqdm

# Import configuration
import config

from experiments.run_monte_carlo_parallel import run_monte_carlo_parallel
from experiments.run_sarsa_parallel import run_sarsa_parallel


def compute_throughput(successes):
    """Compute cumulative throughput from success array."""
    cumulative = np.cumsum(successes, axis=1)
    episodes = np.arange(1, successes.shape[1] + 1)
    return cumulative / episodes


def print_results_table(results, algorithm_name):
    """
    Print a formatted table of results for all methods.
    
    Args:
        results (dict): Dictionary with method names as keys
        algorithm_name (str): 'Monte Carlo' or 'SARSA'
    """
    print("\n" + "="*80)
    print(f"{algorithm_name.upper()} - RESULTS SUMMARY (PARALLEL)")
    print("="*80)
    
    # Header
    print(f"\n{'Method':<35} | {'Final Throughput':<20} | {'Avg Return':<15}")
    print("-" * 80)
    
    # Collect data for sorting
    method_stats = []
    
    for label, data in results.items():
        # Final throughput (at last episode, averaged across runs)
        final_throughput = np.mean(data['throughput'][:, -1])
        final_throughput_std = np.std(data['throughput'][:, -1])
        
        # Average return (at last episode, averaged across runs)
        avg_return = np.mean(data['returns'][:, -1])
        avg_return_std = np.std(data['returns'][:, -1])
        
        method_stats.append({
            'label': label,
            'throughput': final_throughput,
            'throughput_std': final_throughput_std,
            'return': avg_return,
            'return_std': avg_return_std
        })
    
    # Sort by throughput (descending)
    method_stats.sort(key=lambda x: x['throughput'], reverse=True)
    
    # Print sorted results
    for i, stats in enumerate(method_stats):
        marker = "⭐" if i == 0 else "  "
        print(f"{marker} {stats['label']:<33} | "
              f"{stats['throughput']*100:>6.2f}% ± {stats['throughput_std']*100:>4.2f}% | "
              f"{stats['return']:>6.3f} ± {stats['return_std']:>5.3f}")
    
    print("-" * 80)
    
    # Calculate improvement over baseline
    baseline_throughput = None
    best_throughput = method_stats[0]['throughput']
    
    for stats in method_stats:
        if 'Baseline' in stats['label']:
            baseline_throughput = stats['throughput']
            break
    
    if baseline_throughput and baseline_throughput > 0:
        improvement = ((best_throughput - baseline_throughput) / baseline_throughput) * 100
        print(f"\n🎯 Best method: {method_stats[0]['label']}")
        print(f"   Improvement over baseline: {improvement:+.1f}%")
        print(f"   ({baseline_throughput*100:.2f}% → {best_throughput*100:.2f}%)")
    
    print("="*80 + "\n")


def plot_comparison(results_dict, metric='throughput', title='Comparison', 
                   ylabel='Throughput', save_path=None):
    """Plot comparison of multiple methods with 95% CI."""
    plt.figure(figsize=(12, 7))
    
    for label, data in results_dict.items():
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        n = data.shape[0]
        ci = 1.96 * std / np.sqrt(n)
        
        episodes = np.arange(1, data.shape[1] + 1)
        
        plt.plot(episodes, mean, label=label, linewidth=2.5)
        plt.fill_between(episodes, mean - ci, mean + ci, alpha=0.2)
    
    plt.xlabel('Episode', fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def plot_returns_subplots(results_dict, title_prefix='', save_path=None):
    """Plot returns for each method in separate subplots.
    
    Args:
        results_dict: Dictionary of method names to data arrays
        title_prefix: Prefix for the overall figure title (e.g., 'Monte Carlo' or 'SARSA')
        save_path: Path to save the figure
    """
    num_methods = len(results_dict)
    
    # Create subplots - arrange in a grid
    if num_methods <= 3:
        rows, cols = num_methods, 1
        figsize = (14, 5 * num_methods)
    else:
        rows = (num_methods + 1) // 2  # Round up
        cols = 2
        figsize = (16, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes array for easier iteration
    if num_methods == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (label, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        n = data.shape[0]
        ci = 1.96 * std / np.sqrt(n)
        
        episodes = np.arange(1, data.shape[1] + 1)
        
        ax.plot(episodes, mean, label=label, linewidth=2.5, color=f'C{idx}')
        ax.fill_between(episodes, mean - ci, mean + ci, alpha=0.2, color=f'C{idx}')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Real Return', fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for idx in range(num_methods, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'{title_prefix}: Real Returns Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def compare_all_mc_methods_parallel():
    """Compare all reward shaping methods for Monte Carlo using parallel execution."""
    print("\n" + "="*70)
    print("COMPARING ALL MONTE CARLO METHODS (PARALLEL)")
    print("="*70)
    
    # Get config
    cfg = config.get_final_config() if not config.TEST_MODE else config.get_test_config()
    
    methods = {
        'Baseline (No Shaping)': None,
        'Step-Cost Shaping': 'step',
        'Potential-Based Shaping': 'potential',
        'Safety-Based Shaping (Custom)': 'safety',
        'Exploration Bonus (Custom)': 'exploration'
    }
    
    results = {}
    
    # Use tqdm for progress bar over methods
    for label, shaping_type in tqdm(methods.items(), desc="MC Methods", unit="method"):
        print(f"\n🔄 Running: {label}")
        
        # Get optimized hyperparameters for this shaping method
        hyperparams = config.get_optimized_hyperparams('MC', shaping_type)
        
        # Run with parallel execution - get best agent for visualization
        successes, returns, lengths, best_agent, best_env = run_monte_carlo_parallel(
            num_episodes=cfg['num_episodes'],
            num_runs=cfg['num_runs'],
            shaping_type=shaping_type,
            grid_rows=cfg['grid_rows'],
            grid_cols=cfg['grid_cols'],
            hole_density=cfg['hole_density'],
            is_slippery=cfg['is_slippery'],
            success_rate=cfg['success_rate'],
            epsilon=hyperparams['epsilon'],  # Optimized per shaping method
            alpha=hyperparams['alpha'],      # Optimized per shaping method
            gamma=cfg['gamma'],
            initial_q_value=config.INITIAL_Q_VALUE,
            shaping_params=config.get_shaping_params(shaping_type, algorithm='MC'),
            return_best_agent=True,  # Get the trained agent for visualization
            n_jobs=-1  # Use all CPU cores
        )
        results[label] = {
            'successes': successes,
            'returns': returns,
            'lengths': lengths,
            'throughput': compute_throughput(successes),
            'agent': best_agent,  # Store the trained agent
            'env': best_env       # Store the environment
        }
    
    # Print results table
    print_results_table(results, "Monte Carlo")
    
    # Plot throughput comparison
    throughput_data = {label: data['throughput'] for label, data in results.items()}
    plot_comparison(
        throughput_data,
        title='Monte Carlo (PARALLEL): All Shaping Methods - Throughput Comparison',
        ylabel='Throughput',
        save_path=f'{config.RESULTS_DIR}/mc_all_methods_throughput_parallel.png'
    )
    
    # Plot returns comparison as subplots
    returns_data = {label: data['returns'] for label, data in results.items()}
    plot_returns_subplots(
        returns_data,
        title_prefix='Monte Carlo (PARALLEL): All Shaping Methods',
        save_path=f'{config.RESULTS_DIR}/mc_all_methods_returns_parallel.png'
    )
    
    return results


def compare_all_sarsa_methods_parallel():
    """Compare all reward shaping methods for SARSA using parallel execution."""
    print("\n" + "="*70)
    print("COMPARING ALL SARSA METHODS (PARALLEL)")
    print("="*70)
    
    # Get config
    cfg = config.get_final_config() if not config.TEST_MODE else config.get_test_config()
    
    methods = {
        'Baseline (No Shaping)': None,
        'Step-Cost Shaping': 'step',
        'Potential-Based Shaping': 'potential',
        'Safety-Based Shaping (Custom)': 'safety',
        'Exploration Bonus (Custom)': 'exploration'
    }
    
    results = {}
    
    # Use tqdm for progress bar over methods
    for label, shaping_type in tqdm(methods.items(), desc="SARSA Methods", unit="method"):
        print(f"\n🔄 Running: {label}")
        
        # Get optimized hyperparameters for this shaping method
        hyperparams = config.get_optimized_hyperparams('SARSA', shaping_type)
        
        # Run with parallel execution - get best agent for visualization
        successes, returns, lengths, best_agent, best_env = run_sarsa_parallel(
            num_episodes=cfg['num_episodes'],
            num_runs=cfg['num_runs'],
            shaping_type=shaping_type,
            grid_rows=cfg['grid_rows'],
            grid_cols=cfg['grid_cols'],
            hole_density=cfg['hole_density'],
            is_slippery=cfg['is_slippery'],
            success_rate=cfg['success_rate'],
            epsilon=hyperparams['epsilon'],  # Optimized per shaping method
            alpha=hyperparams['alpha'],      # Optimized per shaping method
            gamma=cfg['gamma'],
            initial_q_value=config.INITIAL_Q_VALUE,
            shaping_params=config.get_shaping_params(shaping_type, algorithm='SARSA'),
            return_best_agent=True,  # Get the trained agent for visualization
            n_jobs=-1  # Use all CPU cores
        )
        results[label] = {
            'successes': successes,
            'returns': returns,
            'lengths': lengths,
            'throughput': compute_throughput(successes),
            'agent': best_agent,  # Store the trained agent
            'env': best_env       # Store the environment
        }
    
    # Print results table
    print_results_table(results, "SARSA")
    
    # Plot throughput comparison
    throughput_data = {label: data['throughput'] for label, data in results.items()}
    plot_comparison(
        throughput_data,
        title='SARSA (PARALLEL): All Shaping Methods - Throughput Comparison',
        ylabel='Throughput',
        save_path=f'{config.RESULTS_DIR}/sarsa_all_methods_throughput_parallel.png'
    )
    
    # Plot returns comparison as subplots
    returns_data = {label: data['returns'] for label, data in results.items()}
    plot_returns_subplots(
        returns_data,
        title_prefix='SARSA (PARALLEL): All Shaping Methods',
        save_path=f'{config.RESULTS_DIR}/sarsa_all_methods_returns_parallel.png'
    )
    
    return results


def get_best_method(results):
    """
    Determine the best method from results based on throughput.
    
    Args:
        results (dict): Dictionary with method names as keys and data as values
        
    Returns:
        tuple: (best_method_label, throughput)
    """
    best_throughput = -1
    best_label = None
    
    for label, data in results.items():
        final_throughput = np.mean(data['throughput'][:, -1])
        if final_throughput > best_throughput:
            best_throughput = final_throughput
            best_label = label
    
    return best_label, best_throughput


def compare_mc_vs_sarsa_best(mc_results, sarsa_results):
    """
    Compare best MC method vs best SARSA method.
    
    Args:
        mc_results (dict): Results from all MC methods
        sarsa_results (dict): Results from all SARSA methods
    """
    print("\n" + "="*70)
    print("FINAL COMPARISON: BEST MC VS BEST SARSA (PARALLEL)")
    print("="*70)
    
    # Get best methods
    best_mc_label, mc_throughput = get_best_method(mc_results)
    best_sarsa_label, sarsa_throughput = get_best_method(sarsa_results)
    
    print(f"\n🏆 Best MC method: {best_mc_label}")
    print(f"🏆 Best SARSA method: {best_sarsa_label}")
    
    # Extract data for best methods
    mc_data = mc_results[best_mc_label]
    sarsa_data = sarsa_results[best_sarsa_label]
    
    mc_returns = mc_data['returns']
    mc_throughput_array = mc_data['throughput']
    
    sarsa_returns = sarsa_data['returns']
    sarsa_throughput_array = sarsa_data['throughput']
    
    # Print comparison table
    print("\n" + "="*80)
    print("BEST MC VS BEST SARSA - FINAL COMPARISON")
    print("="*80)
    
    mc_throughput_mean = np.mean(mc_throughput_array[:, -1])
    mc_throughput_std = np.std(mc_throughput_array[:, -1])
    sarsa_throughput_mean = np.mean(sarsa_throughput_array[:, -1])
    sarsa_throughput_std = np.std(sarsa_throughput_array[:, -1])
    
    mc_return = np.mean(mc_returns[:, -1])
    mc_return_std = np.std(mc_returns[:, -1])
    sarsa_return = np.mean(sarsa_returns[:, -1])
    sarsa_return_std = np.std(sarsa_returns[:, -1])
    
    print(f"\n{'Algorithm':<35} | {'Final Throughput':<20} | {'Avg Return':<15}")
    print("-" * 80)
    
    # Determine winner marker
    mc_marker = "🏆 " if mc_throughput_mean > sarsa_throughput_mean else "   "
    sarsa_marker = "🏆 " if sarsa_throughput_mean > mc_throughput_mean else "   "
    
    print(f"{mc_marker}{best_mc_label:<33} | "
          f"{mc_throughput_mean*100:>6.2f}% ± {mc_throughput_std*100:>4.2f}% | "
          f"{mc_return:>6.3f} ± {mc_return_std:>5.3f}")
    print(f"{sarsa_marker}{best_sarsa_label:<33} | "
          f"{sarsa_throughput_mean*100:>6.2f}% ± {sarsa_throughput_std*100:>4.2f}% | "
          f"{sarsa_return:>6.3f} ± {sarsa_return_std:>5.3f}")
    print("-" * 80)
    
    # Calculate performance difference
    if sarsa_throughput_mean > mc_throughput_mean:
        diff = ((sarsa_throughput_mean - mc_throughput_mean) / mc_throughput_mean * 100) if mc_throughput_mean > 0 else float('inf')
        print(f"\n🏆 SARSA outperforms MC by {diff:+.1f}% ({sarsa_throughput_mean*100:.2f}% vs {mc_throughput_mean*100:.2f}%)")
    elif mc_throughput_mean > sarsa_throughput_mean:
        diff = ((mc_throughput_mean - sarsa_throughput_mean) / sarsa_throughput_mean * 100) if sarsa_throughput_mean > 0 else float('inf')
        print(f"\n🏆 MC outperforms SARSA by {diff:+.1f}% ({mc_throughput_mean*100:.2f}% vs {sarsa_throughput_mean*100:.2f}%)")
    else:
        print(f"\n⚖️  MC and SARSA perform equally ({mc_throughput_mean*100:.2f}%)")
    
    print("="*80 + "\n")
    
    # Plot throughput comparison
    print("📊 Generating throughput comparison plot...")
    throughput_data = {
        f'MC: {best_mc_label}': mc_throughput_array,
        f'SARSA: {best_sarsa_label}': sarsa_throughput_array
    }
    plot_comparison(
        throughput_data,
        title='Final Comparison (PARALLEL): Best MC vs Best SARSA - Throughput',
        ylabel='Throughput',
        save_path=f'{config.RESULTS_DIR}/final_mc_vs_sarsa_throughput_parallel.png'
    )
    
    # Plot returns comparison
    print("📊 Generating returns comparison plot...")
    returns_data = {
        f'MC: {best_mc_label}': mc_returns,
        f'SARSA: {best_sarsa_label}': sarsa_returns
    }
    plot_comparison(
        returns_data,
        title='Final Comparison (PARALLEL): Best MC vs Best SARSA - Returns',
        ylabel='Real Return',
        save_path=f'{config.RESULTS_DIR}/final_mc_vs_sarsa_returns_parallel.png'
    )


def visualize_optimal_policies(mc_results, sarsa_results):
    """
    Visualize the learned policies for the best MC and SARSA methods side-by-side.
    
    Args:
        mc_results (dict): Results from all MC methods (including agents)
        sarsa_results (dict): Results from all SARSA methods (including agents)
    """
    print("\n📊 Generating side-by-side policy visualization...")
    
    # Get best methods
    best_mc_label, _ = get_best_method(mc_results)
    best_sarsa_label, _ = get_best_method(sarsa_results)
    
    # Get agents and environments
    mc_agent = mc_results[best_mc_label]['agent']
    mc_env = mc_results[best_mc_label]['env']
    sarsa_agent = sarsa_results[best_sarsa_label]['agent']
    sarsa_env = sarsa_results[best_sarsa_label]['env']
    
    # Unwrap environments
    unwrapped_mc_env = mc_env
    while hasattr(unwrapped_mc_env, 'env'):
        unwrapped_mc_env = unwrapped_mc_env.env
    
    desc = unwrapped_mc_env.desc
    rows, cols = desc.shape
    
    # Action mapping
    action_arrows = {
        0: (-0.25, 0),
        1: (0, -0.25),
        2: (0.25, 0),
        3: (0, 0.25)
    }
    
    action_names = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑'
    }
    
    # Create side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(cols * 3, rows * 1.5))
    
    # Function to draw policy on an axis
    def draw_policy(ax, agent, title):
        policy = agent.get_policy()
        value_func = agent.get_value_function()
        
        # Normalize values
        if value_func:
            max_val = max(value_func.values())
            min_val = min(value_func.values())
            val_range = max_val - min_val if max_val != min_val else 1.0
        else:
            max_val, min_val, val_range = 0, 0, 1.0
        
        # Draw grid
        for i in range(rows):
            for j in range(cols):
                state = i * cols + j
                cell_type = desc[i, j].decode('utf-8') if isinstance(desc[i, j], bytes) else desc[i, j]
                
                # Determine cell color
                if cell_type == 'S':
                    color = 'lightgreen'
                    label = 'S'
                elif cell_type == 'G':
                    color = 'gold'
                    label = 'G'
                elif cell_type == 'H':
                    color = 'lightcoral'
                    label = 'H'
                else:
                    if state in value_func:
                        normalized_val = (value_func[state] - min_val) / val_range
                        color = plt.cm.Blues(0.3 + 0.5 * normalized_val)
                    else:
                        color = 'lightblue'
                    label = ''
                
                # Draw cell
                rect = plt.Rectangle((j, rows - i - 1), 1, 1,
                                    facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                
                # Add cell label at top
                if label:
                    ax.text(j + 0.5, rows - i - 0.2, label,
                           ha='center', va='center', fontsize=20, fontweight='bold')
                
                # Add Q-value at bottom
                if state in value_func and cell_type not in ['H']:
                    q_value = value_func[state]
                    ax.text(j + 0.5, rows - i - 0.85, f'{q_value:.2f}',
                           ha='center', va='center', fontsize=11,
                           fontweight='bold', color='darkblue',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor='none', alpha=0.7))
                
                # Draw policy arrow
                if cell_type not in ['H', 'G'] and state in policy:
                    action = policy[state]
                    dx, dy = action_arrows[action]
                    
                    arrow = FancyArrow(j + 0.5, rows - i - 0.4, dx, dy,
                                      width=0.1, head_width=0.2, head_length=0.1,
                                      fc='darkred', ec='darkred', linewidth=1.5)
                    ax.add_patch(arrow)
                    
                    ax.text(j + 0.5 + dx * 1.15, rows - i - 0.4 + dy * 1.15,
                           action_names[action],
                           ha='center', va='center', fontsize=14,
                           fontweight='bold', color='darkred')
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Draw both policies
    print(f"\n🎯 Best MC: {best_mc_label}")
    draw_policy(ax1, mc_agent, f'Monte Carlo\n{best_mc_label}')
    
    print(f"🎯 Best SARSA: {best_sarsa_label}")
    draw_policy(ax2, sarsa_agent, f'SARSA\n{best_sarsa_label}')
    
    # Add overall title
    fig.suptitle('Best Policies Comparison (PARALLEL)', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    save_path = f'{config.RESULTS_DIR}/best_policies_comparison_parallel.png'
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved side-by-side policy visualization: {save_path}")
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    if config.RANDOM_SEED is not None:
        np.random.seed(config.RANDOM_SEED)
        print(f"\n🔒 Random seed set to {config.RANDOM_SEED} for reproducibility")
    
    print("\n" + "="*70)
    print("COMPREHENSIVE REWARD SHAPING COMPARISON (PARALLEL)")
    print("="*70)
    
    # Print current configuration
    config.print_config()
    
    print("\nThis script will:")
    print("  1. Compare all MC shaping methods (PARALLEL)")
    print("  2. Compare all SARSA shaping methods (PARALLEL)")
    print("  3. Generate final comparison (Best MC vs Best SARSA)")
    print("  4. Visualize optimal policies")
    print("\nFeatures:")
    print("  ✅ Parallel execution for maximum speed")
    print("  ✅ Progress tracking")
    print("  ✅ Summary tables after each algorithm")
    print("  ✅ Automatic best method detection")
    print("  ✅ Policy visualization (action arrows on grid)")
    print("  ✅ All plots and visualizations")
    print("="*70)
    
    import time
    start_time = time.time()
    
    # Compare all MC methods
    print("\n[1/4] Comparing all Monte Carlo methods (PARALLEL)...")
    mc_results = compare_all_mc_methods_parallel()
    
    # Compare all SARSA methods
    print("\n[2/4] Comparing all SARSA methods (PARALLEL)...")
    sarsa_results = compare_all_sarsa_methods_parallel()
    
    # Generate final comparison
    print("\n[3/4] Generating final comparison (Best MC vs Best SARSA)...")
    compare_mc_vs_sarsa_best(mc_results, sarsa_results)
    
    # Visualize learned policies
    print("\n[4/4] Generating policy visualizations...")
    visualize_optimal_policies(mc_results, sarsa_results)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*70)
    print("ALL COMPARISONS COMPLETED ✅")
    print("="*70)
    print(f"\n⏱️  Total execution time: {elapsed_time/60:.2f} minutes ({elapsed_time:.1f} seconds)")
    print(f"\nResults saved in '{config.RESULTS_DIR}/' directory.")
    print("\n📊 Review the summary tables above for key findings!")
    print("📈 Check the plots for detailed comparisons!")
    print("\nGenerated plots:")
    print("  • mc_all_methods_throughput_parallel.png")
    print("  • mc_all_methods_returns_parallel.png")
    print("  • sarsa_all_methods_throughput_parallel.png")
    print("  • sarsa_all_methods_returns_parallel.png")
    print("  • final_mc_vs_sarsa_throughput_parallel.png  ⭐")
    print("  • final_mc_vs_sarsa_returns_parallel.png     ⭐")
    print("\nGenerated policy visualizations:")
    print("  • best_policies_comparison_parallel.png      🎯")
    print("="*70)
