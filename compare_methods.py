"""
Compare all reward shaping methods on the same plots.

ENHANCED VERSION with:
- Progress bars for each run
- Summary tables after each algorithm
- Final comparison table

NOW USES config.py FOR ALL SETTINGS!
Just edit config.py to change parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Import configuration
import config

from experiments.run_monte_carlo import run_monte_carlo
from experiments.run_sarsa import run_sarsa


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
    print(f"{algorithm_name.upper()} - RESULTS SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Method':<35} | {'Final Throughput':<20} | {'Avg Return':<15}")
    print("-" * 80)
    
    # Collect data for sorting
    method_stats = []
    
    for label, data in results.items():
        # Final throughput (mean of last 100 episodes across all runs)
        final_throughput = np.mean(data['throughput'][:, -100:])
        final_throughput_std = np.std(data['throughput'][:, -100:])
        
        # Average return (mean of last 100 episodes across all runs)
        avg_return = np.mean(data['returns'][:, -100:])
        avg_return_std = np.std(data['returns'][:, -100:])
        
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


def get_best_method(results):
    """
    Determine the best method from results based on throughput.
    
    Args:
        results (dict): Dictionary with method names as keys and data as values
        
    Returns:
        tuple: (best_method_label, shaping_type)
    """
    best_throughput = -1
    best_label = None
    
    for label, data in results.items():
        final_throughput = np.mean(data['throughput'][:, -100:])
        if final_throughput > best_throughput:
            best_throughput = final_throughput
            best_label = label
    
    # Extract shaping type from label
    # Labels are like: 'Baseline (No Shaping)', 'Step-Cost Shaping', etc.
    shaping_map = {
        'Baseline (No Shaping)': None,
        'Step-Cost Shaping': 'step',
        'Potential-Based Shaping': 'potential',
        'Safety-Based Shaping (Custom)': 'safety',
        'Exploration Bonus (Custom)': 'exploration'
    }
    
    shaping_type = shaping_map.get(best_label, None)
    
    return best_label, shaping_type



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


def compare_all_mc_methods():
    """Compare all reward shaping methods for Monte Carlo."""
    print("\n" + "="*70)
    print("COMPARING ALL MONTE CARLO METHODS")
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
        successes, returns, lengths = run_monte_carlo(
            num_episodes=cfg['num_episodes'],
            num_runs=cfg['num_runs'],
            shaping_type=shaping_type,
            grid_rows=cfg['grid_rows'],
            grid_cols=cfg['grid_cols'],
            hole_density=cfg['hole_density'],
            is_slippery=cfg['is_slippery'],
            success_rate=cfg['success_rate'],
            epsilon=cfg['epsilon'],
            alpha=cfg['alpha'],
            gamma=cfg['gamma'],
            initial_q_value=config.INITIAL_Q_VALUE,
            shaping_params=config.get_shaping_params(shaping_type)
        )
        results[label] = {
            'successes': successes,
            'returns': returns,
            'lengths': lengths,
            'throughput': compute_throughput(successes)
        }
    
    # Print results table
    print_results_table(results, "Monte Carlo")
    
    # Plot throughput comparison
    throughput_data = {label: data['throughput'] for label, data in results.items()}
    plot_comparison(
        throughput_data,
        title='Monte Carlo: All Shaping Methods - Throughput Comparison',
        ylabel='Throughput',
        save_path=f'{config.RESULTS_DIR}/mc_all_methods_throughput.png'
    )
    
    # Plot returns comparison as subplots
    returns_data = {label: data['returns'] for label, data in results.items()}
    plot_returns_subplots(
        returns_data,
        title_prefix='Monte Carlo: All Shaping Methods',
        save_path=f'{config.RESULTS_DIR}/mc_all_methods_returns.png'
    )
    

    
    return results


def compare_all_sarsa_methods():
    """Compare all reward shaping methods for SARSA."""
    print("\n" + "="*70)
    print("COMPARING ALL SARSA METHODS")
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
        successes, returns, lengths = run_sarsa(
            num_episodes=cfg['num_episodes'],
            num_runs=cfg['num_runs'],
            shaping_type=shaping_type,
            grid_rows=cfg['grid_rows'],
            grid_cols=cfg['grid_cols'],
            hole_density=cfg['hole_density'],
            is_slippery=cfg['is_slippery'],
            success_rate=cfg['success_rate'],
            epsilon=cfg['epsilon'],
            alpha=cfg['alpha'],
            gamma=cfg['gamma'],
            initial_q_value=config.INITIAL_Q_VALUE,
            shaping_params=config.get_shaping_params(shaping_type)
        )
        results[label] = {
            'successes': successes,
            'returns': returns,
            'lengths': lengths,
            'throughput': compute_throughput(successes)
        }
    
    # Print results table
    print_results_table(results, "SARSA")
    
    # Plot throughput comparison
    throughput_data = {label: data['throughput'] for label, data in results.items()}
    plot_comparison(
        throughput_data,
        title='SARSA: All Shaping Methods - Throughput Comparison',
        ylabel='Throughput',
        save_path=f'{config.RESULTS_DIR}/sarsa_all_methods_throughput.png'
    )
    
    # Plot returns comparison as subplots
    returns_data = {label: data['returns'] for label, data in results.items()}
    plot_returns_subplots(
        returns_data,
        title_prefix='SARSA: All Shaping Methods',
        save_path=f'{config.RESULTS_DIR}/sarsa_all_methods_returns.png'
    )
    

    
    return results


def compare_mc_vs_sarsa_best_from_results(mc_results, sarsa_results):
    """
    Compare best MC method vs best SARSA method using already collected results.
    
    Args:
        mc_results (dict): Results from all MC methods
        sarsa_results (dict): Results from all SARSA methods
    """
    print("\n" + "="*70)
    print("FINAL COMPARISON: BEST MC VS BEST SARSA")
    print("="*70)
    
    # Get best methods
    best_mc_label, best_mc_shaping = get_best_method(mc_results)
    best_sarsa_label, best_sarsa_shaping = get_best_method(sarsa_results)
    
    print(f"\n🏆 Best MC method: {best_mc_label}")
    print(f"🏆 Best SARSA method: {best_sarsa_label}")
    
    # Extract data for best methods
    mc_data = mc_results[best_mc_label]
    sarsa_data = sarsa_results[best_sarsa_label]
    
    mc_successes = mc_data['successes']
    mc_returns = mc_data['returns']
    mc_throughput_array = mc_data['throughput']
    
    sarsa_successes = sarsa_data['successes']
    sarsa_returns = sarsa_data['returns']
    sarsa_throughput_array = sarsa_data['throughput']
    
    # Print comparison table
    print("\n" + "="*80)
    print("BEST MC VS BEST SARSA - FINAL COMPARISON")
    print("="*80)
    
    mc_throughput = np.mean(mc_throughput_array[:, -100:])
    mc_throughput_std = np.std(mc_throughput_array[:, -100:])
    sarsa_throughput = np.mean(sarsa_throughput_array[:, -100:])
    sarsa_throughput_std = np.std(sarsa_throughput_array[:, -100:])
    
    mc_return = np.mean(mc_returns[:, -100:])
    mc_return_std = np.std(mc_returns[:, -100:])
    sarsa_return = np.mean(sarsa_returns[:, -100:])
    sarsa_return_std = np.std(sarsa_returns[:, -100:])
    
    print(f"\n{'Algorithm':<35} | {'Final Throughput':<20} | {'Avg Return':<15}")
    print("-" * 80)
    
    # Determine winner marker
    mc_marker = "🏆 " if mc_throughput > sarsa_throughput else "   "
    sarsa_marker = "🏆 " if sarsa_throughput > mc_throughput else "   "
    
    print(f"{mc_marker}{best_mc_label:<33} | "
          f"{mc_throughput*100:>6.2f}% ± {mc_throughput_std*100:>4.2f}% | "
          f"{mc_return:>6.3f} ± {mc_return_std:>5.3f}")
    print(f"{sarsa_marker}{best_sarsa_label:<33} | "
          f"{sarsa_throughput*100:>6.2f}% ± {sarsa_throughput_std*100:>4.2f}% | "
          f"{sarsa_return:>6.3f} ± {sarsa_return_std:>5.3f}")
    print("-" * 80)
    
    # Calculate performance difference
    if sarsa_throughput > mc_throughput:
        diff = ((sarsa_throughput - mc_throughput) / mc_throughput * 100) if mc_throughput > 0 else float('inf')
        print(f"\n🏆 SARSA outperforms MC by {diff:+.1f}% ({sarsa_throughput*100:.2f}% vs {mc_throughput*100:.2f}%)")
    elif mc_throughput > sarsa_throughput:
        diff = ((mc_throughput - sarsa_throughput) / sarsa_throughput * 100) if sarsa_throughput > 0 else float('inf')
        print(f"\n🏆 MC outperforms SARSA by {diff:+.1f}% ({mc_throughput*100:.2f}% vs {sarsa_throughput*100:.2f}%)")
    else:
        print(f"\n⚖️  MC and SARSA perform equally ({mc_throughput*100:.2f}%)")
    
    print("="*80 + "\n")
    
    # Plot throughput comparison
    print("📊 Generating throughput comparison plot...")
    throughput_data = {
        f'MC: {best_mc_label}': mc_throughput_array,
        f'SARSA: {best_sarsa_label}': sarsa_throughput_array
    }
    plot_comparison(
        throughput_data,
        title='Final Comparison: Best MC vs Best SARSA - Throughput',
        ylabel='Throughput',
        save_path=f'{config.RESULTS_DIR}/final_mc_vs_sarsa_throughput.png'
    )
    
    # Plot returns comparison
    print("📊 Generating returns comparison plot...")
    returns_data = {
        f'MC: {best_mc_label}': mc_returns,
        f'SARSA: {best_sarsa_label}': sarsa_returns
    }
    plot_comparison(
        returns_data,
        title='Final Comparison: Best MC vs Best SARSA - Returns',
        ylabel='Real Return',
        save_path=f'{config.RESULTS_DIR}/final_mc_vs_sarsa_returns.png'
    )
    



if __name__ == "__main__":
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    if config.RANDOM_SEED is not None:
        np.random.seed(config.RANDOM_SEED)
        print(f"\n🔒 Random seed set to {config.RANDOM_SEED} for reproducibility")
    
    print("\n" + "="*70)
    print("COMPREHENSIVE REWARD SHAPING COMPARISON (ENHANCED)")
    print("="*70)
    
    # Print current configuration
    config.print_config()
    
    print("\nThis script will:")
    print("  1. Compare all MC shaping methods")
    print("  2. Compare all SARSA shaping methods")
    print("  3. Generate final comparison (Best MC vs Best SARSA)")
    print("\nFeatures:")
    print("  ✅ Progress bars for tracking")
    print("  ✅ Summary tables after each algorithm")
    print("  ✅ Automatic best method detection")
    print("  ✅ No redundant re-runs for final comparison")
    print("="*70)
    
    # Compare all MC methods
    print("\n[1/3] Comparing all Monte Carlo methods...")
    mc_results = compare_all_mc_methods()
    
    # Compare all SARSA methods
    print("\n[2/3] Comparing all SARSA methods...")
    sarsa_results = compare_all_sarsa_methods()
    
    # Generate final comparison using saved results
    print("\n[3/3] Generating final comparison (Best MC vs Best SARSA)...")
    compare_mc_vs_sarsa_best_from_results(mc_results, sarsa_results)
    
    print("\n" + "="*70)
    print("ALL COMPARISONS COMPLETED ✅")
    print("="*70)
    print(f"\nResults saved in '{config.RESULTS_DIR}/' directory.")
    print("\n📊 Review the summary tables above for key findings!")
    print("📈 Check the plots for detailed comparisons!")
    print("\nGenerated plots:")
    print("  • mc_all_methods_throughput.png")
    print("  • mc_all_methods_returns.png")
    print("  • sarsa_all_methods_throughput.png")
    print("  • sarsa_all_methods_returns.png")
    print("  • final_mc_vs_sarsa_throughput.png  ⭐")
    print("  • final_mc_vs_sarsa_returns.png     ⭐")
    print("="*70)