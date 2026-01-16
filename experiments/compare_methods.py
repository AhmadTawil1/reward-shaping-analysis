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


def plot_comparison(results_dict, metric='throughput', title='Comparison', 
                   ylabel='Throughput', save_path=None, subplot_ax=None):
    """Plot comparison of multiple methods with 95% CI.
    
    Args:
        results_dict: Dictionary of method names to data arrays
        metric: Metric being plotted (for reference)
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save the figure (only used if subplot_ax is None)
        subplot_ax: If provided, plot on this axis instead of creating new figure
    """
    if subplot_ax is None:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        standalone = True
    else:
        ax = subplot_ax
        standalone = False
    
    for label, data in results_dict.items():
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        n = data.shape[0]
        ci = 1.96 * std / np.sqrt(n)
        
        episodes = np.arange(1, data.shape[1] + 1)
        
        ax.plot(episodes, mean, label=label, linewidth=2.5)
        ax.fill_between(episodes, mean - ci, mean + ci, alpha=0.2)
    
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    if standalone:
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
            grid_size=cfg['grid_size'],
            hole_density=cfg['hole_density'],
            is_slippery=cfg['is_slippery'],
            epsilon=cfg['epsilon'],
            alpha=cfg['alpha'],
            gamma=cfg['gamma'],
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
    
    # Plot returns comparison
    returns_data = {label: data['returns'] for label, data in results.items()}
    plot_comparison(
        returns_data,
        title='Monte Carlo: All Shaping Methods - Real Returns Comparison',
        ylabel='Real Return',
        save_path=f'{config.RESULTS_DIR}/mc_all_methods_returns.png'
    )
    
    # Plot episode lengths comparison
    lengths_data = {label: data['lengths'] for label, data in results.items()}
    plot_comparison(
        lengths_data,
        title='Monte Carlo: All Shaping Methods - Episode Length Comparison',
        ylabel='Episode Length',
        save_path=f'{config.RESULTS_DIR}/mc_all_methods_lengths.png'
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
            grid_size=cfg['grid_size'],
            hole_density=cfg['hole_density'],
            is_slippery=cfg['is_slippery'],
            epsilon=cfg['epsilon'],
            alpha=cfg['alpha'],
            gamma=cfg['gamma'],
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
    
    # Plot returns comparison
    returns_data = {label: data['returns'] for label, data in results.items()}
    plot_comparison(
        returns_data,
        title='SARSA: All Shaping Methods - Real Returns Comparison',
        ylabel='Real Return',
        save_path=f'{config.RESULTS_DIR}/sarsa_all_methods_returns.png'
    )
    
    # Plot episode lengths comparison
    lengths_data = {label: data['lengths'] for label, data in results.items()}
    plot_comparison(
        lengths_data,
        title='SARSA: All Shaping Methods - Episode Length Comparison',
        ylabel='Episode Length',
        save_path=f'{config.RESULTS_DIR}/sarsa_all_methods_lengths.png'
    )
    
    return results


def compare_mc_vs_sarsa_best(mc_shaping='potential', sarsa_shaping='safety'):
    """Compare best MC method vs best SARSA method."""
    print("\n" + "="*70)
    print("COMPARING BEST MC VS BEST SARSA")
    print("="*70)
    
    # Get config
    cfg = config.get_final_config() if not config.TEST_MODE else config.get_test_config()
    
    # Run MC
    print(f"\n🔄 Running MC with {mc_shaping} shaping...")
    mc_successes, mc_returns, mc_lengths = run_monte_carlo(
        num_episodes=cfg['num_episodes'],
        num_runs=cfg['num_runs'],
        shaping_type=mc_shaping,
        grid_size=cfg['grid_size'],
        hole_density=cfg['hole_density'],
        is_slippery=cfg['is_slippery'],
        shaping_params=config.get_shaping_params(mc_shaping)
    )
    
    # Run SARSA
    print(f"\n🔄 Running SARSA with {sarsa_shaping} shaping...")
    sarsa_successes, sarsa_returns, sarsa_lengths = run_sarsa(
        num_episodes=cfg['num_episodes'],
        num_runs=cfg['num_runs'],
        shaping_type=sarsa_shaping,
        grid_size=cfg['grid_size'],
        hole_density=cfg['hole_density'],
        is_slippery=cfg['is_slippery'],
        shaping_params=config.get_shaping_params(sarsa_shaping)
    )
    
    # Print comparison table
    print("\n" + "="*80)
    print("BEST MC VS BEST SARSA - COMPARISON")
    print("="*80)
    
    mc_throughput = np.mean(compute_throughput(mc_successes)[:, -100:])
    mc_throughput_std = np.std(compute_throughput(mc_successes)[:, -100:])
    sarsa_throughput = np.mean(compute_throughput(sarsa_successes)[:, -100:])
    sarsa_throughput_std = np.std(compute_throughput(sarsa_successes)[:, -100:])
    
    mc_return = np.mean(mc_returns[:, -100:])
    mc_return_std = np.std(mc_returns[:, -100:])
    sarsa_return = np.mean(sarsa_returns[:, -100:])
    sarsa_return_std = np.std(sarsa_returns[:, -100:])
    
    print(f"\n{'Algorithm':<30} | {'Final Throughput':<20} | {'Avg Return':<15}")
    print("-" * 80)
    
    mc_label = f"MC ({mc_shaping})"
    sarsa_label = f"SARSA ({sarsa_shaping})"
    
    print(f"{mc_label:<30} | "
          f"{mc_throughput*100:>6.2f}% ± {mc_throughput_std*100:>4.2f}% | "
          f"{mc_return:>6.3f} ± {mc_return_std:>5.3f}")
    print(f"{sarsa_label:<30} | "
          f"{sarsa_throughput*100:>6.2f}% ± {sarsa_throughput_std*100:>4.2f}% | "
          f"{sarsa_return:>6.3f} ± {sarsa_return_std:>5.3f}")
    print("-" * 80)
    
    if sarsa_throughput > mc_throughput:
        ratio = sarsa_throughput / mc_throughput if mc_throughput > 0 else float('inf')
        print(f"\n🏆 SARSA outperforms MC by {ratio:.2f}x ({sarsa_throughput*100:.1f}% vs {mc_throughput*100:.1f}%)")
    else:
        ratio = mc_throughput / sarsa_throughput if sarsa_throughput > 0 else float('inf')
        print(f"\n🏆 MC outperforms SARSA by {ratio:.2f}x ({mc_throughput*100:.1f}% vs {sarsa_throughput*100:.1f}%)")
    
    print("="*80 + "\n")
    
    # Plot throughput
    throughput_data = {
        f'MC ({mc_shaping})': compute_throughput(mc_successes),
        f'SARSA ({sarsa_shaping})': compute_throughput(sarsa_successes)
    }
    plot_comparison(
        throughput_data,
        title='Best MC vs Best SARSA - Throughput',
        ylabel='Throughput',
        save_path=f'{config.RESULTS_DIR}/mc_vs_sarsa_throughput.png'
    )
    
    # Plot returns
    returns_data = {
        f'MC ({mc_shaping})': mc_returns,
        f'SARSA ({sarsa_shaping})': sarsa_returns
    }
    plot_comparison(
        returns_data,
        title='Best MC vs Best SARSA - Real Returns',
        ylabel='Real Return',
        save_path=f'{config.RESULTS_DIR}/mc_vs_sarsa_returns.png'
    )
    
    # Plot lengths
    lengths_data = {
        f'MC ({mc_shaping})': mc_lengths,
        f'SARSA ({sarsa_shaping})': sarsa_lengths
    }
    plot_comparison(
        lengths_data,
        title='Best MC vs Best SARSA - Episode Length',
        ylabel='Episode Length',
        save_path=f'{config.RESULTS_DIR}/mc_vs_sarsa_lengths.png'
    )


if __name__ == "__main__":
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE REWARD SHAPING COMPARISON (ENHANCED)")
    print("="*70)
    
    # Print current configuration
    config.print_config()
    
    print("\nThis script will:")
    print("  1. Compare all MC shaping methods")
    print("  2. Compare all SARSA shaping methods")
    print("  3. Compare best MC vs best SARSA")
    print("\nFeatures:")
    print("  ✅ Progress bars for tracking")
    print("  ✅ Summary tables after each algorithm")
    print("  ✅ Automatic best method detection")
    print("="*70)
    
    # Compare all MC methods
    print("\n[1/3] Comparing all Monte Carlo methods...")
    mc_results = compare_all_mc_methods()
    
    # Compare all SARSA methods
    print("\n[2/3] Comparing all SARSA methods...")
    sarsa_results = compare_all_sarsa_methods()
    
    # Compare best methods
    print("\n[3/3] Comparing best MC vs best SARSA...")
    compare_mc_vs_sarsa_best(
        mc_shaping='potential',      # Change based on your findings
        sarsa_shaping='safety'       # Change based on your findings
    )
    
    print("\n" + "="*70)
    print("ALL COMPARISONS COMPLETED ✅")
    print("="*70)
    print(f"\nResults saved in '{config.RESULTS_DIR}/' directory.")
    print("\n📊 Review the summary tables above for key findings!")
    print("📈 Check the plots for detailed comparisons!")
    print("="*70)