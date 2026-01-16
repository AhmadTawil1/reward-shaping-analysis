"""
Beta Sensitivity Study for Potential-Based Shaping

Compare β=1.0 vs β=0.7 for Potential-Based Distance Shaping
to understand the effect of shaping strength on learning.

Usage:
    python beta_comparison.py

Results:
    - Saves plots to results/ directory
    - Prints summary table
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import config
from experiments.run_monte_carlo import run_monte_carlo
from experiments.run_sarsa import run_sarsa


def compute_throughput(successes):
    """Compute cumulative throughput from success array."""
    cumulative = np.cumsum(successes, axis=1)
    episodes = np.arange(1, successes.shape[1] + 1)
    return cumulative / episodes


def plot_beta_comparison(results_dict, algorithm_name, save_path):
    """
    Plot comparison of different beta values with 95% CI.
    
    Args:
        results_dict (dict): {beta_value: throughput_array (runs, episodes)}
        algorithm_name (str): 'Monte Carlo' or 'SARSA'
        save_path (str): Where to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['C0', 'C1']  # Blue for β=1.0, Orange for β=0.7
    
    for idx, (beta_value, data) in enumerate(sorted(results_dict.items())):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        n = data.shape[0]
        ci = 1.96 * std / np.sqrt(n)  # 95% confidence interval
        
        episodes = np.arange(1, data.shape[1] + 1)
        
        label = f'β = {beta_value}'
        color = colors[idx]
        
        plt.plot(episodes, mean, label=label, linewidth=2.5, color=color)
        plt.fill_between(episodes, mean - ci, mean + ci, alpha=0.2, color=color)
    
    plt.xlabel('Episode', fontsize=13)
    plt.ylabel('Throughput', fontsize=13)
    plt.title(f'{algorithm_name}: Potential-Based Shaping - Beta (β) Comparison', 
              fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()


def plot_returns_comparison(returns_dict, algorithm_name, save_path):
    """
    Plot returns comparison of different beta values with 95% CI.
    
    Args:
        returns_dict (dict): {beta_value: returns_array (runs, episodes)}
        algorithm_name (str): 'Monte Carlo' or 'SARSA'
        save_path (str): Where to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['C0', 'C1']
    
    for idx, (beta_value, data) in enumerate(sorted(returns_dict.items())):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        n = data.shape[0]
        ci = 1.96 * std / np.sqrt(n)
        
        episodes = np.arange(1, data.shape[1] + 1)
        
        label = f'β = {beta_value}'
        color = colors[idx]
        
        plt.plot(episodes, mean, label=label, linewidth=2.5, color=color)
        plt.fill_between(episodes, mean - ci, mean + ci, alpha=0.2, color=color)
    
    plt.xlabel('Episode', fontsize=13)
    plt.ylabel('Real Return', fontsize=13)
    plt.title(f'{algorithm_name}: Potential-Based Shaping - Beta (β) Returns Comparison', 
              fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()


def print_beta_results_table(throughput_dict, returns_dict, algorithm_name):
    """
    Print formatted summary table of beta comparison results.
    
    Args:
        throughput_dict (dict): {beta_value: throughput_array}
        returns_dict (dict): {beta_value: returns_array}
        algorithm_name (str): Algorithm name for display
    """
    print("\n" + "="*80)
    print(f"{algorithm_name.upper()}: BETA (β) SENSITIVITY - RESULTS SUMMARY")
    print("="*80)
    print(f"{'Beta (β)':<15} | {'Throughput (last 100)':<25} | {'Avg Return (last 100)':<25}")
    print("-" * 80)
    
    # Calculate statistics
    stats = []
    for beta_value in sorted(throughput_dict.keys()):
        throughput = throughput_dict[beta_value]
        returns = returns_dict[beta_value]
        
        avg_throughput = np.mean(throughput[:, -100:])
        std_throughput = np.std(throughput[:, -100:])
        
        avg_return = np.mean(returns[:, -100:])
        std_return = np.std(returns[:, -100:])
        
        stats.append((beta_value, avg_throughput, std_throughput, avg_return, std_return))
    
    # Print table
    for beta_value, avg_tp, std_tp, avg_ret, std_ret in stats:
        marker = "⭐" if avg_tp == max(s[1] for s in stats) else "  "
        print(f"{marker} {beta_value:<13.1f} | "
              f"{avg_tp*100:>7.2f}% ± {std_tp*100:>5.2f}% | "
              f"{avg_ret:>8.3f} ± {std_ret:>6.3f}")
    
    print("-" * 80)
    
    # Find best
    best_beta = max(stats, key=lambda x: x[1])[0]
    best_throughput = max(s[1] for s in stats)
    print(f"\n🎯 Best β: {best_beta:.1f}")
    print(f"   Throughput: {best_throughput*100:.2f}%")
    
    # Performance difference
    beta_1_tp = next(s[1] for s in stats if s[0] == 1.0)
    beta_07_tp = next(s[1] for s in stats if s[0] == 0.7)
    
    if beta_1_tp > beta_07_tp:
        diff = ((beta_1_tp - beta_07_tp) / beta_07_tp * 100) if beta_07_tp > 0 else float('inf')
        print(f"\n📊 β=1.0 outperforms β=0.7 by {diff:+.1f}%")
    elif beta_07_tp > beta_1_tp:
        diff = ((beta_07_tp - beta_1_tp) / beta_1_tp * 100) if beta_1_tp > 0 else float('inf')
        print(f"\n📊 β=0.7 outperforms β=1.0 by {diff:+.1f}%")
    else:
        print(f"\n⚖️  β=1.0 and β=0.7 perform equally")
    
    print("="*80)


def compare_beta_monte_carlo():
    """Compare β=1.0 vs β=0.7 for Monte Carlo with Potential-Based Shaping."""
    print("\n" + "="*70)
    print("MONTE CARLO: BETA (β) COMPARISON FOR POTENTIAL-BASED SHAPING")
    print("="*70)
    
    cfg = config.get_final_config() if not config.TEST_MODE else config.get_test_config()
    
    beta_values = [1.0, 0.7]
    
    throughput_results = {}
    returns_results = {}
    
    for beta in tqdm(beta_values, desc="Beta values", unit="value"):
        print(f"\n🔄 Running Monte Carlo with β = {beta}")
        
        successes, returns, lengths = run_monte_carlo(
            num_episodes=cfg['num_episodes'],
            num_runs=cfg['num_runs'],
            shaping_type='potential',
            grid_rows=cfg['grid_rows'],
            grid_cols=cfg['grid_cols'],
            hole_density=cfg['hole_density'],
            is_slippery=cfg['is_slippery'],
            success_rate=cfg['success_rate'],
            epsilon=cfg['epsilon'],
            alpha=cfg['alpha'],
            gamma=cfg['gamma'],
            initial_q_value=config.INITIAL_Q_VALUE,
            shaping_params={'beta': beta}
        )
        
        throughput_results[beta] = compute_throughput(successes)
        returns_results[beta] = returns
        
        # Quick summary
        final_throughput = np.mean(throughput_results[beta][:, -100:])
        avg_return = np.mean(returns[:, -100:])
        print(f"   Final throughput: {final_throughput*100:.2f}%")
        print(f"   Avg return (last 100): {avg_return:.3f}")
    
    # Print summary table
    print_beta_results_table(throughput_results, returns_results, "Monte Carlo")
    
    # Plot throughput comparison
    print("\n📊 Generating throughput comparison plot...")
    plot_beta_comparison(
        throughput_results,
        algorithm_name='Monte Carlo',
        save_path=f'{config.RESULTS_DIR}/mc_beta_comparison_throughput.png'
    )
    
    # Plot returns comparison
    print("📊 Generating returns comparison plot...")
    plot_returns_comparison(
        returns_results,
        algorithm_name='Monte Carlo',
        save_path=f'{config.RESULTS_DIR}/mc_beta_comparison_returns.png'
    )
    
    return throughput_results, returns_results


def compare_beta_sarsa():
    """Compare β=1.0 vs β=0.7 for SARSA with Potential-Based Shaping."""
    print("\n" + "="*70)
    print("SARSA: BETA (β) COMPARISON FOR POTENTIAL-BASED SHAPING")
    print("="*70)
    
    cfg = config.get_final_config() if not config.TEST_MODE else config.get_test_config()
    
    beta_values = [1.0, 0.7]
    
    throughput_results = {}
    returns_results = {}
    
    for beta in tqdm(beta_values, desc="Beta values", unit="value"):
        print(f"\n🔄 Running SARSA with β = {beta}")
        
        successes, returns, lengths = run_sarsa(
            num_episodes=cfg['num_episodes'],
            num_runs=cfg['num_runs'],
            shaping_type='potential',
            grid_rows=cfg['grid_rows'],
            grid_cols=cfg['grid_cols'],
            hole_density=cfg['hole_density'],
            is_slippery=cfg['is_slippery'],
            success_rate=cfg['success_rate'],
            epsilon=cfg['epsilon'],
            alpha=cfg['alpha'],
            gamma=cfg['gamma'],
            initial_q_value=config.INITIAL_Q_VALUE,
            shaping_params={'beta': beta}
        )
        
        throughput_results[beta] = compute_throughput(successes)
        returns_results[beta] = returns
        
        # Quick summary
        final_throughput = np.mean(throughput_results[beta][:, -100:])
        avg_return = np.mean(returns[:, -100:])
        print(f"   Final throughput: {final_throughput*100:.2f}%")
        print(f"   Avg return (last 100): {avg_return:.3f}")
    
    # Print summary table
    print_beta_results_table(throughput_results, returns_results, "SARSA")
    
    # Plot throughput comparison
    print("\n📊 Generating throughput comparison plot...")
    plot_beta_comparison(
        throughput_results,
        algorithm_name='SARSA',
        save_path=f'{config.RESULTS_DIR}/sarsa_beta_comparison_throughput.png'
    )
    
    # Plot returns comparison
    print("📊 Generating returns comparison plot...")
    plot_returns_comparison(
        returns_results,
        algorithm_name='SARSA',
        save_path=f'{config.RESULTS_DIR}/sarsa_beta_comparison_returns.png'
    )
    
    return throughput_results, returns_results


if __name__ == "__main__":
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    if config.RANDOM_SEED is not None:
        np.random.seed(config.RANDOM_SEED)
        print(f"\n🔒 Random seed set to {config.RANDOM_SEED} for reproducibility")
    
    print("\n" + "="*70)
    print("BETA (β) SENSITIVITY COMPARISON")
    print("="*70)
    
    # Print configuration
    cfg = config.get_final_config() if not config.TEST_MODE else config.get_test_config()
    print(f"\n📋 Configuration:")
    print(f"  Episodes per run: {cfg['num_episodes']}")
    print(f"  Number of runs: {cfg['num_runs']}")
    print(f"  Grid: {cfg['grid_rows']}×{cfg['grid_cols']}")
    print(f"  Hole Density: {cfg['hole_density']}")
    print(f"  Slippery: {cfg['is_slippery']}")
    if cfg['is_slippery']:
        print(f"  Success Rate: {cfg['success_rate']}")
    
    print(f"\n📊 This script will compare:")
    print(f"  • β = 1.0 (full strength)")
    print(f"  • β = 0.7 (matches success_rate)")
    
    print("\n" + "="*70)
    
    # Estimate runtime
    total_experiments = 2 * 2  # 2 beta values × 2 algorithms
    estimated_time_min = total_experiments * 8
    estimated_time_max = total_experiments * 15
    
    print(f"\n⏱️  Estimated runtime: {estimated_time_min}-{estimated_time_max} minutes")
    
    input("\n👉 Press ENTER to start, or Ctrl+C to cancel...")
    
    # Run Monte Carlo comparison
    print("\n[1/2] Running Monte Carlo beta comparison...")
    mc_throughput, mc_returns = compare_beta_monte_carlo()
    
    # Run SARSA comparison
    print("\n[2/2] Running SARSA beta comparison...")
    sarsa_throughput, sarsa_returns = compare_beta_sarsa()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ BETA COMPARISON COMPLETED!")
    print("="*70)
    print(f"\n📁 Results saved to '{config.RESULTS_DIR}/':")
    print("  📈 mc_beta_comparison_throughput.png")
    print("  📈 mc_beta_comparison_returns.png")
    print("  📈 sarsa_beta_comparison_throughput.png")
    print("  📈 sarsa_beta_comparison_returns.png")
    
    print("\n" + "="*70)
    print("💡 KEY INSIGHTS FOR YOUR REPORT")
    print("="*70)
    
    print("\n1️⃣  Beta (β) Parameter:")
    print("   • Controls the strength of potential-based shaping")
    print("   • β=1.0: Full strength shaping signal")
    print("   • β=0.7: Matches the environment's success_rate")
    print("   • Lower β = gentler guidance, more agent exploration")
    
    print("\n2️⃣  Expected Behavior:")
    print("   • β=1.0 might converge faster initially")
    print("   • β=0.7 might be more stable in stochastic environments")
    print("   • The 'best' β depends on algorithm and environment")
    
    print("\n3️⃣  For Your Report:")
    print("   • Include both throughput and returns plots")
    print("   • Explain which β performed better and why")
    print("   • Discuss the tradeoff between shaping strength and stability")
    print("   • Note any differences between MC and SARSA behavior")
    
    print("="*70)