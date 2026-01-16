"""
Visualize the FrozenLake environment and learned policies.
"""

import os  # ADD THIS

from env.frozenlake_env import create_frozenlake_env
from algorithms.sarsa import SARSAAgent
from algorithms.monte_carlo import MonteCarloAgent
from env.reward_shaping import SafetyBasedShaping
from env.metrics_wrapper import MetricsWrapper
from utils.visualization import visualize_grid, print_grid_info, compare_policies

# CREATE FOLDER FIRST - ADD THIS
os.makedirs('report_figures', exist_ok=True)


# 1. Show the environment
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS FOR REPORT")
print("="*70)

env = create_frozenlake_env(
    grid_size=6,
    hole_density=0.15,
    is_slippery=True,
    seed=42
)

# Print grid info
print_grid_info(env)

# Visualize empty grid
print("\n[1/3] Visualizing environment grid...")
visualize_grid(
    env, 
    title="FrozenLake 6x6 Environment (15% holes)",
    save_path="report_figures/environment_grid.png"
)

# 2. Train and visualize SARSA with Safety (your best method)
print("\n[2/3] Training SARSA with Safety-Based shaping...")
env_sarsa = create_frozenlake_env(grid_size=6, hole_density=0.15, is_slippery=True, seed=42)
env_sarsa = SafetyBasedShaping(env_sarsa, beta=0.1)
env_sarsa = MetricsWrapper(env_sarsa)

agent_sarsa = SARSAAgent(env_sarsa, epsilon=0.1, alpha=0.1, gamma=1.0)
agent_sarsa.train(num_episodes=2000)

policy_sarsa = agent_sarsa.get_policy()
value_sarsa = agent_sarsa.get_value_function()

visualize_grid(
    env,
    policy=policy_sarsa,
    value_function=value_sarsa,
    title="SARSA + Safety-Based Shaping - Learned Policy (56% success)",
    save_path="report_figures/sarsa_safety_policy.png"
)

env_sarsa.close()

# 3. Train baseline for comparison
print("\n[3/3] Training baseline SARSA for comparison...")
env_baseline = create_frozenlake_env(grid_size=6, hole_density=0.15, is_slippery=True, seed=42)
env_baseline = MetricsWrapper(env_baseline)

agent_baseline = SARSAAgent(env_baseline, epsilon=0.1, alpha=0.1, gamma=1.0)
agent_baseline.train(num_episodes=2000)

policy_baseline = agent_baseline.get_policy()

# Compare policies
compare_policies(
    env,
    {
        'SARSA Baseline (42%)': policy_baseline,
        'SARSA + Safety (56%)': policy_sarsa
    },
    title="Policy Comparison: Baseline vs Safety-Based Shaping",
    save_path="report_figures/policy_comparison.png"
)

env_baseline.close()
env.close()

print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS GENERATED!")
print("="*70)
print("\nSaved to report_figures/:")
print("  1. environment_grid.png")
print("  2. sarsa_safety_policy.png")
print("  3. policy_comparison.png")
print("="*70)