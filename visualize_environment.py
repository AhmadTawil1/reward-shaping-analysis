import os
import config
import numpy as np

from env.frozenlake_env import create_frozenlake_env
from algorithms.sarsa import SARSAAgent
from env.reward_shaping import SafetyBasedShaping
from env.metrics_wrapper import MetricsWrapper
from utils.visualization import visualize_grid, print_grid_info, compare_policies

os.makedirs('report_figures', exist_ok=True)

# Get SAME settings as experiments
cfg = config.get_final_config() if not config.TEST_MODE else config.get_test_config()

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS FOR REPORT")
print("="*70)
print(f"\nUsing configuration:")
print(f"  Grid: {cfg['grid_rows']}×{cfg['grid_cols']}")
print(f"  Hole Density: {cfg['hole_density']}")
print(f"  Slippery: {cfg['is_slippery']}")
print(f"  Episodes: 5000 (20 runs)")
print(f"  Epsilon: {cfg['epsilon']}")
print(f"  Alpha: {cfg['alpha']}")
print("="*70)

# 1. Environment grid
env = create_frozenlake_env(
    rows=cfg['grid_rows'],
    cols=cfg['grid_cols'],
    hole_density=cfg['hole_density'],  
    is_slippery=cfg['is_slippery'],
    seed=242
)

print_grid_info(env)

visualize_grid(
    env, 
    title=f"FrozenLake {cfg['grid_rows']}×{cfg['grid_cols']} Environment ({int(cfg['hole_density']*100)}% holes)",
    save_path="report_figures/environment_grid.png"
)

# 2. SARSA + Safety (20 runs × 5000 episodes)
print(f"\n[2/3] Training SARSA + Safety (20 runs × 5000 episodes)...")

all_throughputs_sarsa = []

for run in range(20):
    print(f"   Run {run + 1}/20...", end=" ")
    
    env_sarsa = create_frozenlake_env(
        rows=cfg['grid_rows'],
        cols=cfg['grid_cols'],
        hole_density=cfg['hole_density'],
        is_slippery=cfg['is_slippery'],
        seed=242 + run
    )
    env_sarsa = SafetyBasedShaping(env_sarsa, beta=config.BETA_SAFETY)
    env_sarsa = MetricsWrapper(env_sarsa)
    
    agent_sarsa = SARSAAgent(
        env_sarsa,
        epsilon=cfg['epsilon'],
        alpha=cfg['alpha'],
        gamma=cfg['gamma']
    )
    
    successes, returns, lengths = agent_sarsa.train(num_episodes=5000)
    throughput = np.sum(successes) / len(successes)
    all_throughputs_sarsa.append(throughput)
    
    print(f"Throughput: {throughput:.1%}")
    
    env_sarsa.close()

throughput_sarsa = np.mean(all_throughputs_sarsa)

# Train one final time for policy visualization
env_sarsa = create_frozenlake_env(
    rows=cfg['grid_rows'],
    cols=cfg['grid_cols'],
    hole_density=cfg['hole_density'],
    is_slippery=cfg['is_slippery'],
    seed=242
)
env_sarsa = SafetyBasedShaping(env_sarsa, beta=config.BETA_SAFETY)
env_sarsa = MetricsWrapper(env_sarsa)

agent_sarsa = SARSAAgent(
    env_sarsa,
    epsilon=cfg['epsilon'],
    alpha=cfg['alpha'],
    gamma=cfg['gamma']
)
agent_sarsa.train(num_episodes=5000)

policy_sarsa = agent_sarsa.get_policy()
value_sarsa = agent_sarsa.get_value_function()

print(f"\n   ✓ Average Throughput (20 runs): {throughput_sarsa:.1%}")

visualize_grid(
    env,
    policy=policy_sarsa,
    value_function=value_sarsa,
    title=f"SARSA + Safety-Based Shaping (Avg Throughput: {throughput_sarsa:.1%})",
    save_path="report_figures/sarsa_safety_policy.png"
)

env_sarsa.close()

# 3. SARSA Baseline (20 runs × 5000 episodes)
print(f"\n[3/3] Training SARSA Baseline (20 runs × 5000 episodes)...")

all_throughputs_baseline = []

for run in range(20):
    print(f"   Run {run + 1}/20...", end=" ")
    
    env_baseline = create_frozenlake_env(
        rows=cfg['grid_rows'],
        cols=cfg['grid_cols'],
        hole_density=cfg['hole_density'],
        is_slippery=cfg['is_slippery'],
        seed=242 + run
    )
    env_baseline = MetricsWrapper(env_baseline)
    
    agent_baseline = SARSAAgent(
        env_baseline,
        epsilon=cfg['epsilon'],
        alpha=cfg['alpha'],
        gamma=cfg['gamma']
    )
    
    successes, returns, lengths = agent_baseline.train(num_episodes=5000)
    throughput = np.sum(successes) / len(successes)
    all_throughputs_baseline.append(throughput)
    
    print(f"Throughput: {throughput:.1%}")
    
    env_baseline.close()

throughput_baseline = np.mean(all_throughputs_baseline)

# Train one final time for policy visualization
env_baseline = create_frozenlake_env(
    rows=cfg['grid_rows'],
    cols=cfg['grid_cols'],
    hole_density=cfg['hole_density'],
    is_slippery=cfg['is_slippery'],
    seed=242
)
env_baseline = MetricsWrapper(env_baseline)

agent_baseline = SARSAAgent(
    env_baseline,
    epsilon=cfg['epsilon'],
    alpha=cfg['alpha'],
    gamma=cfg['gamma']
)
agent_baseline.train(num_episodes=5000)

policy_baseline = agent_baseline.get_policy()

print(f"\n   ✓ Average Throughput (20 runs): {throughput_baseline:.1%}")

compare_policies(
    env,
    {
        f'SARSA Baseline ({throughput_baseline:.1%})': policy_baseline,
        f'SARSA + Safety ({throughput_sarsa:.1%})': policy_sarsa
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
print(f"\nResults (averaged over 20 runs × 5000 episodes):")
print(f"  SARSA Baseline: {throughput_baseline:.1%}")
print(f"  SARSA + Safety: {throughput_sarsa:.1%}")
print("="*70)