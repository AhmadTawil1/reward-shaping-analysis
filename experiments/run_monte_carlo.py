"""
Multi-run Monte Carlo experiments with reward shaping.
"""

import numpy as np

from env.frozenlake_env import create_frozenlake_env
from algorithms.monte_carlo import MonteCarloAgent
from env.reward_shaping import StepCostShaping, PotentialBasedDistanceShaping
from env.metrics_wrapper import MetricsWrapper  # ✅ ADD THIS


def run_monte_carlo(
    num_episodes=1000,
    num_runs=5,
    shaping_type=None,
    grid_size=6,
    hole_density=0.2,
    is_slippery=True
):
    """
    Run Monte Carlo experiments over multiple random seeds.
    """
    
    # Print hyperparameters
    print("="*60)
    print("MONTE CARLO HYPERPARAMETERS")
    print("="*60)
    print(f"Environment:")
    print(f"  - Grid Size: {grid_size}x{grid_size}")
    print(f"  - Hole Density: {hole_density}")
    print(f"  - Slippery: {is_slippery}")
    print(f"\nTraining:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Runs: {num_runs}")
    print(f"\nAgent (Monte Carlo):")
    print(f"  - Epsilon: 0.1")
    print(f"  - Alpha: 0.1")
    print(f"  - Gamma: 1.0")
    print(f"\nReward Shaping:")
    if shaping_type == "step":
        print(f"  - Type: Step Cost")
        print(f"  - Step Cost: -0.01")
    elif shaping_type == "potential":
        print(f"  - Type: Potential-Based Distance")
        print(f"  - Beta: 1.0")
    else:
        print(f"  - Type: None (baseline)")
    print("="*60)
    print()

    all_successes = []
    all_returns = []
    all_lengths = []

    for seed in range(num_runs):
        print(f"[MC] Run {seed + 1}/{num_runs}")

        # --- Create Base Environment ---
        env = create_frozenlake_env(
            grid_size=grid_size,
            hole_density=hole_density,
            is_slippery=is_slippery,
            seed=seed
        )

        # --- Apply Reward Shaping (if specified) ---
        if shaping_type == "step":
            env = StepCostShaping(env, step_cost=-0.01)
        elif shaping_type == "potential":
            env = PotentialBasedDistanceShaping(env, beta=1.0)

        # --- CRITICAL: Wrap with MetricsWrapper (ALWAYS, as outermost wrapper) ---
        env = MetricsWrapper(env)  # ✅ ADD THIS LINE

        # --- Create Agent ---
        agent = MonteCarloAgent(
            env=env,
            epsilon=0.1,
            alpha=0.1,
            gamma=1.0
        )

        # --- Train ---
        successes, returns, lengths = agent.train(num_episodes=num_episodes)

        env.close()

        all_successes.append(successes)
        all_returns.append(returns)
        all_lengths.append(lengths)

    return (
        np.array(all_successes),
        np.array(all_returns),
        np.array(all_lengths),
    )


if __name__ == "__main__":
    print("Testing Monte Carlo on easy grid (6x6, no holes, deterministic)...")
    successes, returns, lengths = run_monte_carlo(
        num_episodes=500,
        num_runs=3,
        shaping_type=None,
        grid_size=6,
        hole_density=0.2,
        is_slippery=True
    )

    print("\nResults shape:", successes.shape)
    print("Final throughput (last episode, run 0):", 
          np.sum(successes[0]) / len(successes[0]))
    print("Average episode length (last 100 episodes, run 0):", 
          np.mean(lengths[0][-100:]))