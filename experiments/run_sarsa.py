"""
Multi-run SARSA experiments with reward shaping.
"""

import numpy as np

from env.frozenlake_env import create_frozenlake_env
from algorithms.sarsa import SARSAAgent
from env.reward_shaping import (
    StepCostShaping, 
    PotentialBasedDistanceShaping,
    SafetyBasedShaping,
    ExplorationBonusShaping
)
from env.metrics_wrapper import MetricsWrapper


def run_sarsa(
    num_episodes=1000,
    num_runs=5,
    shaping_type=None,
    grid_size=6,
    hole_density=0.15,
    is_slippery=True,
    epsilon=0.1,
    alpha=0.1,
    gamma=1.0,
    shaping_params=None
):
    """
    Run SARSA experiments over multiple random seeds.
    
    Args:
        num_episodes (int): Number of training episodes
        num_runs (int): Number of independent runs
        shaping_type (str): None, 'step', 'potential', 'safety', 'exploration'
        grid_size (int): Grid size
        hole_density (float): Hole density (0.1-0.2)
        is_slippery (bool): Slippery dynamics
        epsilon (float): Exploration rate
        alpha (float): Learning rate
        gamma (float): Discount factor (should be 1.0 per project)
        shaping_params (dict): Additional parameters for shaping (e.g., {'step_cost': -0.01})
    
    Returns:
        tuple: (successes, returns, lengths) - each shape (num_runs, num_episodes)
    """
    
    # Default shaping parameters
    if shaping_params is None:
        shaping_params = {}
    
    # Print hyperparameters
    print("="*60)
    print("SARSA HYPERPARAMETERS")
    print("="*60)
    print(f"Environment:")
    print(f"  - Grid Size: {grid_size}x{grid_size}")
    print(f"  - Hole Density: {hole_density}")
    print(f"  - Slippery: {is_slippery}")
    if is_slippery:
        print(f"  - Success Rate: 0.7")
    print(f"\nTraining:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Runs: {num_runs}")
    print(f"\nAgent (SARSA):")
    print(f"  - Epsilon: {epsilon}")
    print(f"  - Alpha: {alpha}")
    print(f"  - Gamma: {gamma}")
    print(f"\nReward Shaping:")
    
    if shaping_type == "step":
        step_cost = shaping_params.get('step_cost', -0.01)
        print(f"  - Type: Step Cost")
        print(f"  - Step Cost: {step_cost}")
    elif shaping_type == "potential":
        beta = shaping_params.get('beta', 1.0)
        print(f"  - Type: Potential-Based Distance")
        print(f"  - Beta: {beta}")
    elif shaping_type == "safety":
        beta = shaping_params.get('beta', 0.1)
        print(f"  - Type: Safety-Based (Custom)")
        print(f"  - Beta: {beta}")
    elif shaping_type == "exploration":
        beta = shaping_params.get('beta', 0.05)
        print(f"  - Type: Exploration Bonus (Custom)")
        print(f"  - Beta: {beta}")
    else:
        print(f"  - Type: None (baseline)")
    print("="*60)
    print()

    all_successes = []
    all_returns = []
    all_lengths = []

    for seed in range(num_runs):
        print(f"[SARSA] Run {seed + 1}/{num_runs}")

        # --- Create Base Environment ---
        env = create_frozenlake_env(
            grid_size=grid_size,
            hole_density=hole_density,
            is_slippery=is_slippery,
            seed=seed,
            success_rate=0.7
        )

        # --- Apply Reward Shaping (if specified) ---
        if shaping_type == "step":
            step_cost = shaping_params.get('step_cost', -0.01)
            env = StepCostShaping(env, step_cost=step_cost)
        elif shaping_type == "potential":
            beta = shaping_params.get('beta', 1.0)
            env = PotentialBasedDistanceShaping(env, beta=beta)
        elif shaping_type == "safety":
            beta = shaping_params.get('beta', 0.1)
            env = SafetyBasedShaping(env, beta=beta)
        elif shaping_type == "exploration":
            beta = shaping_params.get('beta', 0.05)
            env = ExplorationBonusShaping(env, beta=beta)

        # --- CRITICAL: Wrap with MetricsWrapper (ALWAYS, as outermost wrapper) ---
        env = MetricsWrapper(env)

        # --- Create Agent ---
        agent = SARSAAgent(
            env=env,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma
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
    print("Testing SARSA on 6x6 grid with 15% holes...")
    successes, returns, lengths = run_sarsa(
        num_episodes=500,
        num_runs=3,
        shaping_type="exploration",
        grid_size=6,
        hole_density=0.15,
        is_slippery=True
    )

    print("\nResults shape:", successes.shape)
    print("Final throughput (last episode, run 0):", 
          np.sum(successes[0]) / len(successes[0]))
    print("Average episode length (last 100 episodes, run 0):", 
          np.mean(lengths[0][-100:]))