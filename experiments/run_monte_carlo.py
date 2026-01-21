"""
Monte Carlo Training - Sequential Execution

Trains Monte Carlo agents one run at a time (sequential execution).
Slower than parallel version but easier to debug and inspect individual runs.

Use this for debugging or small tests. For full experiments, use run_monte_carlo_parallel.py.
"""

import numpy as np

import config
from env.frozenlake_env import create_frozenlake_env
from algorithms.monte_carlo import MonteCarloAgent
from env.reward_shaping import (
    StepCostShaping, 
    PotentialBasedDistanceShaping,
    SafetyBasedShaping,
    ExplorationBonusShaping
)
from env.metrics_wrapper import MetricsWrapper


def run_monte_carlo(
    num_episodes=1000,
    num_runs=5,
    shaping_type=None,
    grid_rows=6,
    grid_cols=6,
    hole_density=0.15,
    is_slippery=True,
    success_rate=0.7,
    epsilon=0.1,
    alpha=0.1,
    gamma=1.0,
    initial_q_value=0.0,
    shaping_params=None,
    return_best_agent=False
):
    """
    Run Monte Carlo experiments over multiple random seeds.
    
    Args:
        num_episodes (int): Number of training episodes
        num_runs (int): Number of independent runs
        shaping_type (str): None, 'step', 'potential', 'safety', 'exploration'
        grid_rows (int): Number of rows in the grid
        grid_cols (int): Number of columns in the grid
        hole_density (float): Hole density (0.1-0.2)
        is_slippery (bool): Slippery dynamics
        success_rate (float): Action success rate when slippery (default: 0.7)
        epsilon (float): Exploration rate
        alpha (float): Learning rate
        gamma (float): Discount factor (should be 1.0 per project)
        initial_q_value (float): Initial Q-value for optimistic initialization
        shaping_params (dict): Additional parameters for shaping (e.g., {'step_cost': -0.01})
        return_best_agent (bool): If True, also return the trained agent from the best run
    
    Returns:
        tuple: (successes, returns, lengths) - each shape (num_runs, num_episodes)
               If return_best_agent=True: (successes, returns, lengths, best_agent, best_env)
    """
    
    # Default shaping parameters
    if shaping_params is None:
        shaping_params = {}
    
    # Print hyperparameters
    print("="*60)
    print("MONTE CARLO HYPERPARAMETERS")
    print("="*60)
    print(f"Environment:")
    print(f"  - Grid Size: {grid_rows}x{grid_cols}")
    print(f"  - Hole Density: {hole_density}")
    print(f"  - Slippery: {is_slippery}")
    if is_slippery:
        print(f"  - Success Rate: {success_rate}")
    print(f"\nTraining:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Runs: {num_runs}")
    print(f"\nAgent (Monte Carlo):")
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
    
    # Track best agent if requested
    best_agent = None
    best_env = None
    best_throughput = -1

    for seed in range(num_runs):
        print(f"[MC] Run {seed + 1}/{num_runs}")

        # --- Seed numpy RNG for this run (for agent's random actions) ---
        # Use different seed for each run to ensure independence
        # while keeping environment seed constant for same map
        run_seed = config.RANDOM_SEED + seed + 1000  # Offset to avoid collision
        np.random.seed(run_seed)

        # --- Create Base Environment ---
        # Use fixed seed from config for all runs to ensure same map
        env = create_frozenlake_env(
            rows=grid_rows,
            cols=grid_cols,
            hole_density=hole_density,
            is_slippery=is_slippery,
            seed=config.RANDOM_SEED,
            success_rate=success_rate
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
        agent = MonteCarloAgent(
            env=env,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            initial_q_value=initial_q_value
        )

        # --- Train ---
        successes, returns, lengths = agent.train(num_episodes=num_episodes)
        
        # Track best agent based on final throughput
        if return_best_agent:
            run_throughput = np.sum(successes) / len(successes)
            if run_throughput > best_throughput:
                best_throughput = run_throughput
                best_agent = agent
                best_env = env

        # Close env if not keeping it
        if not return_best_agent or agent is not best_agent:
            env.close()

        all_successes.append(successes)
        all_returns.append(returns)
        all_lengths.append(lengths)

    if return_best_agent:
        return (
            np.array(all_successes),
            np.array(all_returns),
            np.array(all_lengths),
            best_agent,
            best_env
        )
    else:
        return (
            np.array(all_successes),
            np.array(all_returns),
            np.array(all_lengths),
        )


if __name__ == "__main__":
    print("Testing Monte Carlo on 6x6 grid with 15% holes...")
    successes, returns, lengths = run_monte_carlo(
        num_episodes=500,
        num_runs=3,
        shaping_type="safety",
        grid_rows=6,
        grid_cols=6,
        hole_density=0.15,
        is_slippery=True
    )

    print("\nResults shape:", successes.shape)
    print("Final throughput (last episode, run 0):", 
          np.sum(successes[0]) / len(successes[0]))
    print("Average episode length (last 100 episodes, run 0):", 
          np.mean(lengths[0][-100:]))