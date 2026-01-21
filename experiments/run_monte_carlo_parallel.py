"""
Multi-run Monte Carlo experiments with reward shaping - PARALLEL VERSION.

This version uses joblib to run multiple experiments in parallel,
significantly reducing total execution time.
"""

import numpy as np
from joblib import Parallel, delayed

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


def _run_single_experiment(
    seed_idx,
    num_episodes,
    shaping_type,
    grid_rows,
    grid_cols,
    hole_density,
    is_slippery,
    success_rate,
    epsilon,
    alpha,
    gamma,
    initial_q_value,
    shaping_params
):
    """
    Run a single Monte Carlo experiment (one seed).
    
    This function is designed to be called in parallel.
    
    Args:
        seed_idx (int): Index for this run (0 to num_runs-1)
        ... (other parameters same as run_monte_carlo)
    
    Returns:
        tuple: (successes, returns, lengths, throughput) for this run
    """
    print(f"[MC] Starting Run {seed_idx + 1}")
    
    # --- Seed numpy RNG for this run ---
    run_seed = config.RANDOM_SEED + seed_idx + 1000
    np.random.seed(run_seed)
    
    # --- Create Base Environment ---
    env = create_frozenlake_env(
        rows=grid_rows,
        cols=grid_cols,
        hole_density=hole_density,
        is_slippery=is_slippery,
        seed=config.RANDOM_SEED,
        success_rate=success_rate
    )
    
    # --- Apply Reward Shaping ---
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
    
    # --- Wrap with MetricsWrapper ---
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
    
    # Calculate throughput for this run
    throughput = np.sum(successes) / len(successes)
    
    # Close environment
    env.close()
    
    print(f"[MC] Completed Run {seed_idx + 1} (throughput: {throughput:.4f})")
    
    return successes, returns, lengths, throughput, agent, env


def run_monte_carlo_parallel(
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
    return_best_agent=False,
    n_jobs=-1
):
    """
    Run Monte Carlo experiments in parallel over multiple random seeds.
    
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
        shaping_params (dict): Additional parameters for shaping
        return_best_agent (bool): If True, also return the trained agent from the best run
        n_jobs (int): Number of parallel jobs. -1 means use all CPU cores.
    
    Returns:
        tuple: (successes, returns, lengths) - each shape (num_runs, num_episodes)
               If return_best_agent=True: (successes, returns, lengths, best_agent, best_env)
    """
    
    # Default shaping parameters
    if shaping_params is None:
        shaping_params = {}
    
    # Print hyperparameters
    print("="*60)
    print("MONTE CARLO HYPERPARAMETERS (PARALLEL)")
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
    print(f"  - Parallel Jobs: {n_jobs if n_jobs > 0 else 'All CPU cores'}")
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
    
    # Run experiments in parallel
    print(f"Running {num_runs} experiments in parallel...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_single_experiment)(
            seed_idx=i,
            num_episodes=num_episodes,
            shaping_type=shaping_type,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            hole_density=hole_density,
            is_slippery=is_slippery,
            success_rate=success_rate,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            initial_q_value=initial_q_value,
            shaping_params=shaping_params
        )
        for i in range(num_runs)
    )
    
    print("\nAll runs completed!")
    
    # Unpack results
    all_successes = []
    all_returns = []
    all_lengths = []
    all_throughputs = []
    all_agents = []
    all_envs = []
    
    for successes, returns, lengths, throughput, agent, env in results:
        all_successes.append(successes)
        all_returns.append(returns)
        all_lengths.append(lengths)
        all_throughputs.append(throughput)
        all_agents.append(agent)
        all_envs.append(env)
    
    # Find best agent if requested
    if return_best_agent:
        best_idx = np.argmax(all_throughputs)
        best_agent = all_agents[best_idx]
        best_env = all_envs[best_idx]
        
        print(f"\nBest run: #{best_idx + 1} with throughput {all_throughputs[best_idx]:.4f}")
        
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
    print("Testing PARALLEL Monte Carlo on 6x6 grid with 15% holes...")
    successes, returns, lengths = run_monte_carlo_parallel(
        num_episodes=10000,
        num_runs=20,  # Try 20 runs in parallel!
        shaping_type="safety",
        grid_rows=6,
        grid_cols=6,
        hole_density=0.15,
        is_slippery=True,
        n_jobs=-1  # Use all CPU cores
    )

    print("\nResults shape:", successes.shape)
    print("Final throughput (last episode, run 0):", 
          np.sum(successes[0]) / len(successes[0]))
    print("Average episode length (last 100 episodes, run 0):", 
          np.mean(lengths[0][-100:]))
