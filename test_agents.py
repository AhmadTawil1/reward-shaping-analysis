"""
Simple test script to verify agents are working correctly.
"""

import numpy as np
from env.frozenlake_env import create_frozenlake_env
from env.metrics_wrapper import MetricsWrapper  # ✅ ADD THIS
from env.reward_shaping import StepCostShaping, PotentialBasedDistanceShaping
from algorithms.monte_carlo import MonteCarloAgent
from algorithms.sarsa import SARSAAgent
import matplotlib.pyplot as plt


# ========================================
# COMMON HYPERPARAMETERS
# ========================================
COMMON_HYPERPARAMS = {
    # Environment
    'grid_size': 6,
    'hole_density': 0.15,
    'is_slippery': True,
    'seed': 42,
    
    # Agent
    'epsilon': 0.1,
    'alpha': 0.1,
    'gamma': 1.0,
    
    # Training
    'num_episodes': 3000,
    'num_runs': 20,  # Number of runs with different seeds
    
    # Reward Shaping
    'shaping_type': None,  # Options: None, "step", "potential"
    'step_cost': -0.01,  # Used if shaping_type == "step"
    'beta': 1.0,  # Used if shaping_type == "potential"
}


def test_monte_carlo(hyperparams=None):
    """Test Monte Carlo agent.
    
    Args:
        hyperparams (dict): Hyperparameter dictionary. If None, uses COMMON_HYPERPARAMS.
    """
    if hyperparams is None:
        hyperparams = COMMON_HYPERPARAMS
    
    # Extract hyperparameters
    grid_size = hyperparams['grid_size']
    hole_density = hyperparams['hole_density']
    is_slippery = hyperparams['is_slippery']
    seed = hyperparams['seed']
    epsilon = hyperparams['epsilon']
    alpha = hyperparams['alpha']
    gamma = hyperparams['gamma']
    num_episodes = hyperparams['num_episodes']
    num_runs = hyperparams['num_runs']
    shaping_type = hyperparams['shaping_type']
    step_cost = hyperparams['step_cost']
    beta = hyperparams['beta']
    print("\n" + "="*60)
    print("TESTING MONTE CARLO")
    print("="*60)
    

    
    # Print hyperparameters
    print("\nMONTE CARLO HYPERPARAMETERS:")
    print("-" * 40)
    print(f"Environment:")
    print(f"  - Grid Size: {grid_size}x{grid_size}")
    print(f"  - Hole Density: {hole_density}")
    print(f"  - Slippery: {is_slippery}")
    print(f"  - Seed: {seed}")
    print(f"\nAgent (Monte Carlo):")
    print(f"  - Epsilon: {epsilon}")
    print(f"  - Alpha: {alpha}")
    print(f"  - Gamma: {gamma}")
    print(f"\nTraining:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Runs: {num_runs}")
    print(f"\nReward Shaping:")
    if shaping_type == "step":
        print(f"  - Type: Step Cost")
        print(f"  - Step Cost: {step_cost}")
    elif shaping_type == "potential":
        print(f"  - Type: Potential-Based Distance")
        print(f"  - Beta: {beta}")
    else:
        print(f"  - Type: None (baseline)")
    print("-" * 40)
    
    # Run multiple experiments with different seeds
    all_successes = []
    all_returns = []
    all_lengths = []
    
    for run in range(num_runs):
        print(f"\n[MC] Run {run + 1}/{num_runs}")
        
        # Create environment with run-specific seed
        env = create_frozenlake_env(
            grid_size=grid_size,
            hole_density=hole_density,
            is_slippery=is_slippery,
            seed=seed + run  # Different seed for each run
        )
        
        # Apply reward shaping (if specified)
        if shaping_type == "step":
            env = StepCostShaping(env, step_cost=step_cost)
        elif shaping_type == "potential":
            env = PotentialBasedDistanceShaping(env, beta=beta)
        
        # ✅ ADD METRICS WRAPPER (always outermost)
        env = MetricsWrapper(env)
        
        # Create agent
        agent = MonteCarloAgent(
            env=env,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma
        )
        
        # Train
        successes, returns, lengths = agent.train(num_episodes=num_episodes)
        
        env.close()
        
        all_successes.append(successes)
        all_returns.append(returns)
        all_lengths.append(lengths)
    
    # Aggregate results across runs
    all_successes = np.array(all_successes)
    all_returns = np.array(all_returns)
    all_lengths = np.array(all_lengths)
    
    # Print aggregated results
    mean_successes = np.mean(all_successes, axis=0)
    mean_returns = np.mean(all_returns, axis=0)
    mean_lengths = np.mean(all_lengths, axis=0)
    
    print(f"\n=== AGGREGATED RESULTS (across {num_runs} runs) ===")
    print(f"Final success rate (last 100 episodes):")
    print(f"  Mean: {np.mean(mean_successes[-100:])*100:.1f}%")
    print(f"  Std:  {np.std(np.mean(all_successes[:, -100:], axis=1))*100:.1f}%")
    print(f"Average episode length (last 100 episodes):")
    print(f"  Mean: {np.mean(mean_lengths[-100:]):.1f}")
    print(f"  Std:  {np.std(np.mean(all_lengths[:, -100:], axis=1)):.1f}")
    print(f"Average return (last 100 episodes):")
    print(f"  Mean: {np.mean(mean_returns[-100:]):.2f}")
    print(f"  Std:  {np.std(np.mean(all_returns[:, -100:], axis=1)):.2f}")
    
    # Plot learning curves with mean and std across runs
    window = 50
    
    plt.figure(figsize=(16, 4))
    
    # Success rate plot
    plt.subplot(1, 3, 1)
    for run in range(num_runs):
        smoothed = np.convolve(all_successes[run], np.ones(window)/window, mode='valid')
        plt.plot(smoothed, alpha=0.3, color='blue')
    
    # Plot mean
    mean_smoothed = np.mean([np.convolve(all_successes[run], np.ones(window)/window, mode='valid') 
                             for run in range(num_runs)], axis=0)
    std_smoothed = np.std([np.convolve(all_successes[run], np.ones(window)/window, mode='valid') 
                           for run in range(num_runs)], axis=0)
    
    plt.plot(mean_smoothed, color='blue', linewidth=2, label=f'Mean (n={num_runs})')
    plt.fill_between(range(len(mean_smoothed)), 
                     mean_smoothed - std_smoothed, 
                     mean_smoothed + std_smoothed, 
                     alpha=0.2, color='blue')
    plt.title('MC: Success Rate (50-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True)
    
    # Episode length plot
    plt.subplot(1, 3, 2)
    mean_lengths_plot = np.mean(all_lengths, axis=0)
    std_lengths_plot = np.std(all_lengths, axis=0)
    
    plt.plot(mean_lengths_plot, color='green', linewidth=2, label=f'Mean (n={num_runs})')
    plt.fill_between(range(len(mean_lengths_plot)),
                     mean_lengths_plot - std_lengths_plot,
                     mean_lengths_plot + std_lengths_plot,
                     alpha=0.2, color='green')
    plt.title('MC: Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)
    
    # Policy matrix visualization
    plt.subplot(1, 3, 3)
    
    # Recreate environment to get the grid description
    temp_env = create_frozenlake_env(
        grid_size=grid_size,
        hole_density=hole_density,
        is_slippery=is_slippery,
        seed=seed
    )
    
    desc = temp_env.unwrapped.desc
    temp_env.close()
    
    # Access Q-table directly for better policy visualization
    q_table = agent.q_table
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    # Color map: 0=normal, 1=hole, 2=goal, 3=start
    color_matrix = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            cell = desc[i][j]
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            
            if cell == 'H':
                color_matrix[i, j] = 1  # Hole
            elif cell == 'G':
                color_matrix[i, j] = 2  # Goal
            elif cell == 'S':
                color_matrix[i, j] = 3  # Start
    
    # Plot the policy matrix
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'gold', 'lightblue'])
    plt.imshow(color_matrix, cmap=cmap, alpha=0.6)
    
    # Add arrows or ? for policy
    tolerance = 1e-6  # Tolerance for considering Q-values as equal
    
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            cell = desc[i][j]
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            
            if cell != 'H' and cell != 'G':
                # Check if state was visited
                if state in q_table:
                    q_values = q_table[state]
                    max_q = np.max(q_values)
                    
                    # Check if all Q-values are zero (unvisited)
                    if np.allclose(q_values, 0.0, atol=tolerance):
                        # Unvisited state
                        plt.text(j, i, '?', ha='center', va='center',
                                fontsize=16, fontweight='bold', color='gray')
                    else:
                        # Check for ties (multiple actions with max Q-value)
                        max_actions = np.where(np.abs(q_values - max_q) < tolerance)[0]
                        
                        if len(max_actions) > 1:
                            # Tie between multiple actions
                            plt.text(j, i, '?', ha='center', va='center',
                                    fontsize=16, fontweight='bold', color='orange')
                        else:
                            # Clear winner
                            action = max_actions[0]
                            plt.text(j, i, arrows[action], ha='center', va='center',
                                    fontsize=16, fontweight='bold', color='darkblue')
                else:
                    # State never visited
                    plt.text(j, i, '?', ha='center', va='center',
                            fontsize=16, fontweight='bold', color='gray')
    
    plt.title('MC: Learned Policy')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig('mc_test_results.png', dpi=150)
    print("\nPlot saved as 'mc_test_results.png'")
    plt.show()
    
    env.close()
    return agent, grid_size, hole_density, is_slippery, seed


def test_sarsa(hyperparams=None):
    """Test SARSA agent.
    
    Args:
        hyperparams (dict): Hyperparameter dictionary. If None, uses COMMON_HYPERPARAMS.
    """
    if hyperparams is None:
        hyperparams = COMMON_HYPERPARAMS
    
    # Extract hyperparameters
    grid_size = hyperparams['grid_size']
    hole_density = hyperparams['hole_density']
    is_slippery = hyperparams['is_slippery']
    seed = hyperparams['seed']
    epsilon = hyperparams['epsilon']
    alpha = hyperparams['alpha']
    gamma = hyperparams['gamma']
    num_episodes = hyperparams['num_episodes']
    num_runs = hyperparams['num_runs']
    shaping_type = hyperparams['shaping_type']
    step_cost = hyperparams['step_cost']
    beta = hyperparams['beta']
    print("\n" + "="*60)
    print("TESTING SARSA")
    print("="*60)
    

    
    # Print hyperparameters
    print("\nSARSA HYPERPARAMETERS:")
    print("-" * 40)
    print(f"Environment:")
    print(f"  - Grid Size: {grid_size}x{grid_size}")
    print(f"  - Hole Density: {hole_density}")
    print(f"  - Slippery: {is_slippery}")
    print(f"  - Seed: {seed}")
    print(f"\nAgent (SARSA):")
    print(f"  - Epsilon: {epsilon}")
    print(f"  - Alpha: {alpha}")
    print(f"  - Gamma: {gamma}")
    print(f"\nTraining:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Runs: {num_runs}")
    print(f"\nReward Shaping:")
    if shaping_type == "step":
        print(f"  - Type: Step Cost")
        print(f"  - Step Cost: {step_cost}")
    elif shaping_type == "potential":
        print(f"  - Type: Potential-Based Distance")
        print(f"  - Beta: {beta}")
    else:
        print(f"  - Type: None (baseline)")
    print("-" * 40)
    
    # Run multiple experiments with different seeds
    all_successes = []
    all_returns = []
    all_lengths = []
    
    for run in range(num_runs):
        print(f"\n[SARSA] Run {run + 1}/{num_runs}")
        
        # Create environment with run-specific seed
        env = create_frozenlake_env(
            grid_size=grid_size,
            hole_density=hole_density,
            is_slippery=is_slippery,
            seed=seed + run  # Different seed for each run
        )
        
        # Apply reward shaping (if specified)
        if shaping_type == "step":
            env = StepCostShaping(env, step_cost=step_cost)
        elif shaping_type == "potential":
            env = PotentialBasedDistanceShaping(env, beta=beta)
        
        # ✅ ADD METRICS WRAPPER (always outermost)
        env = MetricsWrapper(env)
        
        # Create agent
        agent = SARSAAgent(
            env=env,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma
        )
        
        # Train
        successes, returns, lengths = agent.train(num_episodes=num_episodes)
        
        env.close()
        
        all_successes.append(successes)
        all_returns.append(returns)
        all_lengths.append(lengths)
    
    # Aggregate results across runs
    all_successes = np.array(all_successes)
    all_returns = np.array(all_returns)
    all_lengths = np.array(all_lengths)
    
    # Print aggregated results
    mean_successes = np.mean(all_successes, axis=0)
    mean_returns = np.mean(all_returns, axis=0)
    mean_lengths = np.mean(all_lengths, axis=0)
    
    print(f"\n=== AGGREGATED RESULTS (across {num_runs} runs) ===")
    print(f"Final success rate (last 100 episodes):")
    print(f"  Mean: {np.mean(mean_successes[-100:])*100:.1f}%")
    print(f"  Std:  {np.std(np.mean(all_successes[:, -100:], axis=1))*100:.1f}%")
    print(f"Average episode length (last 100 episodes):")
    print(f"  Mean: {np.mean(mean_lengths[-100:]):.1f}")
    print(f"  Std:  {np.std(np.mean(all_lengths[:, -100:], axis=1)):.1f}")
    print(f"Average return (last 100 episodes):")
    print(f"  Mean: {np.mean(mean_returns[-100:]):.2f}")
    print(f"  Std:  {np.std(np.mean(all_returns[:, -100:], axis=1)):.2f}")
    
    # Plot learning curves with mean and std across runs
    window = 50
    
    plt.figure(figsize=(16, 4))
    
    # Success rate plot
    plt.subplot(1, 3, 1)
    for run in range(num_runs):
        smoothed = np.convolve(all_successes[run], np.ones(window)/window, mode='valid')
        plt.plot(smoothed, alpha=0.3, color='orange')
    
    # Plot mean
    mean_smoothed = np.mean([np.convolve(all_successes[run], np.ones(window)/window, mode='valid') 
                             for run in range(num_runs)], axis=0)
    std_smoothed = np.std([np.convolve(all_successes[run], np.ones(window)/window, mode='valid') 
                           for run in range(num_runs)], axis=0)
    
    plt.plot(mean_smoothed, color='orange', linewidth=2, label=f'Mean (n={num_runs})')
    plt.fill_between(range(len(mean_smoothed)), 
                     mean_smoothed - std_smoothed, 
                     mean_smoothed + std_smoothed, 
                     alpha=0.2, color='orange')
    plt.title('SARSA: Success Rate (50-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True)
    
    # Episode length plot
    plt.subplot(1, 3, 2)
    mean_lengths_plot = np.mean(all_lengths, axis=0)
    std_lengths_plot = np.std(all_lengths, axis=0)
    
    plt.plot(mean_lengths_plot, color='red', linewidth=2, label=f'Mean (n={num_runs})')
    plt.fill_between(range(len(mean_lengths_plot)),
                     mean_lengths_plot - std_lengths_plot,
                     mean_lengths_plot + std_lengths_plot,
                     alpha=0.2, color='red')
    plt.title('SARSA: Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)
    
    # Policy matrix visualization
    plt.subplot(1, 3, 3)
    
    # Recreate environment to get the grid description
    temp_env = create_frozenlake_env(
        grid_size=grid_size,
        hole_density=hole_density,
        is_slippery=is_slippery,
        seed=seed
    )
    
    desc = temp_env.unwrapped.desc
    temp_env.close()
    
    # Access Q-table directly for better policy visualization
    q_table = agent.q_table
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    # Color map: 0=normal, 1=hole, 2=goal, 3=start
    color_matrix = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            cell = desc[i][j]
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            
            if cell == 'H':
                color_matrix[i, j] = 1  # Hole
            elif cell == 'G':
                color_matrix[i, j] = 2  # Goal
            elif cell == 'S':
                color_matrix[i, j] = 3  # Start
    
    # Plot the policy matrix
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'gold', 'lightblue'])
    plt.imshow(color_matrix, cmap=cmap, alpha=0.6)
    
    # Add arrows or ? for policy
    tolerance = 1e-6  # Tolerance for considering Q-values as equal
    
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            cell = desc[i][j]
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            
            if cell != 'H' and cell != 'G':
                # Check if state was visited
                if state in q_table:
                    q_values = q_table[state]
                    max_q = np.max(q_values)
                    
                    # Check if all Q-values are zero (unvisited)
                    if np.allclose(q_values, 0.0, atol=tolerance):
                        # Unvisited state
                        plt.text(j, i, '?', ha='center', va='center',
                                fontsize=16, fontweight='bold', color='gray')
                    else:
                        # Check for ties (multiple actions with max Q-value)
                        max_actions = np.where(np.abs(q_values - max_q) < tolerance)[0]
                        
                        if len(max_actions) > 1:
                            # Tie between multiple actions
                            plt.text(j, i, '?', ha='center', va='center',
                                    fontsize=16, fontweight='bold', color='orange')
                        else:
                            # Clear winner
                            action = max_actions[0]
                            plt.text(j, i, arrows[action], ha='center', va='center',
                                    fontsize=16, fontweight='bold', color='darkred')
                else:
                    # State never visited
                    plt.text(j, i, '?', ha='center', va='center',
                            fontsize=16, fontweight='bold', color='gray')
    
    plt.title('SARSA: Learned Policy')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig('sarsa_test_results.png', dpi=150)
    print("\nPlot saved as 'sarsa_test_results.png'")
    plt.show()
    
    env.close()
    return agent, grid_size, hole_density, is_slippery, seed


def visualize_policy(env, agent, agent_name="Agent"):
    """Simple policy visualization."""
    print(f"\n{agent_name} Policy:")
    print("="*40)
    
    policy = agent.get_policy()
    desc = env.unwrapped.desc
    grid_size = len(desc)
    
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            state = i * grid_size + j
            cell = desc[i][j]
            
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            
            if cell == 'S':
                if state in policy:
                    row.append(f'S{arrows[policy[state]]}')
                else:
                    row.append('S ')
            elif cell == 'G':
                row.append('G ')
            elif cell == 'H':
                row.append('H ')
            else:
                if state in policy:
                    row.append(arrows[policy[state]] + ' ')
                else:
                    row.append('? ')
        
        print(' '.join(row))


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AGENT TESTING SUITE")
    print("="*60)
    print("\nThis will test both MC and SARSA")
    print("Expected: Both should achieve >80% success rate")
    print("="*60)
    
    # Test Monte Carlo
    mc_agent, mc_grid_size, mc_hole_density, mc_is_slippery, mc_seed = test_monte_carlo()
    
    # Recreate env for visualization
    env = create_frozenlake_env(grid_size=mc_grid_size, hole_density=mc_hole_density, 
                                is_slippery=mc_is_slippery, seed=mc_seed)
    visualize_policy(env, mc_agent, "Monte Carlo")
    env.close()
    
    # Test SARSA
    sarsa_agent, sarsa_grid_size, sarsa_hole_density, sarsa_is_slippery, sarsa_seed = test_sarsa()
    
    # Recreate env for visualization
    env = create_frozenlake_env(grid_size=sarsa_grid_size, hole_density=sarsa_hole_density, 
                                is_slippery=sarsa_is_slippery, seed=sarsa_seed)
    visualize_policy(env, sarsa_agent, "SARSA")
    env.close()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)