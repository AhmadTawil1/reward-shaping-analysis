"""
Central Configuration File for Reward Shaping Project

Change all hyperparameters here in ONE place!
All experiment scripts will use these settings.
"""

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

GRID_ROWS = 6         # Number of rows (6, 7, 8, etc.)
GRID_COLS = 6         # Number of columns (6, 7, 8, etc.)
HOLE_DENSITY = 0.15    # Hole density (0.10 - 0.20)
IS_SLIPPERY = True    # Slippery dynamics (True = stochastic)
SUCCESS_RATE = 0.7    # Action success rate when slippery (0.33 = default, 0.7 = easier)

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# For quick testing
TEST_MODE = False    # Set to True for quick tests
TEST_EPISODES = 1000    # Episodes when TEST_MODE = True
TEST_RUNS = 5          # Runs when TEST_MODE = True

# For final experiments
FINAL_EPISODES = 10000   # Episodes for final experiments
FINAL_RUNS = 20         # Runs for final experiments (recommended: 20-50)

# ============================================================
# ALGORITHM HYPERPARAMETERS
# ============================================================

GAMMA = 1.0             # Discount factor (REQUIRED: 1.0 for this project)
INITIAL_Q_VALUE = 0.0   # Optimistic initialization (0.0 = neutral)
RANDOM_SEED = 242       # Global random seed for reproducibility


# ============================================================
# OPTIMIZED HYPERPARAMETERS (from grid search)
# ============================================================
# These are the best hyperparameters found through systematic search
# Use get_optimized_hyperparams() and get_shaping_params() for final experiments

# Monte Carlo - Optimized per shaping method
MC_HYPERPARAMS = {
    'baseline': {'epsilon': 0.05, 'alpha': 0.05},
    'step': {'epsilon': 0.05, 'alpha': 0.05},
    'potential': {'epsilon': 0.05, 'alpha': 0.05},
    'safety': {'epsilon': 0.05, 'alpha': 0.05},
    'exploration': {'epsilon': 0.1, 'alpha': 0.05}
}

# SARSA - Optimized per shaping method
SARSA_HYPERPARAMS = {
    'baseline': {'epsilon': 0.1, 'alpha': 0.2},
    'step': {'epsilon': 0.1, 'alpha': 0.2},
    'potential': {'epsilon': 0.05, 'alpha': 0.1},
    'safety': {'epsilon': 0.05, 'alpha': 0.2},
    'exploration': {'epsilon': 0.05, 'alpha': 0.2}
}

# Optimized shaping parameters per algorithm
STEP_COST_MC = -0.01      # Best for Monte Carlo
STEP_COST_SARSA = -0.005  # Best for SARSA

BETA_POTENTIAL_MC = 0.5    # Best for Monte Carlo
BETA_POTENTIAL_SARSA = 1.0 # Best for SARSA

BETA_SAFETY_MC = 0.05      # Best for Monte Carlo
BETA_SAFETY_SARSA = 0.05   # Best for SARSA

BETA_EXPLORATION_MC = 0.05   # Best for Monte Carlo
BETA_EXPLORATION_SARSA = 0.02 # Best for SARSA

# ============================================================
# HYPERPARAMETER SWEEP RANGES
# ============================================================

# For hyperparameter_sweep.py
SWEEP_EPISODES = 10000   # Episodes per sweep experiment
SWEEP_RUNS = 10         # Runs per sweep experiment

# Epsilon sweep values
EPSILON_SWEEP = [0.05, 0.2, 0.5]

# Alpha sweep values
ALPHA_SWEEP = [0.05, 0.2, 0.5]

# # Step-cost sweep values
# STEP_COST_SWEEP = [-0.001, -0.005, -0.01, -0.05]

# # Beta sweep values
# BETA_POTENTIAL_SWEEP = [0.5, 1.0, 2.0]
# BETA_SAFETY_SWEEP = [0.05, 0.1, 0.2, 0.5]
# BETA_EXPLORATION_SWEEP = [0.01, 0.05, 0.1, 0.2]

# # Gamma sweep values (educational purposes - project requires 1.0)
# GAMMA_SWEEP = [0.95, 0.98, 0.99, 0.995, 1.0]

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

RESULTS_DIR = 'results'         # Directory for saving plots
REPORT_FIGURES_DIR = 'report_figures'  # Directory for report figures

# ============================================================
# QUICK PRESETS
# ============================================================

def get_test_config():
    """Get configuration for quick testing (fast)."""
    return {
        'grid_rows': GRID_ROWS,
        'grid_cols': GRID_COLS,
        'hole_density': HOLE_DENSITY,
        'is_slippery': IS_SLIPPERY,
        'success_rate': SUCCESS_RATE,
        'num_episodes': TEST_EPISODES,
        'num_runs': TEST_RUNS,
        'gamma': GAMMA,
    }

def get_final_config():
    """Get configuration for final experiments (slow but proper statistics)."""
    return {
        'grid_rows': GRID_ROWS,
        'grid_cols': GRID_COLS,
        'hole_density': HOLE_DENSITY,
        'is_slippery': IS_SLIPPERY,
        'success_rate': SUCCESS_RATE,
        'num_episodes': FINAL_EPISODES,
        'num_runs': FINAL_RUNS,
        'gamma': GAMMA,
    }

def get_shaping_params(shaping_type, algorithm='MC'):
    """
    Get optimized parameters for specific shaping type and algorithm.
    
    Args:
        shaping_type (str): 'step', 'potential', 'safety', 'exploration', or None
        algorithm (str): 'MC' or 'SARSA'
    
    Returns:
        dict: Shaping parameters optimized for the algorithm
    """
    if shaping_type == 'step':
        if algorithm == 'MC':
            return {'step_cost': STEP_COST_MC}
        else:  # SARSA
            return {'step_cost': STEP_COST_SARSA}
    elif shaping_type == 'potential':
        if algorithm == 'MC':
            return {'beta': BETA_POTENTIAL_MC}
        else:  # SARSA
            return {'beta': BETA_POTENTIAL_SARSA}
    elif shaping_type == 'safety':
        if algorithm == 'MC':
            return {'beta': BETA_SAFETY_MC}
        else:  # SARSA
            return {'beta': BETA_SAFETY_SARSA}
    elif shaping_type == 'exploration':
        if algorithm == 'MC':
            return {'beta': BETA_EXPLORATION_MC}
        else:  # SARSA
            return {'beta': BETA_EXPLORATION_SARSA}
    else:
        return {}


def get_optimized_hyperparams(algorithm, shaping_type):
    """
    Get optimized epsilon and alpha for specific algorithm and shaping method.
    
    Args:
        algorithm (str): 'MC' or 'SARSA'
        shaping_type (str): 'baseline', 'step', 'potential', 'safety', 'exploration', or None
    
    Returns:
        dict: {'epsilon': float, 'alpha': float}
    """
    # Map None to 'baseline'
    if shaping_type is None:
        shaping_type = 'baseline'
    
    if algorithm == 'MC':
        return MC_HYPERPARAMS.get(shaping_type, MC_HYPERPARAMS['baseline'])
    else:  # SARSA
        return SARSA_HYPERPARAMS.get(shaping_type, SARSA_HYPERPARAMS['baseline'])

# ============================================================
# PRINT CURRENT CONFIGURATION
# ============================================================

def print_config():
    """Print current configuration for verification."""
    print("=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    print(f"\nEnvironment:")
    print(f"  Grid Size: {GRID_ROWS}×{GRID_COLS} ({GRID_ROWS * GRID_COLS} cells)")
    print(f"  Hole Density: {HOLE_DENSITY} (~{round(GRID_ROWS * GRID_COLS * HOLE_DENSITY)} holes)")
    print(f"  Slippery: {IS_SLIPPERY}")
    if IS_SLIPPERY:
        print(f"  Success Rate: {SUCCESS_RATE} (default: 0.33)")
    
    print(f"\nTraining:")
    print(f"  Mode: {'TEST' if TEST_MODE else 'FINAL'}")
    if TEST_MODE:
        print(f"  Episodes: {TEST_EPISODES}")
        print(f"  Runs: {TEST_RUNS}")
    else:
        print(f"  Episodes: {FINAL_EPISODES}")
        print(f"  Runs: {FINAL_RUNS}")
    
    print(f"\nGlobal Parameters:")
    print(f"  Gamma (γ): {GAMMA}")
    print(f"  Random Seed: {RANDOM_SEED if RANDOM_SEED is not None else 'None (random)'}")
    
    print(f"\nOptimized Hyperparameters (use get_optimized_hyperparams()):")
    print(f"  MC Baseline: ε={MC_HYPERPARAMS['baseline']['epsilon']}, α={MC_HYPERPARAMS['baseline']['alpha']}")
    print(f"  SARSA Baseline: ε={SARSA_HYPERPARAMS['baseline']['epsilon']}, α={SARSA_HYPERPARAMS['baseline']['alpha']}")
    
    print("=" * 70)

if __name__ == "__main__":
    # Print configuration when run directly
    print_config()