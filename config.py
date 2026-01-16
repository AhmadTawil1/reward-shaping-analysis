"""
Central Configuration File for Reward Shaping Project

Change all hyperparameters here in ONE place!
All experiment scripts will use these settings.
"""

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

GRID_SIZE = 6         # Grid size (6, 7, 8, etc.)
HOLE_DENSITY = 0.1    # Hole density (0.10 - 0.20)
IS_SLIPPERY = True    # Slippery dynamics (True = stochastic)
SUCCESS_RATE = 0.7    # Action success rate when slippery (0.33 = default, 0.7 = easier)

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# For quick testing
TEST_MODE = True      # Set to True for quick tests
TEST_EPISODES = 1000    # Episodes when TEST_MODE = True
TEST_RUNS = 5           # Runs when TEST_MODE = True

# For final experiments
FINAL_EPISODES = 5000   # Episodes for final experiments
FINAL_RUNS = 20         # Runs for final experiments (recommended: 20-50)

# ============================================================
# ALGORITHM HYPERPARAMETERS
# ============================================================

# Exploration & Learning
EPSILON = 0.1           # Exploration rate (0.0 - 1.0)
ALPHA = 0.1             # Learning rate (0.0 - 1.0)
GAMMA = 1.0             # Discount factor (REQUIRED: 1.0 for this project)

# Advanced (optional)
EPSILON_DECAY = 0.995     # Epsilon decay per episode (1.0 = no decay)
EPSILON_MIN = 0.0       # Minimum epsilon value
INITIAL_Q_VALUE = 0.0   # Optimistic initialization (0.0 = neutral)

# ============================================================
# REWARD SHAPING PARAMETERS
# ============================================================

# Step-Cost Shaping
STEP_COST = -0.01       # Penalty per step (negative value)

# Potential-Based Shaping
BETA_POTENTIAL = 1.0    # Shaping strength for potential-based

# Safety-Based Shaping (Custom #1)
BETA_SAFETY = 0.1       # Shaping strength for safety-based
                        # Recommended: 0.1 for SARSA

# Exploration Bonus (Custom #2)
BETA_EXPLORATION = 0.05 # Shaping strength for exploration bonus
                        # Recommended: 0.02-0.05 for MC

# ============================================================
# HYPERPARAMETER SWEEP RANGES
# ============================================================

# For hyperparameter_sweep.py
SWEEP_EPISODES = 2000   # Episodes per sweep experiment
SWEEP_RUNS = 10         # Runs per sweep experiment

# Epsilon sweep values
EPSILON_SWEEP = [0.05, 0.1, 0.2]

# Alpha sweep values
ALPHA_SWEEP = [0.05, 0.1, 0.2]

# Step-cost sweep values
STEP_COST_SWEEP = [-0.001, -0.005, -0.01, -0.05]

# Beta sweep values
BETA_POTENTIAL_SWEEP = [0.5, 1.0, 2.0]
BETA_SAFETY_SWEEP = [0.05, 0.1, 0.2, 0.5]
BETA_EXPLORATION_SWEEP = [0.01, 0.05, 0.1, 0.2]

# Gamma sweep values (educational purposes - project requires 1.0)
GAMMA_SWEEP = [0.95, 0.98, 0.99, 0.995, 1.0]

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
        'grid_size': GRID_SIZE,
        'hole_density': HOLE_DENSITY,
        'is_slippery': IS_SLIPPERY,
        'success_rate': SUCCESS_RATE,
        'num_episodes': TEST_EPISODES,
        'num_runs': TEST_RUNS,
        'epsilon': EPSILON,
        'alpha': ALPHA,
        'gamma': GAMMA,
    }

def get_final_config():
    """Get configuration for final experiments (slow but proper statistics)."""
    return {
        'grid_size': GRID_SIZE,
        'hole_density': HOLE_DENSITY,
        'is_slippery': IS_SLIPPERY,
        'success_rate': SUCCESS_RATE,
        'num_episodes': FINAL_EPISODES,
        'num_runs': FINAL_RUNS,
        'epsilon': EPSILON,
        'alpha': ALPHA,
        'gamma': GAMMA,
    }

def get_shaping_params(shaping_type):
    """Get parameters for specific shaping type."""
    if shaping_type == 'step':
        return {'step_cost': STEP_COST}
    elif shaping_type == 'potential':
        return {'beta': BETA_POTENTIAL}
    elif shaping_type == 'safety':
        return {'beta': BETA_SAFETY}
    elif shaping_type == 'exploration':
        return {'beta': BETA_EXPLORATION}
    else:
        return {}

# ============================================================
# PRINT CURRENT CONFIGURATION
# ============================================================

def print_config():
    """Print current configuration for verification."""
    print("=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    print(f"\nEnvironment:")
    print(f"  Grid Size: {GRID_SIZE}×{GRID_SIZE} ({GRID_SIZE**2} cells)")
    print(f"  Hole Density: {HOLE_DENSITY} (~{round(GRID_SIZE**2 * HOLE_DENSITY)} holes)")
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
    
    print(f"\nAlgorithm Hyperparameters:")
    print(f"  Epsilon (ε): {EPSILON}")
    print(f"  Alpha (α): {ALPHA}")
    print(f"  Gamma (γ): {GAMMA}")
    
    print(f"\nReward Shaping Parameters:")
    print(f"  Step-Cost: c = {STEP_COST}")
    print(f"  Potential-Based: β = {BETA_POTENTIAL}")
    print(f"  Safety-Based: β = {BETA_SAFETY}")
    print(f"  Exploration Bonus: β = {BETA_EXPLORATION}")
    
    print("=" * 70)

if __name__ == "__main__":
    # Print configuration when run directly
    print_config()