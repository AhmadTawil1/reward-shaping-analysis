# Reward Shaping Analysis for Reinforcement Learning

A comprehensive analysis framework for evaluating different reward shaping techniques in reinforcement learning, specifically applied to the FrozenLake environment using Monte Carlo and SARSA algorithms.

## 📋 Overview

This project implements and compares five different reward shaping methods to improve learning efficiency in reinforcement learning agents. The framework includes parallel execution, extensive visualization tools, and automated hyperparameter optimization.

### Key Features

- **5 Reward Shaping Methods**: Baseline, Step-Cost, Potential-Based, Safety-Based, and Exploration Bonus
- **2 RL Algorithms**: Monte Carlo and SARSA
- **Parallel Execution**: 6-9x faster training using multi-core processing
- **Comprehensive Visualization**: Policy evolution, learning curves, and agent demonstrations
- **Automated Analysis**: Statistical comparison with 95% confidence intervals
- **Reproducible Results**: Controlled random seeds for consistent experiments

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

```bash
git clone <repository-url>
cd reward-shaping-analysis
pip install -r requirements.txt
```

### Basic Usage

Run the complete comparison (recommended):

```bash
python compare_methods_parallel.py
```

This will:
1. Train 5 reward shaping methods with Monte Carlo (20 runs each, parallel)
2. Train 5 reward shaping methods with SARSA (20 runs each, parallel)
3. Generate comparison plots and statistics
4. Visualize optimal policies
5. Save all results to `results/` directory

**Expected runtime:** 5-10 minutes (with parallel execution)

## 📊 Project Structure

```
reward-shaping-analysis/
├── algorithms/              # RL algorithm implementations
│   ├── monte_carlo.py      # Monte Carlo agent
│   └── sarsa.py            # SARSA agent
├── env/                    # Environment and wrappers
│   ├── frozenlake_env.py   # Custom FrozenLake environment
│   ├── reward_shaping.py   # Reward shaping implementations
│   └── metrics_wrapper.py  # Metrics tracking wrapper
├── experiments/            # Experiment runners
│   ├── run_monte_carlo_parallel.py
│   ├── run_sarsa_parallel.py
│   └── find_best_hyperparameters.py
├── utils/                  # Utility functions
│   ├── visualization.py    # Policy visualization tools
│   ├── agent_rendering.py  # Animated GIF generation
│   └── metrics.py          # Performance metrics
├── results/                # Generated plots and data
├── config.py               # Configuration and hyperparameters
├── compare_methods_parallel.py  # Main comparison script
└── test_agent_rendering.py      # Policy evolution visualization
```

## 🎯 Reward Shaping Methods

### 1. Baseline (No Shaping)
Standard reward: +1 for goal, 0 otherwise

### 2. Step-Cost Shaping
Penalizes each step to encourage efficiency
- Formula: `r_shaped = r_original - step_cost`

### 3. Potential-Based Shaping
Uses distance-based potential function
- Formula: `r_shaped = r_original + γΦ(s') - Φ(s)`
- Φ(s) = negative Manhattan distance to goal

### 4. Safety-Based Shaping (Custom)
Rewards safe exploration, penalizes holes
- Bonus for avoiding holes
- Penalty for approaching dangerous states

### 5. Exploration Bonus (Custom)
Encourages visiting new states
- Bonus for first-time state visits
- Decays over time

## 📈 Results and Visualization

### Generated Outputs

After running `compare_methods_parallel.py`, you'll get:

**Plots:**
- `mc_all_methods_throughput_parallel.png` - Monte Carlo throughput comparison
- `mc_all_methods_returns_parallel.png` - Monte Carlo returns comparison
- `sarsa_all_methods_throughput_parallel.png` - SARSA throughput comparison
- `sarsa_all_methods_returns_parallel.png` - SARSA returns comparison
- `final_mc_vs_sarsa_throughput_parallel.png` - Best method comparison
- `final_mc_vs_sarsa_returns_parallel.png` - Best method returns
- `best_policies_comparison_parallel.png` - Policy visualization

**Console Output:**
- Summary tables with mean ± std for each method
- Statistical significance tests
- Best method identification
- Performance improvements over baseline

### Interpreting Results

**Throughput**: Cumulative success rate over episodes
- Formula: `throughput[t] = (total_successes_up_to_t) / t`
- Higher is better (closer to 100%)
- Shows learning efficiency

**Returns**: Real (unshaped) rewards received
- Shows actual task performance
- Not affected by shaping rewards

**95% Confidence Intervals**: Shown as shaded areas in plots
- Calculated as: `mean ± 1.96 × (std / √20)`
- Non-overlapping CIs indicate statistical significance

## 🔬 Advanced Usage

### Policy Evolution Visualization

Generate policy snapshots at episodes 1, 100, 1000, 10000:

```bash
python test_agent_rendering.py
```

Outputs:
- `policy_evolution.png` - 2×2 grid showing learning progress
- `sarsa_safety_best.gif` - Animated demonstration of best agent

### Beta Sensitivity Analysis

Analyze how the beta parameter affects Safety-Based and Exploration Bonus shaping:

```bash
python beta_comparison.py
```

This compares different beta values (0.01, 0.05, 0.1, 0.5) to find the optimal shaping strength.

Outputs:
- `mc_beta_comparison.png` - Monte Carlo beta sensitivity
- `sarsa_beta_comparison.png` - SARSA beta sensitivity
- Console tables showing performance for each beta value

**Beta Parameter:**
- Controls the strength of reward shaping
- Lower values (0.01): Subtle shaping, closer to original rewards
- Higher values (0.5): Strong shaping, may dominate original rewards
- Optimal value balances learning speed and final performance

### Hyperparameter Sensitivity Analysis

Visualize how epsilon and alpha affect learning:

```bash
python hyperparameters_sweep.py
```

This runs a grid search over:
- **Epsilon values**: 0.01, 0.05, 0.1, 0.2 (exploration rate)
- **Alpha values**: 0.01, 0.05, 0.1, 0.2 (learning rate, SARSA only)

Outputs:
- `epsilon_sensitivity.png` - Individual subplots for each epsilon value
- `alpha_sensitivity.png` - Individual subplots for each alpha value (SARSA)
- Shows returns over episodes for each parameter combination

**Use cases:**
- Finding optimal hyperparameters for your environment
- Understanding exploration-exploitation tradeoff
- Tuning learning rate for stability vs. speed

### Custom Experiments

Modify `config.py` to adjust:
- Grid size: `GRID_ROWS`, `GRID_COLS`
- Hole density: `HOLE_DENSITY`
- Episodes: `FINAL_EPISODES`
- Number of runs: `FINAL_RUNS`
- Hyperparameters: `OPTIMIZED_HYPERPARAMS`

### Hyperparameter Tuning

Find optimal hyperparameters:

```bash
python experiments/find_best_hyperparameters.py
```

## 🧪 Experimental Setup

### Environment Configuration
- **Grid Size**: 6×6
- **Hole Density**: 15%
- **Slippery**: Yes (stochastic transitions)
- **Success Rate**: 70% (action success probability)
- **Random Seed**: 242 (for reproducibility)

### Training Configuration
- **Episodes per run**: 10,000
- **Independent runs**: 20
- **Execution**: Parallel (all CPU cores)
- **Algorithms**: Monte Carlo, SARSA

### Hyperparameters (Optimized)
```python
Monte Carlo:
  - epsilon: 0.05 (exploration rate)
  - gamma: 1.0 (discount factor)

SARSA:
  - epsilon: 0.05
  - alpha: 0.05 (learning rate)
  - gamma: 1.0
```

## 📊 Performance Metrics

### Throughput
- **Definition**: Cumulative success rate
- **Calculation**: `(successes_so_far) / (episodes_so_far)`
- **Range**: 0-100%
- **Interpretation**: Higher = better learning efficiency

### Real Return
- **Definition**: Actual rewards (without shaping)
- **Calculation**: Sum of original environment rewards
- **Interpretation**: True task performance

### Episode Length
- **Definition**: Steps taken per episode
- **Interpretation**: Shorter = more efficient policy

## 🔧 Configuration

All configuration is centralized in `config.py`:

```python
# Environment
GRID_ROWS = 6
GRID_COLS = 6
HOLE_DENSITY = 0.15
IS_SLIPPERY = True
SUCCESS_RATE = 0.7

# Training
FINAL_EPISODES = 10000
FINAL_RUNS = 20
RANDOM_SEED = 242

# Hyperparameters
OPTIMIZED_HYPERPARAMS = {
    'MC': {'safety': {'epsilon': 0.05, 'gamma': 1.0}},
    'SARSA': {'safety': {'epsilon': 0.05, 'alpha': 0.05, 'gamma': 1.0}}
}
```

## 🎨 Visualization Tools

### Static Policy Visualization
```python
from utils.visualization import visualize_grid

visualize_grid(env, policy, value_function, 
               title="Learned Policy",
               save_path="policy.png")
```

### Animated Agent Demonstration
```python
from utils.agent_rendering import create_agent_demo

create_agent_demo(agent, 
                 save_path="agent_demo.gif",
                 fps=2)
```

### Learning Curves
```python
from utils.visualization import compare_policies

compare_policies(env, policies_dict, 
                value_functions_dict,
                title="Policy Comparison")
```

## 📝 Key Findings

Based on experimental results:

1. **Safety-Based Shaping** typically achieves highest throughput
2. **Exploration Bonus** shows fastest early learning
3. **Potential-Based** provides consistent improvement over baseline
4. **Step-Cost** encourages efficient paths
5. **SARSA** generally outperforms Monte Carlo in this environment

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional reward shaping methods
- More RL algorithms (Q-Learning, DQN, etc.)
- Different environments
- Advanced visualization techniques


## 🙏 Acknowledgments

- Built using [Gymnasium](https://gymnasium.farama.org/)
- Parallel execution with [joblib](https://joblib.readthedocs.io/)
- Visualization with [matplotlib](https://matplotlib.org/)



---

## 🚀 Quick Reference

### Most Common Commands

```bash
# Full comparison (recommended)
python compare_methods_parallel.py

# Policy evolution visualization
python test_agent_rendering.py

# Beta sensitivity analysis
python beta_comparison.py

# Hyperparameter sweep
python hyperparameters_sweep.py

# Hyperparameter tuning
python experiments/find_best_hyperparameters.py
```

### Troubleshooting

**Issue**: Tkinter warnings during parallel execution
**Solution**: Add `matplotlib.use('Agg')` before importing pyplot

**Issue**: Out of memory
**Solution**: Reduce `FINAL_RUNS` or `FINAL_EPISODES` in config.py

**Issue**: Slow execution
**Solution**: Ensure parallel version is being used (compare_methods_parallel.py)

---

**Last Updated**: January 2026
