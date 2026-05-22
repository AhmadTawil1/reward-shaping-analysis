"""
Create agent demonstration with policy evolution visualization.

This script:
1. Trains a SARSA Safety-Based agent
2. Captures policy snapshots at episodes 1, 100, 1000, 10000
3. Visualizes the policy evolution in a grid
4. Creates an animated GIF of the final optimal policy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import config
from env.frozenlake_env import create_frozenlake_env
from algorithms.sarsa import SARSAAgent
from env.reward_shaping import SafetyBasedShaping
from env.metrics_wrapper import MetricsWrapper
from utils.agent_rendering import create_agent_demo

print("="*70)
print("AGENT RENDERING WITH POLICY EVOLUTION")
print("="*70)

# Get configuration
cfg = config.get_final_config() if not config.TEST_MODE else config.get_test_config()
hyperparams = config.get_optimized_hyperparams('SARSA', 'safety')
shaping_params = config.get_shaping_params('safety', algorithm='SARSA')

print(f"\n[1/3] Training ONE SARSA Safety-Based agent with snapshots...")
print(f"      Capturing policy snapshots at episodes: 1, 500, 10000")
print(f"      Total episodes: {cfg['num_episodes']}")
print(f"      This will show how ONE agent learns over time")

# Create environment for training
training_env = create_frozenlake_env(
    rows=cfg['grid_rows'],
    cols=cfg['grid_cols'],
    hole_density=cfg['hole_density'],
    is_slippery=cfg['is_slippery'],
    seed=config.RANDOM_SEED,
    success_rate=cfg['success_rate']
)
training_env = SafetyBasedShaping(training_env, beta=shaping_params['beta'])
training_env = MetricsWrapper(training_env)

# Create agent
agent = SARSAAgent(
    env=training_env,
    epsilon=hyperparams['epsilon'],
    alpha=hyperparams['alpha'],
    gamma=cfg['gamma'],
    initial_q_value=config.INITIAL_Q_VALUE
)

# Snapshot episodes
snapshot_episodes = [1, 500, 10000]
policy_snapshots = {}

# Training loop with snapshots
print("   Training progress:")
for episode in range(1, cfg['num_episodes'] + 1):
    state, _ = training_env.reset()
    action = agent.select_action(state)
    done = False
    
    while not done:
        next_state, reward, terminated, truncated, _ = training_env.step(action)
        done = terminated or truncated
        next_action = agent.select_action(next_state)
        
        # SARSA update
        if not done:
            td_target = reward + cfg['gamma'] * agent.q_table[next_state][next_action]
        else:
            td_target = reward
        
        td_error = td_target - agent.q_table[state][action]
        agent.q_table[state][action] += agent.alpha * td_error
        
        state = next_state
        action = next_action
    
    # Capture snapshot
    if episode in snapshot_episodes:
        policy_snapshots[episode] = {
            'policy': agent.get_policy().copy(),
            'value_function': agent.get_value_function().copy()
        }
        print(f"   📸 Episode {episode:5d}: Policy snapshot captured")

training_env.close()
print("   ✅ Training completed!")

# Create results directory
os.makedirs('results/agent_demos', exist_ok=True)

# [2/3] Visualize policy evolution
print(f"\n[2/3] Visualizing policy evolution...")

# Get grid description
temp_env = create_frozenlake_env(
    rows=cfg['grid_rows'],
    cols=cfg['grid_cols'],
    hole_density=cfg['hole_density'],
    is_slippery=cfg['is_slippery'],
    seed=config.RANDOM_SEED,
    success_rate=cfg['success_rate']
)
desc = temp_env.unwrapped.desc
rows, cols = desc.shape
temp_env.close()

# Create figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(cols * 6, rows * 2.5))
if len(snapshot_episodes) == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# Action mapping
action_arrows = {
    0: (-0.25, 0),    # LEFT
    1: (0, -0.25),    # DOWN
    2: (0.25, 0),     # RIGHT
    3: (0, 0.25)      # UP
}

action_names = {
    0: '←',
    1: '↓',
    2: '→',
    3: '↑'
}

# Draw each snapshot
for idx, episode in enumerate(snapshot_episodes):
    ax = axes[idx]
    snapshot = policy_snapshots[episode]
    policy = snapshot['policy']
    value_func = snapshot['value_function']
    
    # Normalize values for color mapping
    if value_func:
        max_val = max(value_func.values())
        min_val = min(value_func.values())
        val_range = max_val - min_val if max_val != min_val else 1.0
    else:
        max_val, min_val, val_range = 0, 0, 1.0
    
    # Draw grid
    for i in range(rows):
        for j in range(cols):
            state = i * cols + j
            cell_type = desc[i, j].decode('utf-8') if isinstance(desc[i, j], bytes) else desc[i, j]
            
            # Determine cell color
            if cell_type == 'S':
                color = 'lightgreen'
                label = 'S'
            elif cell_type == 'G':
                color = 'gold'
                label = 'G'
            elif cell_type == 'H':
                color = 'lightcoral'
                label = 'H'
            else:
                # Color based on value function
                if state in value_func:
                    normalized_val = (value_func[state] - min_val) / val_range
                    color = plt.cm.Blues(0.3 + 0.5 * normalized_val)
                else:
                    color = 'lightblue'
                label = ''
            
            # Draw cell
            rect = plt.Rectangle((j, rows - i - 1), 1, 1,
                                facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add cell label at top
            if label:
                ax.text(j + 0.5, rows - i - 0.2, label,
                       ha='center', va='center', fontsize=18, fontweight='bold')
            
            # Add Q-value at bottom
            if state in value_func and cell_type not in ['H']:
                q_value = value_func[state]
                ax.text(j + 0.5, rows - i - 0.85, f'{q_value:.2f}',
                       ha='center', va='center', fontsize=16,
                       fontweight='bold', color='darkblue',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='none', alpha=0.7))
            
            # Draw policy arrow
            if cell_type not in ['H', 'G'] and state in policy:
                action = policy[state]
                dx, dy = action_arrows[action]
                
                arrow = FancyArrow(j + 0.5, rows - i - 0.4, dx, dy,
                                  width=0.08, head_width=0.15, head_length=0.1,
                                  fc='darkred', ec='darkred', linewidth=1.5)
                ax.add_patch(arrow)
                
                ax.text(j + 0.5 + dx * 1.15, rows - i - 0.4 + dy * 1.15,
                       action_names[action],
                       ha='center', va='center', fontsize=14,
                       fontweight='bold', color='darkred')
    
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Episode {episode}', fontsize=15, fontweight='bold', pad=10)

# Add overall title
fig.suptitle('Policy Evolution: SARSA with Safety-Based Shaping',
            fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

evolution_path = 'results/agent_demos/policy_evolution.png'
plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved policy evolution: {evolution_path}")

plt.show()
plt.close()

# [3/3] Create animated GIF using the trained agent
print(f"\n[3/3] Creating animated GIF with the trained agent...")
print("      → Using the same agent from policy evolution")
print("      → Using Gymnasium's rgb_array rendering")
print("      → Recording until agent reaches goal")

success, steps, attempts = create_agent_demo(
    agent=agent,  # Use the trained agent
    grid_rows=cfg['grid_rows'],
    grid_cols=cfg['grid_cols'],
    hole_density=cfg['hole_density'],
    is_slippery=cfg['is_slippery'],
    success_rate=cfg['success_rate'],
    seed=config.RANDOM_SEED,
    save_path='results/agent_demos/sarsa_safety_best.gif',
    fps=2,
    max_attempts=20
)

# Summary
print("\n" + "="*70)
print("ALL VISUALIZATIONS COMPLETED! ✅")
print("="*70)
print("\nGenerated files:")
print("  📊 Policy Evolution (Grid):")
print("     results/agent_demos/policy_evolution.png")
print("     → Shows how policy improves from episode 1 to 10,000")
print("\n  🎬 Animated GIF:") 
print("     results/agent_demos/sarsa_safety_best.gif")
print("     → Shows the trained agent's policy in action")
print(f"\n📊 Agent Performance:")
print(f"   Success: {'✅ YES' if success else '❌ NO'}")
print(f"   Steps taken: {steps}")
print(f"   Attempts needed: {attempts}")
print("\n" + "="*70)
print("\n💡 What you have now:")
print("   1. Policy evolution showing learning progress (4 snapshots)")
print("   2. Animated GIF of the same agent demonstrating the learned policy")
print("   3. Consistent policy across all visualizations")
print("   4. All using SARSA with Safety-Based reward shaping")
print("\n" + "="*70)
