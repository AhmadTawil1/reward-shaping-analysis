"""
Policy visualization utilities for FrozenLake.

Visualizes learned policies as arrows on the grid showing
which action the agent prefers in each state.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
import os


def visualize_policy_grid(agent, env, title='Learned Policy', save_path=None):
    """
    Visualize the learned policy as arrows on the FrozenLake grid.
    
    Args:
        agent: Trained agent with q_table attribute
        env: The environment (should be unwrapped to access desc)
        title (str): Title for the plot
        save_path (str): Path to save the figure
    """
    # Unwrap environment to get the base FrozenLake
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    
    # Get grid description
    desc = unwrapped_env.desc
    rows, cols = desc.shape
    
    # Action mapping: 0=Left, 1=Down, 2=Right, 3=Up
    action_arrows = {
        0: (-0.25, 0),      # Left (reduced from -0.3)
        1: (0, -0.25),      # Down (reduced from -0.3)
        2: (0.25, 0),       # Right (reduced from 0.3)
        3: (0, 0.25)        # Up (reduced from 0.3)
    }
    
    action_names = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))
    
    # Get policy and value function
    policy = agent.get_policy()
    value_func = agent.get_value_function()
    
    # Normalize values for coloring
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
            
            # Determine cell color based on type
            if cell_type == 'S':
                color = 'lightgreen'
                label = 'S'
            elif cell_type == 'G':
                color = 'gold'
                label = 'G'
            elif cell_type == 'H':
                color = 'lightcoral'
                label = 'H'
            else:  # Frozen
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
            
            # Add cell label (S, G, H) at the top
            if label:
                ax.text(j + 0.5, rows - i - 0.2, label, 
                       ha='center', va='center', fontsize=20, fontweight='bold')
            
            # Add Q-value text at the bottom of the cell
            if state in value_func and cell_type not in ['H']:
                q_value = value_func[state]
                ax.text(j + 0.5, rows - i - 0.85, f'{q_value:.2f}',
                       ha='center', va='center', fontsize=11, 
                       fontweight='bold', color='darkblue',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='none', alpha=0.7))
            
            # Draw policy arrow (if not hole or goal)
            if cell_type not in ['H', 'G'] and state in policy:
                action = policy[state]
                dx, dy = action_arrows[action]
                
                # Draw arrow (smaller size)
                arrow = FancyArrow(j + 0.5, rows - i - 0.4, dx, dy,
                                  width=0.1, head_width=0.2, head_length=0.1,
                                  fc='darkred', ec='darkred', linewidth=1.5)
                ax.add_patch(arrow)
                
                # Add action symbol (smaller)
                ax.text(j + 0.5 + dx * 1.15, rows - i - 0.4 + dy * 1.15, 
                       action_names[action],
                       ha='center', va='center', fontsize=14, 
                       fontweight='bold', color='darkred')
    
    # Set axis properties
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Start (S)'),
        mpatches.Patch(facecolor='gold', edgecolor='black', label='Goal (G)'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='black', label='Hole (H)'),
        mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Frozen'),
        mpatches.FancyArrow(0, 0, 0.3, 0, width=0.1, 
                           fc='darkred', ec='darkred', label='Policy Action')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
             fontsize=10, frameon=True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved policy visualization: {save_path}")
    
    plt.show()
    plt.close()


def visualize_policy_comparison(agents_dict, env, title='Policy Comparison', save_path=None):
    """
    Visualize multiple policies side by side for comparison.
    
    Args:
        agents_dict (dict): Dictionary of {label: agent}
        env: The environment
        title (str): Overall title
        save_path (str): Path to save the figure
    """
    num_policies = len(agents_dict)
    
    # Create subplots
    if num_policies <= 2:
        rows, cols = 1, num_policies
        figsize = (cols * 8, 8)
    else:
        rows = (num_policies + 1) // 2
        cols = 2
        figsize = (16, rows * 8)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if num_policies == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    # Unwrap environment
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    
    desc = unwrapped_env.desc
    grid_rows, grid_cols = desc.shape
    
    action_arrows = {
        0: (-0.3, 0),
        1: (0, -0.3),
        2: (0.3, 0),
        3: (0, 0.3)
    }
    
    action_names = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑'
    }
    
    for idx, (label, agent) in enumerate(agents_dict.items()):
        ax = axes[idx]
        
        policy = agent.get_policy()
        value_func = agent.get_value_function()
        
        # Normalize values
        if value_func:
            max_val = max(value_func.values())
            min_val = min(value_func.values())
            val_range = max_val - min_val if max_val != min_val else 1.0
        else:
            max_val, min_val, val_range = 0, 0, 1.0
        
        # Draw grid
        for i in range(grid_rows):
            for j in range(grid_cols):
                state = i * grid_cols + j
                cell_type = desc[i, j].decode('utf-8') if isinstance(desc[i, j], bytes) else desc[i, j]
                
                if cell_type == 'S':
                    color = 'lightgreen'
                    cell_label = 'S'
                elif cell_type == 'G':
                    color = 'gold'
                    cell_label = 'G'
                elif cell_type == 'H':
                    color = 'lightcoral'
                    cell_label = 'H'
                else:
                    if state in value_func:
                        normalized_val = (value_func[state] - min_val) / val_range
                        color = plt.cm.Blues(0.3 + 0.5 * normalized_val)
                    else:
                        color = 'lightblue'
                    cell_label = ''
                
                rect = plt.Rectangle((j, grid_rows - i - 1), 1, 1,
                                    facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                
                if cell_label:
                    ax.text(j + 0.5, grid_rows - i - 0.5, cell_label,
                           ha='center', va='center', fontsize=16, fontweight='bold')
                
                if cell_type not in ['H', 'G'] and state in policy:
                    action = policy[state]
                    dx, dy = action_arrows[action]
                    
                    arrow = FancyArrow(j + 0.5, grid_rows - i - 0.5, dx, dy,
                                      width=0.12, head_width=0.25, head_length=0.12,
                                      fc='darkred', ec='darkred', linewidth=1.5)
                    ax.add_patch(arrow)
                    
                    ax.text(j + 0.5 + dx * 1.2, grid_rows - i - 0.5 + dy * 1.2,
                           action_names[action],
                           ha='center', va='center', fontsize=12,
                           fontweight='bold', color='darkred')
        
        ax.set_xlim(0, grid_cols)
        ax.set_ylim(0, grid_rows)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(label, fontsize=13, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_policies, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved policy comparison: {save_path}")
    
    plt.show()
    plt.close()
