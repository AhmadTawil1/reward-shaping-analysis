"""
Visualization utilities for FrozenLake environments.

This module provides functions to:
- Visualize grid maps with cell types
- Display policy arrows
- Show value functions
- Render policies in action

Used by:
- Presentation preparation
- Debugging
- Report generation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_grid(env, policy=None, value_function=None, title="FrozenLake Grid", save_path=None):
    """
    Visualize FrozenLake grid with optional policy arrows and values.
    
    Args:
        env: Gymnasium FrozenLake environment
        policy (dict): {state: action} mapping (optional)
        value_function (dict): {state: value} mapping (optional)
        title (str): Plot title
        save_path (str): Path to save figure (optional)
    """
    desc = env.unwrapped.desc
    rows = len(desc)
    cols = len(desc[0]) if rows > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color mapping
    colors = {
        'S': '#90EE90',  # Light green for start
        'G': '#FFD700',  # Gold for goal
        'H': '#FF6B6B',  # Red for holes
        'F': '#87CEEB'   # Sky blue for frozen
    }
    
    for i in range(rows):
        for j in range(cols):
            cell = desc[i][j]
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            
            # Draw cell
            color = colors.get(cell, 'white')
            rect = plt.Rectangle((j, rows - 1 - i), 1, 1, 
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add cell label
            ax.text(j + 0.5, rows - 1 - i + 0.85, cell,
                   ha='center', va='top', fontsize=14, fontweight='bold')
            
            # Add value function
            state = i * cols + j
            if value_function is not None and state in value_function:
                value = value_function[state]
                ax.text(j + 0.5, rows - 1 - i + 0.5, f'{value:.2f}',
                       ha='center', va='center', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add policy arrow
            if policy is not None and state in policy:
                action = policy[state]
                # Action mapping: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
                arrows = ['←', '↓', '→', '↑']
                arrow_offsets = [(-0.2, 0), (0, -0.2), (0.2, 0), (0, 0.2)]
                
                arrow_x = j + 0.5 + arrow_offsets[action][0]
                arrow_y = rows - 1 - i + 0.15 + arrow_offsets[action][1]
                
                ax.text(arrow_x, arrow_y, arrows[action],
                       ha='center', va='center', fontsize=20, 
                       color='darkblue', fontweight='bold')
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['S'], edgecolor='black', label='Start (S)'),
        mpatches.Patch(facecolor=colors['G'], edgecolor='black', label='Goal (G)'),
        mpatches.Patch(facecolor=colors['H'], edgecolor='black', label='Hole (H)'),
        mpatches.Patch(facecolor=colors['F'], edgecolor='black', label='Frozen (F)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_policies(env, policies_dict, title="Policy Comparison", save_path=None):
    """
    Visualize multiple policies side by side.
    
    Args:
        env: Gymnasium FrozenLake environment
        policies_dict (dict): {label: policy} mappings
        title (str): Overall title
        save_path (str): Path to save figure (optional)
    """
    n_policies = len(policies_dict)
    desc = env.unwrapped.desc
    rows = len(desc)
    cols = len(desc[0]) if rows > 0 else 0
    
    fig, axes = plt.subplots(1, n_policies, figsize=(6 * n_policies, 6))
    if n_policies == 1:
        axes = [axes]
    
    colors = {
        'S': '#90EE90', 'G': '#FFD700', 
        'H': '#FF6B6B', 'F': '#87CEEB'
    }
    
    for idx, (label, policy) in enumerate(policies_dict.items()):
        ax = axes[idx]
        
        for i in range(rows):
            for j in range(cols):
                cell = desc[i][j]
                if isinstance(cell, bytes):
                    cell = cell.decode('utf-8')
                
                color = colors.get(cell, 'white')
                rect = plt.Rectangle((j, rows - 1 - i), 1, 1, 
                                     facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                
                ax.text(j + 0.5, rows - 1 - i + 0.85, cell,
                       ha='center', va='top', fontsize=12, fontweight='bold')
                
                state = i * cols + j
                if policy is not None and state in policy:
                    action = policy[state]
                    arrows = ['←', '↓', '→', '↑']
                    ax.text(j + 0.5, rows - 1 - i + 0.15, arrows[action],
                           ha='center', va='center', fontsize=18, 
                           color='darkblue', fontweight='bold')
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols + 1))
        ax.set_yticks(range(rows + 1))
        ax.grid(True, alpha=0.3)
        ax.set_title(label, fontsize=14, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_learning_progress(env, policies_at_episodes, value_functions_at_episodes=None, 
                                title="Learning Progress", save_path=None):
    """
    Show policy evolution at different training episodes.
    
    Args:
        env: Gymnasium FrozenLake environment
        policies_at_episodes (dict): {episode_num: policy}
        value_functions_at_episodes (dict): {episode_num: value_func} (optional)
        title (str): Overall title
        save_path (str): Path to save figure (optional)
    """
    n_snapshots = len(policies_at_episodes)
    desc = env.unwrapped.desc
    rows = len(desc)
    cols = len(desc[0]) if rows > 0 else 0
    
    fig, axes = plt.subplots(1, n_snapshots, figsize=(6 * n_snapshots, 6))
    if n_snapshots == 1:
        axes = [axes]
    
    colors = {
        'S': '#90EE90', 'G': '#FFD700', 
        'H': '#FF6B6B', 'F': '#87CEEB'
    }
    
    sorted_episodes = sorted(policies_at_episodes.keys())
    
    for idx, episode in enumerate(sorted_episodes):
        ax = axes[idx]
        policy = policies_at_episodes[episode]
        value_func = value_functions_at_episodes.get(episode) if value_functions_at_episodes else None
        
        for i in range(rows):
            for j in range(cols):
                cell = desc[i][j]
                if isinstance(cell, bytes):
                    cell = cell.decode('utf-8')
                
                color = colors.get(cell, 'white')
                rect = plt.Rectangle((j, rows - 1 - i), 1, 1, 
                                     facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                
                state = i * cols + j
                
                # Show value if available
                if value_func is not None and state in value_func:
                    value = value_func[state]
                    ax.text(j + 0.5, rows - 1 - i + 0.5, f'{value:.2f}',
                           ha='center', va='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Show policy arrow
                if policy is not None and state in policy:
                    action = policy[state]
                    arrows = ['←', '↓', '→', '↑']
                    ax.text(j + 0.5, rows - 1 - i + 0.15, arrows[action],
                           ha='center', va='center', fontsize=16, 
                           color='darkblue', fontweight='bold')
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols + 1))
        ax.set_yticks(range(rows + 1))
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Episode {episode}', fontsize=14, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_grid_info(env):
    """
    Print textual information about the FrozenLake grid.
    
    Args:
        env: Gymnasium FrozenLake environment
    """
    desc = env.unwrapped.desc
    rows = len(desc)
    cols = len(desc[0]) if rows > 0 else 0
    
    print("=" * 50)
    print("FROZENLAKE GRID INFORMATION")
    print("=" * 50)
    print(f"Grid Size: {rows}x{cols}")
    print(f"Total Cells: {rows * cols}")
    
    # Count cell types
    counts = {'S': 0, 'G': 0, 'H': 0, 'F': 0}
    for row in desc:
        for cell in row:
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            counts[cell] = counts.get(cell, 0) + 1
    
    print(f"\nCell Distribution:")
    print(f"  Start (S):  {counts.get('S', 0)}")
    print(f"  Goal (G):   {counts.get('G', 0)}")
    print(f"  Holes (H):  {counts.get('H', 0)}")
    print(f"  Frozen (F): {counts.get('F', 0)}")
    
    hole_density = counts.get('H', 0) / (rows * cols)
    print(f"\nHole Density: {hole_density:.2%}")
    print(f"Is Slippery: {env.unwrapped.spec.kwargs.get('is_slippery', True)}")
    print("=" * 50)
    print("\nGrid Layout:")
    for i, row in enumerate(desc):
        row_str = ""
        for cell in row:
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            row_str += cell + " "
        print(f"  {row_str}")
    print("=" * 50)