"""
Agent Trajectory Rendering - Animated GIF with Gymnasium Human Rendering

This module creates animated GIFs showing how a trained agent navigates through
the FrozenLake environment using its learned policy. Uses Gymnasium's built-in
human rendering mode for authentic visualization.

The animation continues until the agent successfully reaches the goal.

Usage:
    from utils.agent_rendering import create_agent_demo
    
    # After training an agent
    create_agent_demo(agent, save_path='results/agent_demo.gif')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import gymnasium as gym
import os
import io


def create_agent_demo(agent, grid_rows=6, grid_cols=6, hole_density=0.15, 
                     is_slippery=True, success_rate=0.7, seed=242,
                     save_path='agent_demo.gif', fps=2, max_attempts=10):
    """
    Create an animated GIF showing the agent navigating the environment.
    
    Uses Gymnasium's built-in human rendering mode to capture frames.
    Continues recording until the agent successfully reaches the goal.
    
    Args:
        agent: Trained agent with get_policy() method
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        hole_density: Density of holes (0.0 to 1.0)
        is_slippery: Whether the ice is slippery
        success_rate: Success rate for slippery environments
        seed: Random seed for environment generation
        save_path: Path to save the GIF
        fps: Frames per second for the animation
        max_attempts: Maximum number of episodes to try before giving up
    
    Returns:
        tuple: (success, num_steps, num_attempts)
            - success: Whether the agent reached the goal
            - num_steps: Number of steps taken in successful episode
            - num_attempts: Number of episodes attempted
    """
    from env.frozenlake_env import create_frozenlake_env
    
    print(f"🎬 Creating agent demonstration GIF...")
    print(f"   Recording until success (max {max_attempts} attempts)...")
    
    # Get policy
    policy = agent.get_policy()
    
    # Try multiple episodes until we get a successful one
    for attempt in range(1, max_attempts + 1):
        print(f"\n   Attempt {attempt}/{max_attempts}...", end=" ")
        
        # Create environment with rgb_array rendering
        env = create_frozenlake_env(
            rows=grid_rows,
            cols=grid_cols,
            hole_density=hole_density,
            is_slippery=is_slippery,
            seed=seed,
            success_rate=success_rate,
            render_mode='rgb_array'  # Capture frames as images
        )
        
        # Collect frames
        frames = []
        state, _ = env.reset()
        
        # Capture initial frame
        frame = env.render()
        frames.append(frame)
        
        # Follow the policy
        success = False
        steps = 0
        
        for step in range(100):  # Max 100 steps per episode
            # Check if we have a policy for this state
            if state not in policy:
                print(f"No policy for state {state}")
                break
            
            # Get action from policy
            action = policy[state]
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Capture frame
            frame = env.render()
            frames.append(frame)
            
            state = next_state
            steps += 1
            
            # Check if episode ended
            if terminated or truncated:
                if reward > 0:  # Success!
                    success = True
                    print(f"✅ SUCCESS in {steps} steps!")
                else:
                    print(f"❌ Failed (fell in hole or timeout)")
                break
        
        env.close()
        
        # If successful, create the GIF
        if success:
            print(f"\n   Creating GIF with {len(frames)} frames...")
            
            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Save as GIF
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
            pil_frames[0].save(
                save_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps),  # Duration in milliseconds
                loop=0  # Loop forever
            )
            
            print(f"   ✅ Saved animated GIF: {save_path}")
            print(f"   📊 Stats: {steps} steps, {len(frames)} frames, {attempt} attempt(s)")
            
            return True, steps, attempt
    
    # Failed to get a successful episode
    print(f"\n   ⚠️  Could not record a successful episode after {max_attempts} attempts")
    print(f"   💡 Try: Increase max_attempts or check if agent policy is good")
    
    return False, 0, max_attempts


def create_comparison_demo(mc_agent, sarsa_agent, grid_rows=6, grid_cols=6, 
                          hole_density=0.15, is_slippery=True, success_rate=0.7, 
                          seed=242, save_dir='results/', fps=2):
    """
    Create side-by-side GIF comparison of Monte Carlo and SARSA agents.
    
    Args:
        mc_agent: Trained Monte Carlo agent
        sarsa_agent: Trained SARSA agent
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        hole_density: Density of holes
        is_slippery: Whether the ice is slippery
        success_rate: Success rate for slippery environments
        seed: Random seed for environment generation
        save_dir: Directory to save the GIFs
        fps: Frames per second
    
    Returns:
        dict: Results for both agents
    """
    print("\n" + "="*70)
    print("CREATING AGENT COMPARISON DEMOS")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    # Create MC demo
    print("\n[1/2] Monte Carlo Agent:")
    mc_success, mc_steps, mc_attempts = create_agent_demo(
        agent=mc_agent,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        hole_density=hole_density,
        is_slippery=is_slippery,
        success_rate=success_rate,
        seed=seed,
        save_path=f'{save_dir}/mc_agent_demo.gif',
        fps=fps
    )
    results['MC'] = {
        'success': mc_success,
        'steps': mc_steps,
        'attempts': mc_attempts
    }
    
    # Create SARSA demo
    print("\n[2/2] SARSA Agent:")
    sarsa_success, sarsa_steps, sarsa_attempts = create_agent_demo(
        agent=sarsa_agent,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        hole_density=hole_density,
        is_slippery=is_slippery,
        success_rate=success_rate,
        seed=seed,
        save_path=f'{save_dir}/sarsa_agent_demo.gif',
        fps=fps
    )
    results['SARSA'] = {
        'success': sarsa_success,
        'steps': sarsa_steps,
        'attempts': sarsa_attempts
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Agent':<15} | {'Success':<10} | {'Steps':<10} | {'Attempts':<10}")
    print("-" * 70)
    print(f"{'Monte Carlo':<15} | {'✅' if mc_success else '❌':<10} | {mc_steps:<10} | {mc_attempts:<10}")
    print(f"{'SARSA':<15} | {'✅' if sarsa_success else '❌':<10} | {sarsa_steps:<10} | {sarsa_attempts:<10}")
    print("-" * 70)
    
    if mc_success and sarsa_success:
        if mc_steps < sarsa_steps:
            print(f"\n🏆 Monte Carlo was more efficient ({mc_steps} vs {sarsa_steps} steps)")
        elif sarsa_steps < mc_steps:
            print(f"\n🏆 SARSA was more efficient ({sarsa_steps} vs {mc_steps} steps)")
        else:
            print(f"\n⚖️  Both agents took the same number of steps ({mc_steps})")
    
    print("\n" + "="*70)
    print(f"GIFs saved to: {save_dir}")
    print("="*70)
    
    return results


# Example usage
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          AGENT RENDERING - GYMNASIUM HUMAN MODE              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This script creates animated GIFs showing trained agents navigating
    the FrozenLake environment using Gymnasium's built-in rendering.
    
    Features:
    • Uses Gymnasium's rgb_array render mode
    • Records until agent successfully reaches the goal
    • Creates smooth animated GIFs
    • Can compare multiple agents side-by-side
    
    ══════════════════════════════════════════════════════════════
    
    To use:
    
    1. Train your agent:
       from experiments.run_monte_carlo import run_monte_carlo
       _, _, _, agent, env = run_monte_carlo(..., return_best_agent=True)
    
    2. Create demo GIF:
       from utils.agent_rendering import create_agent_demo
       create_agent_demo(agent, save_path='results/demo.gif')
    
    3. Or compare two agents:
       from utils.agent_rendering import create_comparison_demo
       create_comparison_demo(mc_agent, sarsa_agent, save_dir='results/')
    
    ══════════════════════════════════════════════════════════════
    
    The GIF will show the agent moving through the grid using
    Gymnasium's official rendering, making it look professional
    and authentic!
    
    ══════════════════════════════════════════════════════════════
    """)
