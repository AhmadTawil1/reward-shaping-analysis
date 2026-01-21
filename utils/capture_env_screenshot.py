"""
Capture a screenshot of the FrozenLake environment using the human GUI.

This script creates a FrozenLake environment with GUI rendering,
displays it for a few seconds, and saves a screenshot.

Usage:
    python capture_env_screenshot.py
"""

import time
import pygame
from env.frozenlake_env import create_frozenlake_env
import config


def capture_environment_screenshot(
    grid_rows=6,
    grid_cols=6,
    hole_density=0.15,
    seed=10,
    display_duration=3,
    save_path='results/frozenlake_environment.png'
):
    """
    Create and capture a screenshot of the FrozenLake environment.
    
    Args:
        grid_rows (int): Number of rows in the grid
        grid_cols (int): Number of columns in the grid
        hole_density (float): Density of holes (0.1-0.2)
        seed (int): Random seed for reproducibility
        display_duration (int): How long to display before capturing (seconds)
        save_path (str): Where to save the screenshot
    """
    print(f"\n{'='*70}")
    print("FROZENLAKE ENVIRONMENT SCREENSHOT CAPTURE")
    print(f"{'='*70}")
    print(f"Grid: {grid_rows}×{grid_cols}")
    print(f"Hole Density: {hole_density}")
    print(f"Seed: {seed}")
    print(f"{'='*70}\n")
    
    # Create environment with human rendering
    print("Creating environment with GUI rendering...")
    env = create_frozenlake_env(
        rows=grid_rows,
        cols=grid_cols,
        hole_density=hole_density,
        is_slippery=True,
        seed=seed,
        render_mode="human"
    )
    
    # Reset and render
    env.reset()
    env.render()
    
    print(f"Environment displayed. Waiting {display_duration} seconds before capturing...")
    time.sleep(display_duration)
    
    # Capture screenshot using pygame
    try:
        # Get the pygame window surface
        screen = pygame.display.get_surface()
        
        if screen is not None:
            # Save the screenshot
            pygame.image.save(screen, save_path)
            print(f"\n✅ Screenshot saved to: {save_path}")
        else:
            print("\n❌ Error: Could not access pygame display surface")
    except Exception as e:
        print(f"\n❌ Error capturing screenshot: {e}")
    
    # Keep window open for a moment
    print(f"\nKeeping window open for 2 more seconds...")
    time.sleep(2)
    
    # Close environment
    env.close()
    print("Environment closed.\n")


if __name__ == "__main__":
    # Use configuration from config.py
    cfg = config.get_final_config() if not config.TEST_MODE else config.get_test_config()
    
    print("\n" + "="*70)
    print("This script will:")
    print("  1. Create a FrozenLake environment with GUI")
    print("  2. Display it for a few seconds")
    print("  3. Capture and save a screenshot")
    print("="*70)
    
    input("\n Press ENTER to start...")
    
    # Capture screenshot
    capture_environment_screenshot(
        grid_rows=cfg['grid_rows'],
        grid_cols=cfg['grid_cols'],
        hole_density=cfg['hole_density'],
        seed=242,
        display_duration=3,
        save_path=f'{config.RESULTS_DIR}/frozenlake_environment.png'
    )
    
    print("="*70)
    print("✅ SCREENSHOT CAPTURE COMPLETE!")
    print("="*70)
    print(f"\nScreenshot saved to: {config.RESULTS_DIR}/frozenlake_environment.png")
    print("="*70)
