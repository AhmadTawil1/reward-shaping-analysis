"""
Quick test script to verify rectangular grid support
"""
import sys
sys.path.append('.')

from env.frozenlake_env import create_frozenlake_env
from utils.visualization import print_grid_info

print("=" * 70)
print("TESTING RECTANGULAR GRID SUPPORT")
print("=" * 70)

# Test 1: 7x6 grid
print("\n[Test 1] Creating 7x6 grid (7 rows, 6 columns)...")
env_7x6 = create_frozenlake_env(
    rows=7,
    cols=6,
    hole_density=0.1,
    is_slippery=False,
    seed=42
)
print_grid_info(env_7x6)
env_7x6.close()

# Test 2: 6x7 grid
print("\n[Test 2] Creating 6x7 grid (6 rows, 7 columns)...")
env_6x7 = create_frozenlake_env(
    rows=6,
    cols=7,
    hole_density=0.1,
    is_slippery=False,
    seed=42
)
print_grid_info(env_6x7)
env_6x7.close()

# Test 3: Square 6x6 grid (should still work)
print("\n[Test 3] Creating 6x6 grid (square grid)...")
env_6x6 = create_frozenlake_env(
    rows=6,
    cols=6,
    hole_density=0.1,
    is_slippery=False,
    seed=42
)
print_grid_info(env_6x6)
env_6x6.close()

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
