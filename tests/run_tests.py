"""
Test runner script for the Reward Shaping Analysis project.

This script runs all tests and provides a summary of results.
"""

import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests(verbosity=2):
    """
    Run all test suites.
    
    Args:
        verbosity (int): Verbosity level (0=quiet, 1=normal, 2=verbose)
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_suite(suite_name, verbosity=2):
    """
    Run a specific test suite.
    
    Args:
        suite_name (str): Name of the test module (e.g., 'test_environment')
        verbosity (int): Verbosity level
    
    Returns:
        bool: True if tests passed, False otherwise
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(suite_name)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for RL Reward Shaping Analysis')
    parser.add_argument(
        '--suite',
        type=str,
        help='Specific test suite to run (e.g., test_environment, test_algorithms)',
        default=None
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run only fast unit tests (skip integration tests)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("REWARD SHAPING ANALYSIS - TEST SUITE")
    print("="*70)
    
    if args.fast:
        print("\n🚀 Running FAST tests only (unit tests)...\n")
        # Run only unit tests, skip integration
        suites = ['test_environment', 'test_algorithms', 'test_reward_shaping']
        all_passed = True
        for suite in suites:
            print(f"\n{'='*70}")
            print(f"Running {suite}...")
            print(f"{'='*70}\n")
            passed = run_specific_suite(suite, args.verbosity)
            all_passed = all_passed and passed
    elif args.suite:
        print(f"\n🎯 Running specific suite: {args.suite}\n")
        all_passed = run_specific_suite(args.suite, args.verbosity)
    else:
        print("\n🧪 Running ALL tests...\n")
        all_passed = run_all_tests(args.verbosity)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        sys.exit(1)
