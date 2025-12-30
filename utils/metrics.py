# utils/metrics.py
"""
Metrics utilities for reinforcement learning experiments.

Conventions:
- successes: binary (0/1) per episode
- returns: REAL environment returns (not shaped)
- throughput: cumulative success ratio
"""


import numpy as np


def cumulative_throughput(successes):
    """
    Compute cumulative throughput over episodes.

    Args:
        successes (list or np.ndarray):
            Binary list where 1 = success, 0 = failure.

    Returns:
        np.ndarray: cumulative success rate per episode.
    """
    successes = np.asarray(successes, dtype=float)
    cumulative = np.cumsum(successes)
    episodes = np.arange(1, len(successes) + 1)
    return cumulative / episodes


def moving_average(data, window):
    """
    Compute moving average over a fixed window.

    Args:
        data (list or np.ndarray)
        window (int)

    Returns:
        np.ndarray
    """
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return np.array([])
    return np.convolve(data, np.ones(window) / window, mode="valid")
