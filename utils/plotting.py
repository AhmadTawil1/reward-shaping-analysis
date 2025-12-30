"""
Plotting utilities with 95% confidence intervals for RL experiments.

All plotting functions expect input arrays of shape:
    (num_runs, num_episodes)

Uses seaborn to automatically compute:
    mean ± 95% confidence interval
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _to_long_dataframe(values, metric_name):
    """
    Convert (runs, episodes) array into long-form DataFrame
    suitable for seaborn.

    Columns:
        episode, value, run
    """
    num_runs, num_episodes = values.shape

    data = {
        "episode": np.tile(np.arange(1, num_episodes + 1), num_runs),
        "value": values.flatten(),
        "run": np.repeat(np.arange(num_runs), num_episodes),
    }

    df = pd.DataFrame(data)
    df["metric"] = metric_name
    return df


def plot_ci(values, title, ylabel, xlabel="Episode", save_path=None):
    """
    Plot mean ± 95% confidence interval across runs.

    Args:
        values (np.ndarray): shape (runs, episodes)
        title (str)
        ylabel (str)
        xlabel (str)
        save_path (str | None)
    """
    assert values.ndim == 2, "Input must be shape (runs, episodes)"

    df = _to_long_dataframe(values, metric_name=ylabel)

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="episode",
        y="value",
        errorbar=("ci", 95),
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


# --------------------------------------------------
# Convenience wrappers
# --------------------------------------------------

def plot_throughput_ci(successes, title="Throughput (95% CI)", save_path=None):
    """
    Plot throughput (mean ± 95% CI).

    successes: np.ndarray (runs, episodes), values ∈ {0,1}
    """
    cumulative = np.cumsum(successes, axis=1)
    episodes = np.arange(1, successes.shape[1] + 1)
    throughput = cumulative / episodes

    plot_ci(
        throughput,
        title=title,
        ylabel="Throughput",
        save_path=save_path
    )


def plot_real_returns_ci(real_returns, title="Real Returns (95% CI)", save_path=None):
    """
    Plot per-episode real returns (mean ± 95% CI).
    """
    plot_ci(
        real_returns,
        title=title,
        ylabel="Real Return",
        save_path=save_path
    )


def plot_episode_length_ci(lengths, title="Episode Length (95% CI)", save_path=None):
    """
    Plot episode lengths (mean ± 95% CI).
    """
    plot_ci(
        lengths,
        title=title,
        ylabel="Episode Length",
        save_path=save_path
    )
