"""
Final experiments for Reward Shaping Analysis.

This script produces all figures used in the report:
- Throughput (mean ± 95% CI)
- Per-episode real returns
- Episode length

All experiments use FINAL FrozenLake settings.
"""

from experiments.run_monte_carlo import run_monte_carlo
from experiments.run_sarsa import run_sarsa

from utils.plotting import (
    plot_throughput_ci,
    plot_real_returns_ci,
    plot_episode_length_ci
)


N_RUNS = 20
N_EPISODES = 5000


def run_mc_experiments():
    print("\n=== Monte Carlo ===")

    for shaping in [None, "step", "potential"]:
        print(f"\nMC shaping: {shaping}")

        successes, returns, lengths = run_monte_carlo(
            num_runs=N_RUNS,
            num_episodes=N_EPISODES,
            shaping_type=shaping
        )

        label = "Baseline" if shaping is None else shaping.capitalize()

        plot_throughput_ci(
            successes,
            title=f"MC - {label} - Throughput (95% CI)"
        )

        plot_real_returns_ci(
            returns,
            title=f"MC - {label} - Real Returns (95% CI)"
        )

        plot_episode_length_ci(
            lengths,
            title=f"MC - {label} - Episode Length (95% CI)"
        )


def run_sarsa_experiments():
    print("\n=== SARSA ===")

    for shaping in [None, "step", "potential"]:
        print(f"\nSARSA shaping: {shaping}")

        successes, returns, lengths = run_sarsa(
            num_runs=N_RUNS,
            num_episodes=N_EPISODES,
            shaping_type=shaping
        )

        label = "Baseline" if shaping is None else shaping.capitalize()

        plot_throughput_ci(
            successes,
            title=f"SARSA - {label} - Throughput (95% CI)"
        )

        plot_real_returns_ci(
            returns,
            title=f"SARSA - {label} - Real Returns (95% CI)"
        )

        plot_episode_length_ci(
            lengths,
            title=f"SARSA - {label} - Episode Length (95% CI)"
        )


if __name__ == "__main__":
    run_mc_experiments()
    run_sarsa_experiments()
