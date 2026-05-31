import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------- Plot training curves from CSV log files ------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from run logs")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Base runs directory")
    parser.add_argument("--algorithms", nargs="+", default=["dqn"], help="Algorithms to plot (e.g. dqn ppo)")
    parser.add_argument("--window", type=int, default=10, help="Smoothing window size")
    parser.add_argument("--no-show", action="store_true", help="Save only, don't display")
    return parser.parse_args()


# Simple moving average for smoother curves
def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# Plot episode rewards from a single seed's metrics.csv
def plot_single_seed(csv_path: Path, ax: plt.Axes, label: str, color: str,
                    window: int = 10) -> None:
    df = pd.read_csv(csv_path)
    rewards = df["episode_reward"].values
    steps = df["step"].values

    smoothed = smooth(rewards, window)
    # Align x-axis after smoothing
    smoothed_steps = steps[window - 1:]

    ax.plot(smoothed_steps, smoothed, label=label, color=color, alpha=0.9)
    ax.fill_between(
        steps, rewards, alpha=0.1, color=color,
    )


# Plot mean +/- std reward across multiple seeds for one algorithm
def plot_multi_seed(run_dirs: list[Path], algorithm: str, ax: plt.Axes, color: str,
                    window: int = 10) -> None:
    all_rewards = []
    min_len = float("inf")

    for run_dir in run_dirs:
        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        rewards = df["episode_reward"].values
        all_rewards.append(rewards)
        min_len = min(min_len, len(rewards))

    if not all_rewards:
        print(f"No data found for {algorithm}")
        return

    # Truncate to shortest run and compute stats
    truncated = np.array([r[:min_len] for r in all_rewards])
    mean_r = np.mean(truncated, axis=0)
    std_r = np.std(truncated, axis=0)

    smoothed_mean = smooth(mean_r, window)
    smoothed_std = smooth(std_r, window)
    x = np.arange(len(smoothed_mean))

    ax.plot(x, smoothed_mean, label=f"{algorithm} (n={len(all_rewards)} seeds)", color=color)
    ax.fill_between(x, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                    alpha=0.2, color=color)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.runs_dir)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"dqn": "#2196F3", "ppo": "#FF5722", "a2c": "#4CAF50"}

    for algorithm in args.algorithms:
        algo_dir = base_dir / algorithm
        if not algo_dir.exists():
            print(f"Warning: {algo_dir} not found, skipping")
            continue

        run_dirs = sorted([d for d in algo_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        color = colors.get(algorithm, "#607D8B")

        if len(run_dirs) == 1:
            plot_single_seed(run_dirs[0] / "metrics.csv", ax, label=algorithm.upper(), color=color,
                            window=args.window)
        else:
            plot_multi_seed(run_dirs, algorithm.upper(), ax, color=color, window=args.window)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("KungFuMaster — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = base_dir / "training_curves.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
