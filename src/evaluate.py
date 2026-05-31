# Standalone evaluation script for trained DQN agents

import argparse
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from agent import DQNAgent
from utils import get_device, make_eval_env, set_seed


# Run n_episodes with greedy policy, return per-episode metrics
def evaluate(agent: DQNAgent, env: gym.Env, n_episodes: int) -> list[dict]:
    episodes = []

    for i in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        actions_taken = []
        done = False

        while not done:
            action = agent.select_action(np.array(obs), epsilon=0.0)
            actions_taken.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        # Count action distribution to understand agent behavior
        action_counts = {}
        for a in actions_taken:
            action_counts[a] = action_counts.get(a, 0) + 1

        episodes.append({
            "reward": episode_reward,
            "length": episode_length,
            "action_distribution": action_counts,
        })
        print(f"  Episode {i + 1}/{n_episodes}: reward = {episode_reward:.1f}, length = {episode_length}")

    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
    parser.add_argument("run_dir", type=str, help="Path to run directory (e.g. runs/dqn/seed_42)")
    parser.add_argument("--model-name", type=str, default="best_model.pt",
                        help="Model filename inside run_dir (default: best_model.pt)")
    parser.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=100, help="Eval seed (should differ from training seed)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Load run metadata to recover env config
    run_dir = Path(args.run_dir)
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        env_cfg = meta["config"]["env"]
    else:
        env_cfg = {"env_id": "ALE/KungFuMaster-v5", "screen_size": 84, "frame_skip": 4, "frame_stack": 4}

    # Create eval environment (no life loss termination, no reward clipping)
    env = make_eval_env(
        env_id=env_cfg["env_id"],
        seed=args.seed,
        screen_size=env_cfg["screen_size"],
        frame_skip=env_cfg["frame_skip"],
        frame_stack=env_cfg["frame_stack"],
    )

    # Load agent
    model_path = run_dir / args.model_name
    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        return

    agent = DQNAgent(
        action_dim=env.action_space.n,
        device=device,
        input_channels=env_cfg["frame_stack"],
    )
    agent.load(model_path)
    agent.policy_net.eval()

    # Get action names from the environment
    action_meanings = env.unwrapped.get_action_meanings()

    print(f"Evaluating {model_path} | {args.episodes} episodes | seed={args.seed}")

    episodes = evaluate(agent, env, args.episodes)
    env.close()

    rewards = [ep["reward"] for ep in episodes]
    lengths = [ep["length"] for ep in episodes]

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))

    print(f"\nReward:  {mean_r:.1f} +/- {std_r:.1f} (min={min(rewards):.1f}, max={max(rewards):.1f})")
    print(f"Length:  {mean_len:.1f} +/- {std_len:.1f} (min={min(lengths)}, max={max(lengths)})")

    # Aggregate action distribution across all episodes
    total_actions: dict[int, int] = {}
    total_steps = sum(lengths)
    for ep in episodes:
        for action_idx, count in ep["action_distribution"].items():
            total_actions[int(action_idx)] = total_actions.get(int(action_idx), 0) + count

    print("\nAction distribution:")
    for action_idx in sorted(total_actions.keys()):
        name = action_meanings[action_idx] if action_idx < len(action_meanings) else f"ACTION_{action_idx}"
        pct = 100.0 * total_actions[action_idx] / total_steps
        print(f"  {action_idx:>2d} {name:<20s}: {pct:5.1f}%")

    # Save results
    results_path = run_dir / "eval_results.json"
    results = {
        "model": args.model_name,
        "seed": args.seed,
        "episodes": args.episodes,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_length": mean_len,
        "std_length": std_len,
        "action_distribution": {
            action_meanings[int(k)] if int(k) < len(action_meanings) else f"ACTION_{k}": v
            for k, v in sorted(total_actions.items())
        },
        "total_steps": total_steps,
        "per_episode": episodes,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
