"""DQN training script for ALE/KungFuMaster-v5."""

import argparse
import time

import numpy as np

from agent import DQNAgent, ReplayBuffer
from config import DQNConfig
from logger import Logger
from utils import get_device, get_run_dir, make_env, make_eval_env, make_video_env, set_seed


def get_epsilon(step: int, cfg: DQNConfig) -> float:
    """Linear epsilon decay from epsilon_start to epsilon_end over epsilon_decay_steps."""
    # Linearly interpolate between start and end: 1.0 → 0.01 over epsilon_decay_steps
    fraction = min(1.0, step / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def evaluate(agent: DQNAgent, env_id: str, seed: int, n_episodes: int,
             env_cfg_kwargs: dict) -> tuple[float, float]:
    """Run n_episodes with greedy policy, return (mean_reward, std_reward)."""
    # Use a separate env with different seed, no reward clipping, and real episode boundaries
    eval_env = make_eval_env(env_id, seed=seed + 1000, **env_cfg_kwargs)
    rewards = []

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # Greedy action selection (epsilon=0.0 means no exploration)
            action = agent.select_action(np.array(obs), epsilon=0.0)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    eval_env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def record_video(agent: DQNAgent, env_id: str, seed: int, video_dir: str,
                 env_cfg_kwargs: dict, n_episodes: int = 1) -> None:
    """Record gameplay videos using the agent's greedy policy."""
    vid_env = make_video_env(env_id, seed=seed + 2000, video_dir=video_dir, **env_cfg_kwargs)

    for _ in range(n_episodes):
        obs, _ = vid_env.reset()
        done = False
        while not done:
            action = agent.select_action(np.array(obs), epsilon=0.0)
            obs, _, terminated, truncated, _ = vid_env.step(action)
            done = terminated or truncated

    vid_env.close()


def train(cfg: DQNConfig, seed: int) -> None:
    """Main DQN training loop."""
    set_seed(seed)
    device = get_device()
    run_dir = get_run_dir("runs", "dqn", seed)

    print(f"Training DQN | seed={seed} | device={device} | run_dir={run_dir}")

    # Pack env config into a dict so we can pass it to make_env, make_eval_env, etc.
    env_cfg_kwargs = {
        "screen_size": cfg.env.screen_size,
        "frame_skip": cfg.env.frame_skip,
        "frame_stack": cfg.env.frame_stack,
    }

    # Training env has reward clipping and life-loss termination for stability
    env = make_env(cfg.env.env_id, seed=seed, clip_rewards=cfg.env.clip_rewards, **env_cfg_kwargs)
    action_dim = env.action_space.n

    agent = DQNAgent(
        action_dim=action_dim,
        device=device,
        input_channels=cfg.env.frame_stack,
        lr=cfg.learning_rate,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
    )
    memory = ReplayBuffer(cfg.buffer_size)
    logger = Logger(run_dir, cfg)

    obs, _ = env.reset()
    episode_shaped_reward = 0.0  
    episode_true_score = 0.0
    episode_length = 0
    episode_count = 0
    learn_steps = 0
    best_eval_reward = -float("inf")
    start_time = time.time()

    for step in range(1, cfg.total_timesteps + 1):
        epsilon = get_epsilon(step, cfg)
        action = agent.select_action(np.array(obs), epsilon)

        # In 'info' there are the original points
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.push(np.array(obs), action, reward, np.array(next_obs), done)

        obs = next_obs
        
        # Counters updating
        episode_shaped_reward += reward
        episode_true_score += info.get("original_reward", 0.0) 
        episode_length += 1

        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            loss = agent.learn(memory)
            if loss is not None:
                learn_steps += 1
                if learn_steps % cfg.target_update_freq == 0:
                    agent.update_target_network()

        if done:
            episode_count += 1
            elapsed = time.time() - start_time
            fps = step / elapsed if elapsed > 0 else 0

            logger.log({
                "step": step,
                "episode": episode_count,
                "episode_reward": episode_true_score,
                "episode_length": episode_length,
                "epsilon": round(epsilon, 4),
                "fps": round(fps, 1),
            })

            if episode_count % 10 == 0:
                print(
                    f"Step {step:>8d}/{cfg.total_timesteps} | "
                    f"Ep {episode_count:>4d} | "
                    f"Score Reale {episode_true_score:>7.0f} | "
                    f"Reward Rete {episode_shaped_reward:>7.1f} | "
                    f"Eps {epsilon:.3f} | "
                    f"FPS {fps:.0f}"
                )

            obs, _ = env.reset()
            episode_shaped_reward = 0.0
            episode_true_score = 0.0
            episode_length = 0

        # Test the agent with no exploration to measure true performance
        if step % cfg.eval_freq == 0:
            mean_r, std_r = evaluate(agent, cfg.env.env_id, seed, cfg.eval_episodes, env_cfg_kwargs)
            print(f"  [EVAL] Step {step} | Mean reward: {mean_r:.1f} +/- {std_r:.1f}")

            # Save the best model since RL training is noisy and the final model isn't always the best
            if mean_r > best_eval_reward:
                best_eval_reward = mean_r
                agent.save(run_dir / "best_model.pt")
                print(f"  [EVAL] New best model saved (reward={mean_r:.1f})")

        # Save periodic checkpoint as insurance against crashes
        if step % cfg.checkpoint_freq == 0:
            agent.save(run_dir / f"checkpoint_{step:08d}.pt")

        # Record a video to visually track how the agent's strategy evolves
        if cfg.video_freq > 0 and step % cfg.video_freq == 0:
            video_dir = str(run_dir / "videos" / f"step_{step:08d}")
            record_video(agent, cfg.env.env_id, seed, video_dir, env_cfg_kwargs)
            print(f"  [VIDEO] Recorded gameplay at step {step} → {video_dir}")

    agent.save(run_dir / "final_model.pt")
    if cfg.video_freq > 0:
        video_dir = str(run_dir / "videos" / "final")
        record_video(agent, cfg.env.env_id, seed, video_dir, env_cfg_kwargs)
        print(f"  [VIDEO] Final gameplay recorded → {video_dir}")
    logger.close()
    env.close()
    print(f"Training complete. Best eval reward: {best_eval_reward:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on KungFuMaster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--buffer-size", type=int, default=None, help="Override buffer size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DQNConfig()

    # Override default config with any CLI arguments provided
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.buffer_size is not None:
        cfg.buffer_size = args.buffer_size

    train(cfg, seed=args.seed)
