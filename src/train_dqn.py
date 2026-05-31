import argparse
import time
import torch
import numpy as np
from agent import DQNAgent, ReplayBuffer
from config import DQNConfig
from logger import Logger
from utils import get_device, get_run_dir, make_env, make_eval_env, make_video_env, set_seed


# Two phases epsilon decay, it starts after the warmup phase
def get_epsilon(step: int, cfg: DQNConfig) -> float:
    
    # Avoiding warmup epsilon decay
    explore_step = max(0, step - cfg.learning_starts)
    
    # From epsilon_start to  epsilon_mid
    if explore_step < cfg.epsilon_decay_phase1:
        fraction = explore_step / cfg.epsilon_decay_phase1
        return cfg.epsilon_start - fraction * (cfg.epsilon_start - cfg.epsilon_mid)
        
    # From epsilon_mid to epsilon_end 
    elif explore_step < cfg.epsilon_decay_phase1 + cfg.epsilon_decay_phase2:
        phase2_step = explore_step - cfg.epsilon_decay_phase1
        fraction = phase2_step / cfg.epsilon_decay_phase2
        return cfg.epsilon_mid - fraction * (cfg.epsilon_mid - cfg.epsilon_end)
        
    # End phase
    else:
        return cfg.epsilon_end


# Run n_episodes with greedy policy, return (mean_reward, std_reward)
def evaluate(agent: DQNAgent, env_id: str, seed: int, n_episodes: int,
             env_cfg_kwargs: dict) -> tuple[float, float]:
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


# Record gameplay videos using the agent's greedy policy
def record_video(agent: DQNAgent, env_id: str, seed: int, video_dir: str,
                 env_cfg_kwargs: dict, n_episodes: int = 1) -> None:
    vid_env = make_video_env(env_id, seed=seed + 2000, video_dir=video_dir, **env_cfg_kwargs)

    for _ in range(n_episodes):
        obs, _ = vid_env.reset()
        done = False
        while not done:
            action = agent.select_action(np.array(obs), epsilon=0.0)
            obs, _, terminated, truncated, _ = vid_env.step(action)
            done = terminated or truncated

    vid_env.close()


# ----------------------- Main DQN training loop ----------------------
def train(cfg: DQNConfig, seed: int) -> None:
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
    
    start_step = 1
    episode_count = 0
    best_eval_reward = -float("inf")

    checkpoint_path = run_dir / "resume_checkpoint.pt"
    if checkpoint_path.exists():
        print(f"\n[RESUME] Found checkpoint: {checkpoint_path}")
        print("Loading the status of models and counters...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Network Recovery and Optimizer
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Resetting loop variables (resume from the next step)
        start_step = checkpoint["step"] + 1
        episode_count = checkpoint["episode_count"]
        best_eval_reward = checkpoint["best_eval_reward"]
        
        print(f"Recovery complete. Resuming from Step {start_step} (Episode {episode_count})\n")

    # Environment reset
    obs, _ = env.reset()
    episode_shaped_reward = 0.0  
    episode_true_score = 0.0
    episode_length = 0
    learn_steps = 0
    start_time = time.time()

    # List for log informations
    episode_losses = []
    episode_q_vals = []

    try:
        for step in range(start_step, cfg.total_timesteps + 1):
            epsilon = get_epsilon(step, cfg)
            action = agent.select_action(np.array(obs), epsilon)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory.push(np.array(obs), action, reward, np.array(next_obs), done)
            obs = next_obs
            
            episode_shaped_reward += reward
            episode_true_score += info.get("original_reward", 0.0) 
            episode_length += 1

            if len(memory) >= cfg.learning_starts and step % cfg.train_freq == 0:
                batch = memory.sample(cfg.batch_size)
                learn_result = agent.learn(batch)
                if learn_result is not None:
                    loss, q_val = learn_result
                    episode_losses.append(loss)
                    episode_q_vals.append(q_val)
                    
                    learn_steps += 1
                    if learn_steps % cfg.target_update_freq == 0:
                        agent.update_target_network()

            if done:
                episode_count += 1
                elapsed = time.time() - start_time
                fps = step / elapsed if elapsed > 0 else 0

                # Calculating mean values
                avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
                avg_q = float(np.mean(episode_q_vals)) if episode_q_vals else 0.0

                # Saving CSV
                logger.log({
                    "step": step,
                    "episode": episode_count,
                    "episode_reward": episode_true_score,
                    "episode_shaped_reward": episode_shaped_reward,
                    "episode_length": episode_length,
                    "epsilon": round(epsilon, 4),
                    "fps": round(fps, 1),
                    "avg_loss": round(avg_loss, 4),
                    "avg_q": round(avg_q, 4)
                })

                # Printing
                if episode_count % 10 == 0:
                    print(
                        f"Step {step:>8d}/{cfg.total_timesteps} | "
                        f"Ep {episode_count:>4d} | "
                        f"Score Reale {episode_true_score:>7.0f} | "
                        f"Reward Rete {episode_shaped_reward:>7.1f} | "
                        f"Loss {avg_loss:>6.4f} | "
                        f"Q-Med {avg_q:>6.3f} | "
                        f"Eps {epsilon:.3f} | "
                        f"FPS {fps:.0f}"
                    )
                
                obs, _ = env.reset()
                episode_shaped_reward = 0.0
                episode_true_score = 0.0
                episode_length = 0
                episode_losses.clear()
                episode_q_vals.clear()

            if step % cfg.eval_freq == 0:
                print(f"\n--- EVALUATION (Step {step}) ---")
                mean_r, std_r = evaluate(agent, cfg.env.env_id, seed, cfg.eval_episodes, env_cfg_kwargs)
                print(f"Result Eval: Mean Reward = {mean_r:.1f} +/- {std_r:.1f}")
                
                if mean_r > best_eval_reward:
                    print(f"New BEST MODEL found! (Reward: {best_eval_reward:.1f} -> {mean_r:.1f}). Saving...")
                    best_eval_reward = mean_r
                    agent.save(run_dir / "best_model.pt")
                else:
                    print(f"No improvement (Current Best: {best_eval_reward:.1f})")
                

            if step % cfg.checkpoint_freq == 0:
                agent.save(run_dir / f"checkpoint_{step:08d}.pt")

            # Checkpoints every 100k steps
            if step % 100000 == 0:
                torch.save({
                    "policy_net": agent.policy_net.state_dict(),
                    "target_net": agent.target_net.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "step": step,
                    "episode_count": episode_count,
                    "best_eval_reward": best_eval_reward
                }, checkpoint_path)
                print(f"Recovery file '{checkpoint_path.name}' updated at the step {step}.")

            if cfg.video_freq > 0 and step % cfg.video_freq == 0:
                video_dir = str(run_dir / "videos" / f"step_{step:08d}")
                record_video(agent, cfg.env.env_id, seed, video_dir, env_cfg_kwargs)

    # Manual saving at ctrl+C
    except KeyboardInterrupt:
        print("\n\nCtrl+C detected! Saving resume checkpoint...")
        torch.save({
            "policy_net": agent.policy_net.state_dict(),
            "target_net": agent.target_net.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "step": step,
            "episode_count": episode_count,
            "best_eval_reward": best_eval_reward
        }, checkpoint_path)
        print(f"Successfully saved at step {step}. Closing environments...")
        logger.close()
        env.close()
        return

    # End of the training
    agent.save(run_dir / "final_model.pt")
    if cfg.video_freq > 0:
        video_dir = str(run_dir / "videos" / "final")
        record_video(agent, cfg.env.env_id, seed, video_dir, env_cfg_kwargs)
        print(f"Final gameplay recorded → {video_dir}")
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
