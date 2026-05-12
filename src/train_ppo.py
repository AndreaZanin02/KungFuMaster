"""PPO training script for ALE/KungFuMaster-v5 using Stable-Baselines3."""

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import PPOConfig
from logger import Logger
from utils import get_device, get_run_dir, make_env, make_eval_env, set_seed


def make_monitored_env(env_id: str, seed: int, idx: int, env_cfg_kwargs: dict):
    """Factory function: produces a Monitor-wrapped env. SB3 needs Monitor to
    extract per-episode reward and length from the info dict."""
    def _make():
        env = make_env(env_id, seed=seed + idx, **env_cfg_kwargs)
        env = Monitor(env)
        return env
    return _make


class CSVLoggerCallback(BaseCallback):
    """Logs per-episode metrics to CSV in the same format as DQN, so
    plot_results.py works without modification."""

    def __init__(self, logger: Logger):
        super().__init__()
        self.csv_logger = logger
        self.episode_count = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                elapsed = time.time() - self.start_time
                fps = self.num_timesteps / elapsed if elapsed > 0 else 0
                self.csv_logger.log({
                    "step": self.num_timesteps,
                    "episode": self.episode_count,
                    "episode_reward": float(info["episode"]["r"]),
                    "episode_length": int(info["episode"]["l"]),
                    "epsilon": 0.0,
                    "fps": round(fps, 1),
                })
        return True


class EvalCallback(BaseCallback):
    """Periodically evaluates the current policy with deterministic actions.
    Saves the best model seen so far (RL final checkpoints often underperform peak)."""

    def __init__(self, env_id: str, seed: int, eval_freq: int, eval_episodes: int,
                 env_cfg_kwargs: dict, run_dir: Path):
        super().__init__()
        self.env_id = env_id
        self.seed = seed
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.env_cfg_kwargs = env_cfg_kwargs
        self.run_dir = run_dir
        self.best_reward = -float("inf")
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            mean_r, std_r = self._evaluate()
            print(f"  [EVAL] Step {self.num_timesteps} | Mean: {mean_r:.1f} +/- {std_r:.1f}")
            if mean_r > self.best_reward:
                self.best_reward = mean_r
                self.model.save(self.run_dir / "best_model")
                print(f"  [EVAL] New best (reward={mean_r:.1f})")
        return True

    def _evaluate(self) -> tuple[float, float]:
        eval_env = make_eval_env(self.env_id, seed=self.seed + 1000, **self.env_cfg_kwargs)
        rewards = []
        for _ in range(self.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = eval_env.step(int(action))
                episode_reward += r
                done = terminated or truncated
            rewards.append(episode_reward)
        eval_env.close()
        return float(np.mean(rewards)), float(np.std(rewards))


def train(cfg: PPOConfig, seed: int) -> None:
    set_seed(seed)
    device = get_device()
    run_dir = get_run_dir("runs", "ppo", seed)
    print(f"Training PPO | seed={seed} | device={device} | run_dir={run_dir}")

    env_cfg_kwargs = {
        "screen_size": cfg.env.screen_size,
        "frame_skip": cfg.env.frame_skip,
        "frame_stack": cfg.env.frame_stack,
        "clip_rewards": cfg.env.clip_rewards,
    }

    env_fns = [
        make_monitored_env(cfg.env.env_id, seed, idx, env_cfg_kwargs)
        for idx in range(cfg.n_envs)
    ]
    env = DummyVecEnv(env_fns)

    logger = Logger(run_dir, cfg)
    csv_callback = CSVLoggerCallback(logger)
    eval_env_kwargs = {k: v for k, v in env_cfg_kwargs.items() if k != "clip_rewards"}
    eval_callback = EvalCallback(
        env_id=cfg.env.env_id,
        seed=seed,
        eval_freq=cfg.eval_freq,
        eval_episodes=cfg.eval_episodes,
        env_cfg_kwargs=eval_env_kwargs,
        run_dir=run_dir,
    )

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        verbose=1,
        device=device,
        seed=seed,
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[csv_callback, eval_callback],
    )

    model.save(run_dir / "final_model")
    logger.close()
    env.close()
    print(f"Training complete. Best eval reward: {eval_callback.best_reward:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on KungFuMaster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of parallel envs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig()
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    train(cfg, seed=args.seed)