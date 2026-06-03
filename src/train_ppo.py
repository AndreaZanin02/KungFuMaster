import argparse
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from config import PPOConfig
from utils import get_device, get_git_hash, get_run_dir, make_env, make_eval_env, make_video_env, set_seed
import csv
import json
from dataclasses import asdict
import gymnasium


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on KungFuMaster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--use-dummy", action="store_true",
                        help="Use DummyVecEnv instead of SubprocVecEnv (slower but compatible with Colab/Windows)")
    return parser.parse_args()


# ---------------- CSV Logger compatible with plot_results.py --------------------
class CSVLogger:

    def __init__(self, run_dir: Path, config: object) -> None:
        self.run_dir = run_dir
        self.csv_path = run_dir / "metrics.csv"
        self.meta_path = run_dir / "metadata.json"
        self._csv_file = None
        self._writer = None
        self._save_metadata(config)


    def _save_metadata(self, config: object) -> None:
        meta = {
            "config": asdict(config),
            "versions": {
                "torch": torch.__version__,
                "gymnasium": gymnasium.__version__,
                "numpy": np.__version__,
            },
            "git_hash": get_git_hash(),
            "device": str(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)


    def log(self, metrics: dict) -> None:
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(metrics.keys()))
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._csv_file.flush()


    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()


# ---------------- Callback SB3: logging + eval + checkpointing ----------------------
class TrainingCallback(BaseCallback):

    def __init__(self, cfg: PPOConfig, run_dir: Path, seed: int,
                 csv_logger: CSVLogger, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.run_dir = run_dir
        self.seed = seed
        self.csv_logger = csv_logger

        self.best_eval_reward = -float("inf")
        self.episode_count = 0
        self.start_time = time.time()

        # Buffer to accumulate metrics of completed episodes across envs
        self._ep_rewards: list[float] = []
        self._ep_true_scores: list[float] = []
        self._ep_lengths: list[int] = []

        # Tracking last events (robust against frequencies not aligned to n_envs)
        self._last_eval_step = 0
        self._last_checkpoint_step = 0
        self._last_resume_step = 0
        self._last_video_step = 0

        # Path resume checkpoint
        self.checkpoint_path = run_dir / "resume_checkpoint.zip"


    def _on_step(self) -> bool:
        # SB3 populates 'self.locals["infos"]' with current step data
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                true_score = info.get("true_score", ep_reward)

                self._ep_rewards.append(ep_reward)
                self._ep_true_scores.append(true_score)
                self._ep_lengths.append(ep_length)

        # Log every 10 episodes
        if len(self._ep_rewards) >= 10:
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            mean_r = float(np.mean(self._ep_rewards))
            mean_score = float(np.mean(self._ep_true_scores))
            mean_len = float(np.mean(self._ep_lengths))

            self.csv_logger.log({
                "step": self.num_timesteps,
                "episode": self.episode_count,
                "episode_reward": mean_score,
                "episode_shaped_reward": mean_r,
                "episode_length": mean_len,
                "epsilon": 0.0,   # PPO does not use epsilon; fake 0 for CSV compatibility with DQN
                "fps": round(fps, 1),
                "avg_loss": 0.0,  # Not directly available from SB3 callback
                "avg_q": 0.0,
            })

            print(
                f"Step {self.num_timesteps:>8d}/{self.cfg.total_timesteps} | "
                f"Ep {self.episode_count:>4d} | "
                f"Score {mean_score:>7.0f} | "
                f"Net Reward {mean_r:>7.1f} | "
                f"FPS {fps:.0f}"
            )

            self._ep_rewards.clear()
            self._ep_true_scores.clear()
            self._ep_lengths.clear()

        # Periodic evaluation (robust: triggers when interval has passed, not on exact modulo)
        if self.num_timesteps - self._last_eval_step >= self.cfg.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_eval()

        # Checkpoint at intervals
        if self.num_timesteps - self._last_checkpoint_step >= self.cfg.checkpoint_freq:
            self._last_checkpoint_step = self.num_timesteps
            ckpt_path = str(self.run_dir / f"checkpoint_{self.num_timesteps:08d}")
            self.model.save(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}.zip")

        # Resume checkpoint every 100k steps (overwrites)
        if self.num_timesteps - self._last_resume_step >= 100_000:
            self._last_resume_step = self.num_timesteps
            self.model.save(str(self.run_dir / "resume_checkpoint"))
            print(f"resume_checkpoint.zip updated at step {self.num_timesteps}.")

        # Recording video
        if self.cfg.video_freq > 0 and self.num_timesteps - self._last_video_step >= self.cfg.video_freq:
            self._last_video_step = self.num_timesteps
            video_dir = str(self.run_dir / "videos" / f"step_{self.num_timesteps:08d}")
            record_video(self.model, self.cfg.env.env_id, self.seed, video_dir, self.cfg.env)
            print(f"Video recorded -> {video_dir}")

        return True


    def _run_eval(self) -> None:
        print(f"\n--- EVALUATION (Step {self.num_timesteps}) ---")
        env_cfg = self.cfg.env
        eval_env = make_eval_env(
            env_cfg.env_id,
            seed=self.seed + 1000,
            screen_size=env_cfg.screen_size,
            frame_skip=env_cfg.frame_skip,
            frame_stack=env_cfg.frame_stack,
        )
        rewards = []
        for i in range(self.cfg.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_r = 0.0
            ep_length = 0
            while not done:
                action, _ = self.model.predict(np.array(obs), deterministic=True)
                obs, r, terminated, truncated, _ = eval_env.step(int(action))
                ep_r += r
                ep_length += 1
                done = terminated or truncated
            rewards.append(ep_r)
            print(f"  Episode {i+1}/{self.cfg.eval_episodes}: score = {ep_r:.1f}, length = {ep_length}")

        eval_env.close()

        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        print(f"Eval Result: Mean reward = {mean_r:.1f} +/- {std_r:.1f}")

        if mean_r > self.best_eval_reward:
            print(f"New best model! ({self.best_eval_reward:.1f} -> {mean_r:.1f}). Saving...")
            self.best_eval_reward = mean_r
            self.model.save(str(self.run_dir / "best_model"))
        else:
            print(f"No improvement (Current best model: {self.best_eval_reward:.1f})")


# ----------------- Wrapper for tracking the true score -------------------
# Accumulates original_reward per episode and inserts it into info["true_score"]
# when the episode ends. Compatible with SB3 Monitor.
class TrueScoreWrapper(gymnasium.Wrapper):

    def reset(self, **kwargs):
        self._true_score = 0.0
        return self.env.reset(**kwargs)


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._true_score += info.get("original_reward", 0.0)
        if terminated or truncated:
            info["true_score"] = self._true_score
        return obs, reward, terminated, truncated, info


# ------------------------ Video recording function -----------------------
def record_video(model, env_id: str, seed: int, video_dir: str,
                 env_cfg: object, n_episodes: int = 1) -> None:
    vid_env = make_video_env(
        env_id, seed=seed + 2000, video_dir=video_dir,
        screen_size=env_cfg.screen_size,
        frame_skip=env_cfg.frame_skip,
        frame_stack=env_cfg.frame_stack,
    )
    for _ in range(n_episodes):
        obs, _ = vid_env.reset()
        done = False
        while not done:
            action, _ = model.predict(np.array(obs), deterministic=True)
            obs, _, terminated, truncated, _ = vid_env.step(int(action))
            done = terminated or truncated
    vid_env.close()


# -------------------- Main training ---------------------
def train(cfg: PPOConfig, seed: int, use_dummy: bool = False) -> None:
    set_seed(seed)
    device = get_device()
    run_dir = get_run_dir("runs", "ppo", seed)

    print(f"Training PPO | seed={seed} | device={device} | n_envs={cfg.n_envs} | run_dir={run_dir}")

    # Factory for each parallel environment
    def make_single_env(rank: int):
        def _init():
            # PPO-specific reward shaping. Key differences vs DQN's defaults:
            #   - step_penalty disabled (0.0): with DQN it acts as a "hurry up" signal,
            #     but PPO's on-policy training interprets a constant negative-per-step
            #     as "every action is bad" -- the policy collapses toward NOT MOVING.
            #     Removing it was necessary to get PPO to explore at all.
            #   - milestone_bonus disabled (0.0): PPO's entropy regularization already
            #     provides exploration pressure; extra exploration bonuses caused
            #     unstable advantage estimates.
            #   - health/death/level_clear scaled down by ~5x: PPO is more sensitive
            #     to large reward magnitudes than DQN (advantages get normalized);
            #     softer values produce more stable policy updates.
            ppo_rewards = {
                "step_penalty": 0.0,          # disabled for PPO (was -0.005 in DQN)
                "base_explore_reward": 0.5,   # same as DQN
                "milestone_bonus": 0.0,       # disabled for PPO (was 0.2 in DQN)
                "health_penalty": -0.02,      # softer than DQN's -0.1
                "death_penalty": -1.0,        # softer than DQN's -3.0
                "level_clear_reward": 5.0,    # softer than DQN's 25.0
                "boss_dmg_reward": 0.1,       # same as DQN
                "scale_factor": 1000.0,       # same as DQN
            }

            env = make_env(
                cfg.env.env_id,
                seed=seed + rank,
                clip_rewards=True,
                screen_size=cfg.env.screen_size,
                frame_skip=cfg.env.frame_skip,
                frame_stack=cfg.env.frame_stack,
                reward_kwargs=ppo_rewards,
            )
            env = TrueScoreWrapper(env)
            env = Monitor(env)  # adds info["episode"]["r"] and info["episode"]["l"] on episode end
            return env
        return _init

    # Vectorization: SubprocVecEnv by default for true parallelism (cluster).
    # DummyVecEnv as fallback for Colab/Windows where multiprocessing is fragile.
    env_fns = [make_single_env(i) for i in range(cfg.n_envs)]
    if use_dummy:
        venv = DummyVecEnv(env_fns)
        print("Using DummyVecEnv (sequential -- slower, but compatible everywhere)")
    else:
        venv = SubprocVecEnv(env_fns)
        print("Using SubprocVecEnv (process-level parallelism -- faster on cluster)")

    csv_logger = CSVLogger(run_dir, cfg)
    callback = TrainingCallback(cfg, run_dir, seed, csv_logger)

    # SB3 CNN Policy handles inputs (N, C, H, W) = (batch, channels, height, width)
    resume_path = run_dir / "resume_checkpoint.zip"

    if resume_path.exists():
        print(f"\nResuming from {resume_path}")
        model = PPO.load(
            str(run_dir / "resume_checkpoint"),
            env=venv,
            device=str(device),
        )
        # Resume best reward from metadata.json
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            callback.best_eval_reward = meta.get("best_eval_reward", -float("inf"))
        print(f"Resumed from step {model.num_timesteps}\n")
    else:
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            verbose=0,
            device=str(device),
            tensorboard_log=None,
        )

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callback,
            reset_num_timesteps=not resume_path.exists(),
        )
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Saving resume checkpoint...")
        model.save(str(run_dir / "resume_checkpoint"))
        print("Saved. Closing environments...")
        csv_logger.close()
        venv.close()
        return

    # End of training
    model.save(str(run_dir / "final_model"))
    print(f"Training completed. Best eval reward: {callback.best_eval_reward:.1f}")
    csv_logger.close()
    venv.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig()
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    train(cfg, seed=args.seed, use_dummy=args.use_dummy)