import random
import subprocess
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch

gym.register_envs(ale_py)


def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(env_id: str, seed: int, screen_size: int = 84, frame_skip: int = 4,
             frame_stack: int = 4, clip_rewards: bool = True) -> gym.Env:
    """Create an Atari env with the standard preprocessing stack for training."""
    env = gym.make(env_id, frameskip=1, render_mode=None)

    # Noop reset: take random number of no-ops at start
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=True,  # episodic life for training
        grayscale_obs=True,
        scale_obs=False,  # keep uint8, CNN normalizes in forward pass
    )

    # Clip rewards to {-1, 0, +1} to stabilize training across varying reward magnitudes (Mnih et al., 2015)
    if clip_rewards:
        env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))

    # Stack N consecutive frames so the agent can perceive motion and velocity from static images
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env.reset(seed=seed)
    return env


def make_eval_env(env_id: str, seed: int, screen_size: int = 84, frame_skip: int = 4,
                  frame_stack: int = 4) -> gym.Env:
    """Create an Atari env for evaluation — no life loss termination, no reward clipping."""
    env = gym.make(env_id, frameskip=1, render_mode=None)

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=False,  # use real episode boundaries
        grayscale_obs=True,
        scale_obs=False,
    )

    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env.reset(seed=seed)
    return env


def make_video_env(env_id: str, seed: int, video_dir: str, screen_size: int = 84,
                   frame_skip: int = 4, frame_stack: int = 4) -> gym.Env:
    """Create an eval env that records RGB video of gameplay."""
    env = gym.make(env_id, frameskip=1, render_mode="rgb_array")

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )

    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda _: True)
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env.reset(seed=seed)
    return env


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_run_dir(base_dir: str, algorithm: str, seed: int) -> Path:
    """Create and return a run directory: base_dir/algorithm/seed_N/."""
    run_dir = Path(base_dir) / algorithm / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
