import random
import subprocess
from pathlib import Path
import ale_py
import gymnasium as gym
import numpy as np
import torch

gym.register_envs(ale_py)

"""
    Personalized wrapper for Reward Shaping
    - Reward Scaling: transforms the points (50, 100, 2000) in useful values (0.5, 1.0, 20.0).
    - Death Penalty: strong signal of error if the character is hitten
"""

# --------------- This version of KungFuMasterRewardShaper doesn't implement walking reward ---------------
"""
class KungFuMasterRewardShaper(gym.Wrapper):

    def __init__(self, env: gym.Env, scale_factor: float = 100.0, life_penalty: float = -10.0):
        super().__init__(env)
        self.scale_factor = scale_factor
        self.life_penalty = life_penalty
        self.current_lives = 0

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self.current_lives = info.get("lives", 0)
        return obs, info

    def step(self, action: int) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Save the points
        info["original_reward"] = reward

        # Reward Scaling
        shaped_reward = reward / self.scale_factor

        # Death Penalty
        lives = info.get("lives", self.current_lives)
        if lives < self.current_lives:
            shaped_reward += self.life_penalty
            self.current_lives = lives

        return obs, shaped_reward, terminated, truncated, info
"""

#---------------- Version of KungFuMasterRewardShaper with spacial rewards --------------------
"""
    Wrapper for Reward Shaping
    - Reward Scaling: trasformation of videogame's points
    - Death Penalty: strong signal of error if the character is hitten
    - Scroll Reward (RAM): uses game's RAM informations in order to guide the character in the level
"""
class KungFuMasterRewardShaper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale_factor: float = 100.0, life_penalty: float = -10.0, scroll_reward: float = 2.0):
        super().__init__(env)
        self.scale_factor = scale_factor
        self.life_penalty = life_penalty
        self.scroll_reward = scroll_reward
        self.current_lives = 0
        
        # Scroll monitoring
        self.last_raw_scroll = 0
        self.cumulative_scroll = 0
        self.max_cumulative_scroll = 0

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self.current_lives = info.get("lives", 0)
        
        self.last_raw_scroll = self.env.unwrapped.ale.getRAM()[3]
        self.cumulative_scroll = 0
        self.max_cumulative_scroll = 0
        return obs, info

    def step(self, action: int) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward
        shaped_reward = reward / self.scale_factor

        ram = self.env.unwrapped.ale.getRAM()
        current_raw_scroll = ram[3]
        lives = info.get("lives", self.current_lives)

        if lives < self.current_lives:
            shaped_reward += self.life_penalty
            self.current_lives = lives
            
            # Allignment between scroll and the new life
            self.last_raw_scroll = current_raw_scroll
            self.cumulative_scroll = 0
            self.max_cumulative_scroll = 0
        else:
            # Unwrapping logic in order to avoid RAM wrap problems
            raw_diff = current_raw_scroll - self.last_raw_scroll
            self.last_raw_scroll = current_raw_scroll
            
            if raw_diff < -4:
                raw_diff += 8
            elif raw_diff > 4:
                raw_diff -= 8
                
            self.cumulative_scroll += raw_diff
            
            # High-Water Mark method on straight line
            if self.cumulative_scroll > self.max_cumulative_scroll:
                shaped_reward += self.scroll_reward
                self.max_cumulative_scroll = self.cumulative_scroll

        return obs, shaped_reward, terminated, truncated, info

def make_env(env_id: str, seed: int, screen_size: int = 84, frame_skip: int = 4,
            frame_stack: int = 4, clip_rewards: bool = True) -> gym.Env:
    env = gym.make(env_id, frameskip=1, render_mode=None)

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False, 
    )

    if clip_rewards:
        env = KungFuMasterRewardShaper(env, scale_factor=100.0, life_penalty=-10.0)

    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env.reset(seed=seed)
    return env

def make_eval_env(env_id: str, seed: int, screen_size: int = 84, frame_skip: int = 4,
                frame_stack: int = 4) -> gym.Env:
    # Create an Atari env for evaluation
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
    # Create an eval env that records RGB video of gameplay
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
    # Return the best available device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_git_hash() -> str:
    # Return the current git commit hash, or 'unknown' if not in a repo
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_run_dir(base_dir: str, algorithm: str, seed: int) -> Path:
    # Create and return a run directory: base_dir/algorithm/seed_N/
    run_dir = Path(base_dir) / algorithm / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
