import random
import subprocess
from pathlib import Path
import ale_py
import gymnasium as gym
import numpy as np
import torch

gym.register_envs(ale_py)

# Seed all RNGs for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------- This version of KungFuMasterRewardShaper doesn't implement walking reward ---------------
"""
    Wrapper for Reward Shaping:
    - Reward Scaling: (ex. 100 points -> 1.0)
    - Life Penalty: penalty for the death
    - Time Penalty: little penalty for each step in order to avoid camping
"""
"""
class KungFuMasterRewardShaper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale_factor: float = 100.0, life_penalty: float = -20.0, step_penalty: float = -0.01):
        super().__init__(env)
        self.scale_factor = scale_factor
        self.life_penalty = life_penalty
        self.step_penalty = step_penalty
        self.current_lives = 0

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self.current_lives = info.get("lives", 0)
        return obs, info

    def step(self, action: int) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Log saving
        info["original_reward"] = reward

        # Points scaling
        shaped_reward = reward / self.scale_factor

        # Time Penalty (-0.01 at each action)
        shaped_reward += self.step_penalty

        # Lives management
        lives = info.get("lives", self.current_lives)
        if lives < self.current_lives:
            shaped_reward += self.life_penalty
        self.current_lives = lives

        return obs, shaped_reward, terminated, truncated, info
"""

#---------------- Version of KungFuMasterRewardShaper with spacial rewards --------------------
"""
    Wrapper for Reward Shaping:
    - Reward Scaling: transforms game points (es. 100 -> 1.0)
    - Life Penalty: dying is bad
    - Time Penalty: avoiding camping
    - Explore Reward: reward if the agent is moving from the spawn point
"""
class KungFuMasterRewardShaper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale_factor: float = 100.0, life_penalty: float = -20.0, step_penalty: float = -0.01, explore_reward: float = 0.005):
        super().__init__(env)
        self.scale_factor = scale_factor
        self.life_penalty = life_penalty
        self.step_penalty = step_penalty
        self.explore_reward = explore_reward
        self.current_lives = 0
        
        # Spatial tracking (RAM[124] is the real X coordinate)
        self.x_byte_idx = 124
        self.last_x = 0
        self.cumulative_x = 0
        self.max_abs_x = 0

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self.current_lives = info.get("lives", 0)
        
        # Spawn point
        ram = self.env.unwrapped.ale.getRAM()
        self.last_x = int(ram[self.x_byte_idx])
        self.cumulative_x = 0
        self.max_abs_x = 0
        
        return obs, info

    def step(self, action: int) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Original log saving
        info["original_reward"] = reward

        # Fighting
        shaped_reward = reward / self.scale_factor

        # Time
        shaped_reward += self.step_penalty

        # Death
        lives = info.get("lives", self.current_lives)
        if lives < self.current_lives:
            shaped_reward += self.life_penalty
            self.current_lives = lives
            
            # Reset of the spawn after death
            ram = self.env.unwrapped.ale.getRAM()
            self.last_x = int(ram[self.x_byte_idx])
            self.cumulative_x = 0
            self.max_abs_x = 0
        else:
            # Exploration
            ram = self.env.unwrapped.ale.getRAM()
            current_x = int(ram[self.x_byte_idx])
            
            # Overflow management
            diff = current_x - self.last_x
            if diff > 128:
                diff -= 256
            elif diff < -128:
                diff += 256
                
            self.last_x = current_x
            self.cumulative_x += diff
            
            # Level 1 (left) -> cumulative_x gets positive 
            # Level 2 (right) -> cumulative_x gets negative. So we use abs()
            abs_distance = abs(self.cumulative_x)
            
            # Reward only for unseen space (of the current life)
            if abs_distance > self.max_abs_x:
                # Reward are correlated to the distance walked
                step_progress = abs_distance - self.max_abs_x
                shaped_reward += (step_progress * self.explore_reward)
                self.max_abs_x = abs_distance

        return obs, shaped_reward, terminated, truncated, info

def make_env(env_id: str, seed: int, screen_size: int = 84, frame_skip: int = 4,
            frame_stack: int = 4, clip_rewards: bool = True) -> gym.Env:
    env = gym.make(env_id, frameskip=1, render_mode=None)

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=frame_skip,
        screen_size=screen_size,
        # The agent plays since he has lives. He still has penalties for death
        terminal_on_life_loss=False, 
        grayscale_obs=True,
        scale_obs=False, 
    )

    if clip_rewards:
        env = KungFuMasterRewardShaper(
            env, 
            scale_factor=100.0, 
            life_penalty=-20.0, 
            step_penalty=-0.01, 
            explore_reward=0.005 
        )

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
