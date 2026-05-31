import random
import subprocess
from pathlib import Path
import ale_py
import gymnasium as gym
import numpy as np
import torch
import cv2

gym.register_envs(ale_py)


# Seed all RNGs for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Reducing noise deleting not playable areas
class MaskFasciaGioco(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = observation.copy()
        
        # First 37 lines (Score/Timer/ceil) and last 19 lines (floor) are completely black. Increasing the focus
        obs[:37, :] = 0  
        obs[65:, :] = 0  
        
        return obs


# -------------------------- Custom Pre-Processing Pipeline -------------------------
class KungFuPreprocessing(gym.Wrapper):

    def __init__(self, env: gym.Env, frame_skip: int = 4, noop_max: int = 30, screen_size: int = 84):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.noop_max = noop_max
        self.screen_size = screen_size
        
        self.knife_color = np.array([74, 74, 74], dtype=np.uint8)
        self.kernel = np.ones((3, 3), np.uint8)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(screen_size, screen_size), dtype=np.uint8
        )


    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        
        # Noop reset anti overfitting
        noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            obs, _, done, truncated, info = self.env.step(0)
            if done or truncated:
                obs, info = self.env.reset(**kwargs)
        
        return self._process_obs(obs), info


    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        done = False
        truncated = False
        obs_buffer = []
        
        # Frame skipping and Max Pooling as atari standard
        for i in range(self.frame_skip):
            obs, reward, d, t, info = self.env.step(action)
            total_reward += reward
            done = done or d
            truncated = truncated or t
            
            # Saving last 2 frames for max pooling
            if i >= self.frame_skip - 2:
                obs_buffer.append(obs)
                
            if done or truncated:
                if len(obs_buffer) == 0:
                    obs_buffer.append(obs)
                break
                
        if len(obs_buffer) == 2:
            max_obs = np.maximum(obs_buffer[0], obs_buffer[1])
        else:
            max_obs = obs_buffer[0]
            
        return self._process_obs(max_obs), total_reward, done, truncated, info


    # Utility for extracting RGB modify image without resize it
    def get_intermediate_rgb(self, obs: np.ndarray) -> np.ndarray:
        # Copy
        obs_highlighted = obs.copy()
        
        # Slice on the copy
        play_area = obs_highlighted[60:160, :]
        
        mask = cv2.inRange(play_area, self.knife_color, self.knife_color)
        dilated_mask = cv2.dilate(mask, self.kernel, iterations=1)
        
        # Knife pixels now are white in the copy
        play_area[dilated_mask > 0] = [255, 255, 255]
        
        return obs_highlighted


    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        # Processing RGB pixels
        obs_highlighted = self.get_intermediate_rgb(obs)
        # Grayscale image
        gray = cv2.cvtColor(obs_highlighted, cv2.COLOR_RGB2GRAY)
        # Resize 84x84
        resized = cv2.resize(gray, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        
        return resized


# ---------------------------- Reward Shaper ------------------------------
class KungFuMasterRewardShaper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale_factor: float = 1000.0, #base enemy is +0.2 or +0.1
                 step_penalty: float = -0.005, 
                 base_explore_reward: float = 0.5,
                 milestone_bonus: float = 0.2,
                 health_penalty: float = -0.1, 
                 death_penalty: float = -3.0,
                 level_clear_reward: float = 25.0,
                 boss_dmg_reward: float = 0.1):

        super().__init__(env)
        self.scale_factor = scale_factor
        self.step_penalty = step_penalty      
        self.base_explore_reward = base_explore_reward
        self.milestone_bonus = milestone_bonus
        self.health_penalty = health_penalty
        self.death_penalty = death_penalty
        self.level_clear_reward = level_clear_reward
        self.boss_dmg_reward = boss_dmg_reward
        
        self.milestone_byte_idx = 6
        self.health_byte_idx = 75
        self.lives_byte_idx = 29
        self.boss_health_byte_idx = 76
        
        self.visited_milestones = set()
        self.last_health = 0
        self.last_lives = 0
        self.last_boss_health = 0

        self.safe_zone = False


    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        ram = self.env.unwrapped.ale.getRAM()
        
        self.last_health = int(ram[self.health_byte_idx])
        self.last_lives = int(ram[self.lives_byte_idx])
        self.last_boss_health = int(ram[self.boss_health_byte_idx])
        
        self.visited_milestones = {int(ram[self.milestone_byte_idx])}
        self.safe_zone = False
        return obs, info

    # Reward logic
    def step(self, action: int) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward

        # Costant step penalty
        shaped_reward = self.step_penalty

        ram = self.env.unwrapped.ale.getRAM()
        current_health = int(ram[self.health_byte_idx])
        current_milestone = int(ram[self.milestone_byte_idx])
        current_lives = int(ram[self.lives_byte_idx])
        current_boss_health = int(ram[self.boss_health_byte_idx])

        # Completed level
        if info["original_reward"] >= 2000:
            self.safe_zone = True
            shaped_reward += self.level_clear_reward
            shaped_reward += (reward / self.scale_factor) 

        # Boss damage reward
        if current_boss_health < self.last_boss_health:
            hp_lost = self.last_boss_health - current_boss_health
            shaped_reward += (hp_lost * self.boss_dmg_reward)
        self.last_boss_health = current_boss_health

        # Death check
        if current_lives < self.last_lives or (self.last_lives == 0 and current_lives == 255):
            shaped_reward += self.death_penalty
            terminated = True

        # Check respawn
        if current_health > self.last_health:  
            if self.safe_zone:
                self.visited_milestones.clear()          
            self.safe_zone = False
            self.visited_milestones.add(current_milestone)

        # Check damage of the agent
        if current_health < self.last_health and not self.safe_zone:
            damage_taken = self.last_health - current_health
            shaped_reward += (damage_taken * self.health_penalty)
        self.last_health = current_health
        self.last_lives = current_lives

        # Progresion variable reward
        if current_milestone not in self.visited_milestones:
            milestones_cleared = len(self.visited_milestones)
            self.visited_milestones.add(current_milestone)
            scaled_reward = self.base_explore_reward + (milestones_cleared * self.milestone_bonus)
            shaped_reward += scaled_reward

        return obs, shaped_reward, terminated, truncated, info


# Training environment
def make_env(env_id: str, seed: int, screen_size: int = 84, frame_skip: int = 4,
            frame_stack: int = 4, clip_rewards: bool = True) -> gym.Env:
    # Creation of game environment
    env = gym.make(env_id, frameskip=1, render_mode=None, repeat_action_probability=0.0)

    # Custom AtariPreprocessing
    env = KungFuPreprocessing(
        env, 
        frame_skip=frame_skip, 
        noop_max=30, 
        screen_size=screen_size
    )
    
    # Removing ceil/floor
    env = MaskFasciaGioco(env)
    
    if clip_rewards:
        env = KungFuMasterRewardShaper(
            env, 
            scale_factor = 1000.0, #base enemy is +0.2 or +0.1
            step_penalty = -0.005, 
            base_explore_reward = 0.5,
            milestone_bonus = 0.2,
            health_penalty = -0.1, 
            death_penalty = -3.0,
            level_clear_reward = 25.0,
            boss_dmg_reward = 0.1
        )

    # Temporal stack of frames
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env.reset(seed=seed)
    return env


# Evaluation environment
def make_eval_env(env_id: str, seed: int, screen_size: int = 84, frame_skip: int = 4,
                frame_stack: int = 4) -> gym.Env:
    # Same environment of the training
    env = gym.make(env_id, frameskip=1, render_mode=None, repeat_action_probability=0.0)

    env = KungFuPreprocessing(
        env, 
        frame_skip=frame_skip, 
        noop_max=30, 
        screen_size=screen_size
    )
    env = MaskFasciaGioco(env)
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env.reset(seed=seed)
    return env


# Video environment (render_mode RGB)
def make_video_env(env_id: str, seed: int, video_dir: str, screen_size: int = 84,
                    frame_skip: int = 4, frame_stack: int = 4) -> gym.Env:
    # Same environment but render_mode="rgb_array" for recording video
    env = gym.make(env_id, frameskip=1, render_mode="rgb_array", repeat_action_probability=0.0)

    env = KungFuPreprocessing(
        env, 
        frame_skip=frame_skip, 
        noop_max=30, 
        screen_size=screen_size
    )
    env = MaskFasciaGioco(env)
    # RecordVideo writes mp4
    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda _: True)
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env.reset(seed=seed)
    return env


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_run_dir(base_dir: str, algorithm: str, seed: int) -> Path:
    run_dir = Path(base_dir) / algorithm / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
