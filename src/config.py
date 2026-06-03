from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # Shared environment configuration
    env_id: str = "ALE/KungFuMaster-v5"
    frame_stack: int = 4
    frame_skip: int = 4
    screen_size: int = 84
    clip_rewards: bool = True


# ---------------------- Laptop 16 GB ram DQN configuration ------------------------
@dataclass
class DQNConfig:
    total_timesteps: int = 15_000_000 
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128 
    
    # Avoiding swap on disk (max 16 GB RAM)
    buffer_size: int = 750_000   
    
    # learning_starts for the new buffer
    learning_starts: int = 80_000  
    
    train_freq: int = 4
    target_update_freq: int = 2500 

    # Epsilon
    epsilon_start: float = 1.0
    epsilon_mid: float = 0.10 
    epsilon_end: float = 0.01
    # Aggressive exploration
    epsilon_decay_phase1: int = 1_000_000 
    # Long fine tuning
    epsilon_decay_phase2: int = 5_000_000

    eval_freq: int = 100_000
    eval_episodes: int = 10
    checkpoint_freq: int = 500_000
    video_freq: int = 500_000
    env: EnvConfig = field(default_factory=EnvConfig)


# -------------------------- Desktop computer 64Gb ram DQN configuration ------------------------------
"""
@dataclass
class DQNConfig:
    total_timesteps: int = 15_000_000 
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128 
    
    # No problem of swap on disk
    buffer_size: int = 3_000_000   
    
    # learning_starts for the new buffer
    learning_starts: int = 80_000  
    
    train_freq: int = 4
    target_update_freq: int = 2500 

    # Epsilon
    epsilon_start: float = 1.0
    epsilon_mid: float = 0.10 
    epsilon_end: float = 0.01
    # Aggressive exploration
    epsilon_decay_phase1: int = 2_000_000 
    # Long fine tuning
    epsilon_decay_phase2: int = 8_000_000

    eval_freq: int = 100_000
    eval_episodes: int = 10
    checkpoint_freq: int = 500_000
    video_freq: int = 500_000
    env: EnvConfig = field(default_factory=EnvConfig)
"""


# ------------------------------ PPO configuration ----------------------------------
@dataclass
class PPOConfig:
    # PPO hyperparameters — SB3 Atari defaults
    total_timesteps: int = 15_000_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    n_steps: int = 128
    n_epochs: int = 4
    batch_size: int = 256
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 8
    video_freq: int = 500_000

    # Evaluation
    eval_freq: int = 100_000
    eval_episodes: int = 10

    # Checkpointing
    checkpoint_freq: int = 100_000

    # Environment
    env: EnvConfig = field(default_factory=EnvConfig)
