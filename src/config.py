from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Shared environment configuration."""
    env_id: str = "ALE/KungFuMaster-v5"
    frame_stack: int = 4
    frame_skip: int = 4  # aka action_repeat
    screen_size: int = 84
    clip_rewards: bool = True


@dataclass
class DQNConfig:
    """DQN hyperparameters — Nature DQN defaults for Atari."""
    # Training
    total_timesteps: int = 1_000_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    train_freq: int = 4  # learn every N env steps

    # Target network
    target_update_freq: int = 1_000  # copy policy → target every N learning steps

    # Exploration (linear epsilon decay)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000

    # Evaluation
    eval_freq: int = 10_000  # evaluate every N env steps
    eval_episodes: int = 10

    # Checkpointing
    checkpoint_freq: int = 50_000  # save every N env steps

    # Video recording
    video_freq: int = 100_000  # record a gameplay video every N env steps (0 to disable)

    # Environment
    env: EnvConfig = field(default_factory=EnvConfig)


@dataclass
class PPOConfig:
    """PPO hyperparameters — SB3 Atari defaults."""
    total_timesteps: int = 1_000_000
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

    # Evaluation
    eval_freq: int = 10_000
    eval_episodes: int = 10

    # Checkpointing
    checkpoint_freq: int = 50_000

    # Environment
    env: EnvConfig = field(default_factory=EnvConfig)
