"""
RL Configuration File

Contains hyperparameters and settings for RL training
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RLConfig:
    """Configuration for RL training"""
    
    # Environment
    max_episode_steps: int = 1000
    goal_distance_threshold: float = 0.5  # meters
    collision_threshold: float = 0.15  # meters
    max_linear_velocity: float = 1.0  # m/s
    max_angular_velocity: float = 1.0  # rad/s
    
    # Reward weights
    distance_weight: float = -0.1
    goal_bonus: float = 100.0
    collision_penalty: float = -50.0
    step_penalty: float = -0.01
    
    # PPO hyperparameters
    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.01
    
    # SAC hyperparameters
    sac_learning_rate: float = 3e-4
    sac_buffer_size: int = 100000
    sac_learning_starts: int = 1000
    sac_batch_size: int = 256
    sac_tau: float = 0.005
    sac_gamma: float = 0.99
    
    # Training
    total_timesteps: int = 100000
    eval_freq: int = 10000
    save_freq: int = 10000
    log_dir: str = "rl_logs"
    
    # Rendering
    render_training: bool = False
    render_eval: bool = True


# Default configuration
default_config = RLConfig()






