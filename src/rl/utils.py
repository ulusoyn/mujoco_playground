"""
RL Utilities - Helper functions for RL training and evaluation
"""
import numpy as np
from typing import List, Dict, Tuple


def compute_episode_stats(returns: List[float], lengths: List[int]) -> Dict[str, float]:
    """Compute statistics from episode returns and lengths"""
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
    }


def evaluate_agent(env, agent, num_episodes: int = 10) -> Dict[str, float]:
    """Evaluate an agent on the environment"""
    returns = []
    lengths = []
    successes = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if hasattr(agent, 'predict'):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                returns.append(total_reward)
                lengths.append(steps)
                successes.append(terminated)  # True if goal reached
                break
    
    stats = compute_episode_stats(returns, lengths)
    stats['success_rate'] = np.mean(successes)
    
    return stats


def normalize_observation(obs: np.ndarray, obs_space) -> np.ndarray:
    """Normalize observation to [0, 1] range"""
    if hasattr(obs_space, 'low') and hasattr(obs_space, 'high'):
        low = obs_space.low
        high = obs_space.high
        # Avoid division by zero
        range_vals = high - low
        range_vals[range_vals == 0] = 1.0
        normalized = (obs - low) / range_vals
        return normalized
    return obs


def create_action_mapping(action_space, max_linear: float = 1.0, max_angular: float = 1.0):
    """Create a function to map normalized actions to actual velocities"""
    def map_action(action: np.ndarray) -> Tuple[float, float]:
        linear_x = action[0] * max_linear
        angular_z = action[1] * max_angular
        return linear_x, angular_z
    return map_action






