"""RL Package"""
from .envs import AckermannRobotEnv
from .make_env import make_ackermann_env, list_available_mazes

# Try to import maze environment
try:
    from .envs import AckermannGymnasiumMazeEnv
    __all__ = ['AckermannRobotEnv', 'AckermannGymnasiumMazeEnv', 'make_ackermann_env', 'list_available_mazes']
except ImportError:
    __all__ = ['AckermannRobotEnv', 'make_ackermann_env', 'list_available_mazes']


