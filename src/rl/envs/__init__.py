"""RL Environments Package"""
from .ackermann_env import AckermannRobotEnv

# Try to import Gymnasium Robotics maze environment
try:
    from .ackermann_gymnasium_maze_env import AckermannGymnasiumMazeEnv
    __all__ = ['AckermannRobotEnv', 'AckermannGymnasiumMazeEnv']
except ImportError:
    __all__ = ['AckermannRobotEnv']


