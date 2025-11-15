# RL Package - Reinforcement Learning Components

## Structure

```
src/rl/
├── envs/
│   ├── __init__.py
│   ├── ackermann_env.py                    # Standard RL environment
│   ├── ackermann_gymnasium_maze_env.py     # Gymnasium Robotics maze integration ⭐ NEW
│   └── simple_map_spawner.py               # Fallback map loader
├── agents/                                 # Future: RL agent implementations
├── __init__.py
├── config.py                               # Configuration and hyperparameters
├── train.py                                # Training script
├── test_env.py                             # Environment test script
├── test_gymnasium_maze.py                  # Gymnasium maze test ⭐ NEW
├── utils.py                                # Utility functions
└── README.md                               # Documentation
```

## Environments

### 1. AckermannRobotEnv
Standard environment with dynamic map loading from mobile-mujoco-environments.

### 2. AckermannGymnasiumMazeEnv ⭐ NEW
Environment using **Gymnasium Robotics** maze structures:
- `PointMaze_UMaze-v3` - U-shaped maze
- `PointMaze-Open-v3` - Open maze
- `PointMaze-Medium-v3` - Medium complexity maze
- `PointMaze-Large-v3` - Large maze

## Usage

### Basic Training with Maze

```bash
# Train with Gymnasium Robotics UMaze
python src/rl/train.py --algo ppo --maze umaze --timesteps 100000

# Train with specific maze ID
python src/rl/train.py --algo sac --maze-id PointMaze-Medium-v3 --timesteps 100000

# Test maze environment
python src/rl/test_gymnasium_maze.py
```

### Python API

```python
from src.rl.envs import AckermannGymnasiumMazeEnv

# Create environment with UMaze
env = AckermannGymnasiumMazeEnv(
    maze_env_id="PointMaze_UMaze-v3",
    render_mode="human"
)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

## Installation

```bash
# Install Gymnasium Robotics for maze environments
pip install gymnasium-robotics

# Install Stable-Baselines3 for training
pip install stable-baselines3
```

## Environment Features

- ✅ Gymnasium Robotics maze integration
- ✅ Automatic goal/start position extraction from maze
- ✅ 72-beam lidar navigation
- ✅ Collision detection
- ✅ Goal reached detection
- ✅ Same observation/action space as standard environment

## Observation Space

- **72 lidar readings**: Distance to obstacles (0-12m)
- **3 odometry values**: x, y, heading
- **4 goal relative**: dx, dy, distance, angle_to_goal
- **Total**: 79 floats

## Action Space

- **linear_x**: [-1.0, 1.0] normalized velocity
- **angular_z**: [-1.0, 1.0] normalized angular velocity
- **Total**: Box(2,) continuous

## Rewards

- Distance to goal penalty: -0.1 × distance
- Goal reached bonus: +100
- Collision penalty: -50
- Step penalty: -0.01

## Next Steps

1. Add maze wall visualization
2. Integrate with full maze structure (walls as obstacles)
3. Add curriculum learning (easy → hard mazes)
4. Add multi-goal support
5. Add path planning integration
