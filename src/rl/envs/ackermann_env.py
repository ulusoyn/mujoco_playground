"""
Ackermann Robot RL Environment - Gymnasium-compatible environment for training

This environment provides:
- Observations: Lidar scans (72 beams), odometry, goal position
- Actions: cmd_vel commands (linear_x, angular_z)
- Rewards: Distance to goal, collision penalties, completion bonuses
- Dynamic maps: Random map loading from mobile-mujoco-environments
"""
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import sys
import random

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to path for imports
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from core.controller import BicycleController
from core.odometry import Odometry


class AckermannRobotEnv(gym.Env):
    """
    Ackermann Robot RL Environment
    
    Observation Space:
        - Lidar scans: 72 float values (0-12m, -1 if no hit)
        - Odometry: [x, y, heading] (3 floats)
        - Goal relative: [dx, dy, distance, angle_to_goal] (4 floats)
        Total: 79 floats
    
    Action Space:
        - linear_x: [-1.0, 1.0] (normalized forward/backward velocity)
        - angular_z: [-1.0, 1.0] (normalized angular velocity)
        Total: Box(2,) continuous actions
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        map_spawner=None,
        max_episode_steps=1000,
        goal_distance_threshold=0.5,
        collision_threshold=0.15,  # meters - lidar reading closer than this = collision
        max_linear_velocity=1.0,  # m/s
        max_angular_velocity=1.0,  # rad/s
        render_mode=None,
    ):
        """
        Initialize the RL environment
        
        Args:
            map_spawner: MapSpawner instance (will create if None)
            max_episode_steps: Maximum steps per episode
            goal_distance_threshold: Distance to goal for success (meters)
            collision_threshold: Lidar reading threshold for collision (meters)
            max_linear_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
            render_mode: 'human' or None
        """
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.goal_distance_threshold = goal_distance_threshold
        self.collision_threshold = collision_threshold
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.render_mode = render_mode
        
        # Initialize map spawner if not provided
        if map_spawner is None:
            # Try to use full map spawner, fallback to simple version
            try:
                from environments.map_spawner import MapSpawner
                self.map_spawner = MapSpawner()
            except ImportError:
                from rl.envs.simple_map_spawner import SimpleMapSpawner
                self.map_spawner = SimpleMapSpawner()
        else:
            self.map_spawner = map_spawner
        
        # Observation space: 72 lidar + 3 odom + 4 goal relative = 79 floats
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(79,),
            dtype=np.float32
        )
        
        # Action space: normalized [-1, 1] for linear_x and angular_z
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Initialize MuJoCo objects (will be set in reset)
        self.model = None
        self.data = None
        self.controller = None
        self.odometry = None
        
        # Episode tracking
        self.step_count = 0
        self.goal_position = None
        self.robot_start_position = None
        self.lidar_addrs = []
        self.lidar_site_ids = []
        
        # Viewer for rendering
        self.viewer = None
        
    def _setup_lidar(self):
        """Initialize lidar sensor addresses"""
        self.lidar_addrs = []
        self.lidar_site_ids = []
        
        for i in range(72):
            try:
                sname = f"lidar-{i}"
                site_name = f"rf-{i}"
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
                self.lidar_addrs.append(self.model.sensor_adr[sid])
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                self.lidar_site_ids.append(site_id)
            except Exception:
                self.lidar_addrs.append(-1)
                self.lidar_site_ids.append(-1)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.step_count = 0
        
        # Load random map with robot
        self.model, self.data, map_name = self.map_spawner.load_random_environment(
            robot_pos=[0, 0, 0.1],  # Spawn at origin
            robot_quat=None  # Random orientation
        )
        
        # Initialize controller and odometry
        self.controller = BicycleController(self.model, self.data)
        self.odometry = Odometry(self.model, self.data, robot_body_name="chassis")
        
        # Setup lidar
        self._setup_lidar()
        
        # Get initial robot position
        odom = self.odometry.calculate_odom()
        self.robot_start_position = np.array(odom['position'][:2])  # x, y only
        
        # Set random goal position (at least 2m away, within reasonable bounds)
        goal_distance = random.uniform(2.0, 8.0)
        goal_angle = random.uniform(0, 2 * np.pi)
        self.goal_position = self.robot_start_position + np.array([
            goal_distance * np.cos(goal_angle),
            goal_distance * np.sin(goal_angle)
        ])
        
        # Initialize viewer if rendering
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        observation = self._get_observation()
        info = {
            'map_name': map_name,
            'goal_position': self.goal_position.tolist(),
            'start_position': self.robot_start_position.tolist()
        }
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Convert normalized actions to actual velocities
        linear_x = action[0] * self.max_linear_velocity
        angular_z = action[1] * self.max_angular_velocity
        
        # Apply control
        self.controller.apply_cmd_vel(linear_x, angular_z)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Sync viewer if rendering - MUST happen after mj_step to show updated state
        if self.render_mode == "human" and self.viewer is not None:
            # Ensure data is up to date before syncing
            mujoco.mj_forward(self.model, self.data)
            # Sync the viewer to show current state
            self.viewer.sync()
        
        # Get observations
        observation = self._get_observation()
        
        # Calculate reward
        reward, terminated, truncated, info = self._calculate_reward()
        
        # Update step count
        self.step_count += 1
        
        # Check episode termination
        if self.step_count >= self.max_episode_steps:
            truncated = True
        
        # Update info
        info.update({
            'step': self.step_count,
            'linear_velocity': linear_x,
            'angular_velocity': angular_z,
        })
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation"""
        # Lidar readings (72 values)
        lidar_values = np.array([
            self.data.sensordata[addr] if addr >= 0 else -1.0
            for addr in self.lidar_addrs
        ], dtype=np.float32)
        
        # Odometry (x, y, heading)
        odom = self.odometry.calculate_odom()
        odom_values = np.array([
            odom['position'][0],  # x
            odom['position'][1],  # y
            odom['heading'],      # heading (radians)
        ], dtype=np.float32)
        
        # Goal relative information
        robot_pos = odom['position'][:2]
        goal_vec = self.goal_position - robot_pos
        goal_distance = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0]) - odom['heading']
        # Normalize angle to [-pi, pi]
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))
        
        goal_values = np.array([
            goal_vec[0],      # dx
            goal_vec[1],      # dy
            goal_distance,    # distance
            goal_angle,       # angle to goal relative to heading
        ], dtype=np.float32)
        
        # Concatenate all observations
        observation = np.concatenate([lidar_values, odom_values, goal_values])
        
        return observation
    
    def _calculate_reward(self):
        """Calculate reward, termination, and info"""
        odom = self.odometry.calculate_odom()
        robot_pos = odom['position'][:2]
        
        # Distance to goal
        goal_distance = np.linalg.norm(self.goal_position - robot_pos)
        
        # Check if goal reached
        terminated = goal_distance < self.goal_distance_threshold
        
        # Check for collision (any lidar reading too close)
        lidar_values = [
            self.data.sensordata[addr] if addr >= 0 else 999.0
            for addr in self.lidar_addrs
        ]
        min_lidar = min(lidar_values)
        collision = min_lidar < self.collision_threshold
        
        # Reward components
        reward = 0.0
        
        # Distance reward (negative, encourages getting closer)
        reward -= goal_distance * 0.1
        
        # Goal reached bonus
        if terminated:
            reward += 100.0
        
        # Collision penalty
        if collision:
            reward -= 50.0
        
        # Step penalty (encourage efficiency)
        reward -= 0.01
        
        # Truncated flag
        truncated = False
        
        info = {
            'goal_distance': goal_distance,
            'collision': collision,
            'min_lidar': min_lidar,
        }
        
        return reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
    
    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    # Test the environment
    env = AckermannRobotEnv(render_mode="human")
    
    print("Testing RL Environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Random actions test
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, distance={info['goal_distance']:.2f}")
        
        if terminated or truncated:
            print("Episode finished!")
            break
    
    env.close()

