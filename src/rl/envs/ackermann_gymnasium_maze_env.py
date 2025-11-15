"""
Ackermann Robot in Gymnasium Robotics Maze - Proper Integration

This version properly integrates Gymnasium Robotics maze environments with our MuJoCo robot model.
"""
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import sys
import random
import xml.etree.ElementTree as ET
import tempfile

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


class AckermannGymnasiumMazeEnv(gym.Env):
    """
    Ackermann Robot in Gymnasium Robotics Maze Environment
    
    Uses Gymnasium Robotics maze structure and spawns our Ackermann robot within it.
    The maze structure is extracted from Gymnasium Robotics and used as a navigation target.
    
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
        maze_env_id="PointMaze_UMaze-v3",
        max_episode_steps=1000,
        goal_distance_threshold=0.5,
        collision_threshold=0.15,
        max_linear_velocity=1.0,
        max_angular_velocity=1.0,
        render_mode=None,
    ):
        """
        Initialize the RL environment
        
        Args:
            maze_env_id: Gymnasium Robotics maze environment ID
            max_episode_steps: Maximum steps per episode
            goal_distance_threshold: Distance to goal for success (meters)
            collision_threshold: Lidar reading threshold for collision (meters)
            max_linear_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
            render_mode: 'human' or None
        """
        super().__init__()
        
        self.maze_env_id = maze_env_id
        self.max_episode_steps = max_episode_steps
        self.goal_distance_threshold = goal_distance_threshold
        self.collision_threshold = collision_threshold
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.render_mode = render_mode
        
        # Initialize Gymnasium Robotics maze environment to get maze structure
        try:
            import gymnasium_robotics
            self.maze_env = gym.make(maze_env_id, render_mode=None)
            self.maze_env.reset()
            # Extract maze structure and get maze XML path
            self.maze_structure = self._extract_maze_structure()
            self.maze_xml_path = self.maze_env.unwrapped.tmp_xml_file_path
        except ImportError:
            raise ImportError(
                "gymnasium-robotics not installed!\n"
                "Install with: pip install gymnasium-robotics"
            )
        except Exception as e:
            raise ValueError(f"Failed to load maze environment '{maze_env_id}': {e}")
        
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
        
        # Load robot model
        self.robot_model_path = project_root / "models" / "ackermann_robot_v2.xml"
        
    def _extract_maze_structure(self):
        """Extract maze structure from Gymnasium Robotics environment"""
        # Reset maze to get initial state
        obs, info = self.maze_env.reset()
        
        # Extract goal and start positions
        if isinstance(obs, dict):
            start = obs.get('observation', obs.get('state', [0, 0]))[:2]
            goal = obs.get('desired_goal', obs.get('goal', [5, 5]))[:2]
        else:
            start = obs[:2]
            goal = info.get('goal', [5.0, 5.0])[:2]
        
        return {
            'start': np.array(start),
            'goal': np.array(goal),
            'bounds': self._get_maze_bounds(),
        }
    
    def _get_maze_bounds(self):
        """Get maze bounds from environment"""
        # Default bounds for common mazes
        maze_bounds = {
            'PointMaze_UMaze-v3': (-4, 4, -4, 4),
            'PointMaze-Open-v3': (-4, 4, -4, 4),
            'PointMaze-Medium-v3': (-8, 8, -8, 8),
            'PointMaze-Large-v3': (-12, 12, -12, 12),
        }
        
        if self.maze_env_id in maze_bounds:
            return maze_bounds[self.maze_env_id]
        
        # Default bounds
        return (-10, 10, -10, 10)
    
    def _load_robot_model(self, start_pos=None):
        """Load robot model merged with maze and set initial position"""
        if start_pos is None:
            start_pos = [0, 0, 0.1]
        
        # Merge maze and robot XML files
        merged_xml_path = self._merge_maze_and_robot_xml()
        
        # Load merged model
        self.model = mujoco.MjModel.from_xml_path(merged_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Find robot freejoint and set position
        # The robot body should be named "chassis" and have a freejoint
        # In MuJoCo, when a body has a freejoint, the freejoint position directly sets the body's world position
        # The body's pos attribute is NOT added to the freejoint position
        # So if chassis has pos="0 0 0.065", the freejoint position directly sets the body's origin
        # The body's pos is used for the body's local frame, but the freejoint sets the body's world position
        # So if we want chassis center at robot_z, and chassis has pos="0 0 0.065":
        #   freejoint_z directly sets the body's origin, and body's pos is relative to that
        #   But actually, the freejoint position IS the body's world position
        #   So we need to set freejoint_z = robot_z (the desired chassis center)
        robot_z = start_pos[2] if len(start_pos) > 2 else 0.1
        
        try:
            chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
            
            # Find freejoint associated with chassis
            for i in range(self.model.njnt):
                if self.model.jnt_bodyid[i] == chassis_id and self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    # Set position for freejoint (7 DOF: x, y, z, quat_w, quat_x, quat_y, quat_z)
                    joint_qpos_addr = self.model.jnt_qposadr[i]
                    # Set position: x, y from start_pos
                    self.data.qpos[joint_qpos_addr:joint_qpos_addr+2] = start_pos[0:2]
                    # For z: set freejoint directly to desired chassis center
                    # The freejoint position directly sets the body's world position
                    self.data.qpos[joint_qpos_addr+2] = robot_z
                    # Set orientation to identity quaternion [w, x, y, z] = [1, 0, 0, 0] (no rotation)
                    self.data.qpos[joint_qpos_addr+3:joint_qpos_addr+7] = [1, 0, 0, 0]
                    break
        except Exception as e:
            # Fallback: try to set first freejoint
            print(f"Warning: Could not find chassis freejoint, using first freejoint: {e}")
            for i in range(self.model.njnt):
                if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    joint_qpos_addr = self.model.jnt_qposadr[i]
                    self.data.qpos[joint_qpos_addr:joint_qpos_addr+2] = start_pos[0:2]
                    self.data.qpos[joint_qpos_addr+2] = robot_z
                    self.data.qpos[joint_qpos_addr+3:joint_qpos_addr+7] = [1, 0, 0, 0]
                    break
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Run a few physics steps to stabilize the robot and ensure proper ground contact
        # This helps the robot settle and prevents tilting
        # Use only 2-3 steps to avoid sinking too much
        for _ in range(3):
            mujoco.mj_step(self.model, self.data)
        
        # Initialize controller and odometry
        self.controller = BicycleController(self.model, self.data)
        self.odometry = Odometry(self.model, self.data, robot_body_name="chassis")
        
        # Setup lidar
        self._setup_lidar()
    
    def _merge_maze_and_robot_xml(self):
        """Merge maze XML with robot XML to create combined model"""
        # Parse maze XML
        maze_tree = ET.parse(self.maze_xml_path)
        maze_root = maze_tree.getroot()
        
        # Parse robot XML
        robot_tree = ET.parse(str(self.robot_model_path))
        robot_root = robot_tree.getroot()
        
        # Override maze compiler options with robot's settings
        # This is critical: maze uses angle="radian" but robot uses angle="degree"
        # If we don't override, all robot euler angles (lidar, wheels, etc.) will be wrong!
        robot_compiler = robot_root.find('compiler')
        maze_compiler = maze_root.find('compiler')
        if robot_compiler is not None:
            if maze_compiler is None:
                maze_compiler = ET.SubElement(maze_root, 'compiler')
            # Override maze compiler attributes with robot's (especially angle)
            for attr, value in robot_compiler.attrib.items():
                maze_compiler.set(attr, value)  # Override, don't just add
        
        # Override maze option settings with robot's settings
        # Robot needs gravity="0 0 -9.81" and proper timestep
        robot_option = robot_root.find('option')
        maze_option = maze_root.find('option')
        if robot_option is not None:
            if maze_option is None:
                maze_option = ET.SubElement(maze_root, 'option')
            # Override critical options: gravity and timestep
            for attr, value in robot_option.attrib.items():
                if attr in ['gravity', 'timestep']:
                    maze_option.set(attr, value)  # Override gravity and timestep
        
        # Get robot model directory for resolving relative paths
        robot_model_dir = self.robot_model_path.parent.resolve()
        
        # Helper function to convert relative paths to absolute
        def fix_path(elem, attr_name='file'):
            """Convert relative paths to absolute paths"""
            if attr_name in elem.attrib:
                path = elem.attrib[attr_name]
                if not Path(path).is_absolute():
                    # Resolve relative to robot model directory
                    abs_path = (robot_model_dir / path).resolve()
                    elem.attrib[attr_name] = str(abs_path)
        
        # Find worldbody in both
        maze_worldbody = maze_root.find('worldbody')
        robot_worldbody = robot_root.find('worldbody')
        
        if maze_worldbody is None or robot_worldbody is None:
            raise ValueError("Could not find worldbody in maze or robot XML")
        
        # Find robot chassis body (which contains freejoint)
        robot_chassis = robot_worldbody.find('.//body[@name="chassis"]')
        if robot_chassis is None:
            # Try to find body with freejoint
            for body in robot_worldbody.findall('body'):
                if body.find('freejoint') is not None:
                    robot_chassis = body
                    break
        
        if robot_chassis is None:
            raise ValueError("Could not find robot chassis body with freejoint in robot XML")
        
        # Copy robot assets to maze and fix paths
        maze_assets = maze_root.find('asset')
        robot_assets = robot_root.findall('asset')
        if maze_assets is None:
            maze_assets = ET.SubElement(maze_root, 'asset')
        
        for asset in robot_assets:
            for item in asset:
                # Check if item already exists
                existing = maze_assets.find(f".//{item.tag}[@name='{item.get('name', '')}']")
                if existing is None:
                    # Create a deep copy
                    item_copy = ET.fromstring(ET.tostring(item))
                    # Fix file paths (for mesh, texture, etc.)
                    fix_path(item_copy, 'file')
                    maze_assets.append(item_copy)
        
        # Set maze ground/floor to a fixed low position
        # Find ground geom and set it to a fixed low position
        ground_geom = maze_worldbody.find('geom[@name="ground"]')
        if ground_geom is not None:
            # Set ground to z = -0.5 (low enough for robot to sit on)
            pos_parts = ground_geom.get('pos', '0 0 -0.1').split()
            if len(pos_parts) >= 3:
                ground_geom.set('pos', f"{pos_parts[0]} {pos_parts[1]} -0.5")
        
        # Also set any other floor/ground geoms to the same height
        for geom in maze_worldbody.findall('geom'):
            geom_name = geom.get('name', '')
            if 'ground' in geom_name.lower() or 'floor' in geom_name.lower():
                pos_parts = geom.get('pos', '0 0 0').split()
                if len(pos_parts) >= 3:
                    geom.set('pos', f"{pos_parts[0]} {pos_parts[1]} -0.5")
        
        # Lower all maze block geoms (walls/obstacles) to sit on the ground
        # Blocks should be at ground level + half their height
        for geom in maze_worldbody.findall('geom'):
            geom_name = geom.get('name', '')
            if 'block' in geom_name.lower():
                current_pos = geom.get('pos', '0 0 0')
                pos_parts = current_pos.split()
                if len(pos_parts) >= 3:
                    # Get block size to calculate proper height
                    size_str = geom.get('size', '0.2 0.2 0.2')
                    size_parts = size_str.split()
                    if len(size_parts) >= 3:
                        block_height = float(size_parts[2])  # Half-height
                        # Position block so its bottom is at ground level (-0.5)
                        new_z = -0.5 + block_height
                        geom.set('pos', f"{pos_parts[0]} {pos_parts[1]} {new_z}")
                    else:
                        # Default: set to ground level + 0.2 (half of 0.4 height)
                        geom.set('pos', f"{pos_parts[0]} {pos_parts[1]} -0.3")
        
        # Copy robot chassis body (with all children) to maze worldbody
        # Keep the original pos attribute - it's part of the robot's internal structure
        # The freejoint will control the body's world position, and pos is relative to parent
        robot_body_copy = ET.fromstring(ET.tostring(robot_chassis))
        maze_worldbody.append(robot_body_copy)
        
        # Copy robot sensors if they exist
        maze_sensors = maze_root.find('sensor')
        robot_sensors = robot_root.find('sensor')
        if robot_sensors is not None:
            if maze_sensors is None:
                maze_sensors = ET.SubElement(maze_root, 'sensor')
            for sensor in robot_sensors:
                sensor_copy = ET.fromstring(ET.tostring(sensor))
                maze_sensors.append(sensor_copy)
        
        # Copy robot actuators if they exist
        maze_actuators = maze_root.find('actuator')
        robot_actuators = robot_root.find('actuator')
        if robot_actuators is not None:
            if maze_actuators is None:
                maze_actuators = ET.SubElement(maze_root, 'actuator')
            for actuator in robot_actuators:
                actuator_copy = ET.fromstring(ET.tostring(actuator))
                maze_actuators.append(actuator_copy)
        
        # Copy robot equality constraints if they exist
        maze_equality = maze_root.find('equality')
        robot_equality = robot_root.find('equality')
        if robot_equality is not None:
            if maze_equality is None:
                maze_equality = ET.SubElement(maze_root, 'equality')
            for constraint in robot_equality:
                constraint_copy = ET.fromstring(ET.tostring(constraint))
                maze_equality.append(constraint_copy)
        
        # Save merged XML to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        temp_file.write(ET.tostring(maze_root, encoding='unicode'))
        temp_file.close()
        
        return temp_file.name
        
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
        
        # Reset maze environment to get new goal/start
        maze_obs, maze_info = self.maze_env.reset(seed=seed)
        
        # Extract positions from maze
        if isinstance(maze_obs, dict):
            self.robot_start_position = np.array(maze_obs.get('observation', maze_obs.get('state', [0, 0]))[:2])
            self.goal_position = np.array(maze_obs.get('desired_goal', maze_obs.get('goal', [5, 5]))[:2])
        else:
            self.robot_start_position = np.array(maze_obs[:2])
            self.goal_position = np.array(maze_info.get('goal', [5.0, 5.0])[:2])
        
        # Load robot model at start position
        # Ground is at z=-0.5, so robot should spawn on it
        # Robot chassis has pos="0 0 0.065" relative to freejoint
        # Wheels are at z=-0.0325 relative to chassis, with radius 0.0325
        # For wheel centers to be at ground level + radius:
        #   wheel_center = ground_z + wheel_radius = -0.5 + 0.0325 = -0.4675
        #   wheel_center = chassis_center - 0.0325 (wheels are below chassis)
        #   So: chassis_center = wheel_center + 0.0325 = -0.4675 + 0.0325 = -0.435
        # But we need wheels to press slightly into ground for proper contact
        # So lower by 0.01m to ensure contact: chassis_center = -0.435 - 0.01 = -0.445
        # The _load_robot_model function expects robot_z to be the desired chassis center z
        # Spawn at -0.445 so wheels press into ground slightly for proper contact
        robot_spawn_z = -0.445
        self._load_robot_model(start_pos=[self.robot_start_position[0], 
                                          self.robot_start_position[1], 
                                          robot_spawn_z])
        
        # Initialize viewer if rendering
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
        
        observation = self._get_observation()
        info = {
            'maze_type': self.maze_env_id,
            'goal_position': self.goal_position.tolist(),
            'start_position': self.robot_start_position.tolist(),
            **maze_info
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
            # Update viewer with current data
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
        if hasattr(self.maze_env, 'close'):
            self.maze_env.close()
        # Clean up temporary merged XML file if it exists
        if hasattr(self, 'model') and self.model is not None:
            # The temp file was created in _merge_maze_and_robot_xml
            # It will be cleaned up by Python's tempfile module eventually
            pass


if __name__ == "__main__":
    # Test the environment
    try:
        env = AckermannGymnasiumMazeEnv(
            maze_env_id="PointMaze_UMaze-v3",
            render_mode="human"
        )
        
        print("Testing Ackermann Gymnasium Maze RL Environment...")
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
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo use Gymnasium Robotics maze environments, install:")
        print("  pip install gymnasium-robotics")


