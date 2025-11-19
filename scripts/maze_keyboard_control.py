"""
Keyboard Teleop Control for Robot in Maze Environment

This script spawns the robot directly in a Gymnasium maze environment without
using the RL training wrapper. It merges the robot model with the maze and
allows keyboard control through the MuJoCo viewer.

Controls (when MuJoCo viewer window is focused):
    Arrow Keys / NumPad:
        ↑ / 8 - Increase forward velocity
        ↓ / 2 - Increase backward velocity
        ← / 4 - Turn left (increase angular velocity)
        → / 6 - Turn right (decrease angular velocity)
        5 - Stop (set velocities to zero)
    
    WASD:
        W - Forward
        S - Backward
        A - Turn left
        D - Turn right
        Space - Stop
    
    Other:
        R - Reset environment
        Q / ESC - Quit

Usage:
    python scripts/maze_keyboard_control.py --maze umaze
"""
import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
import sys
import glfw
import xml.etree.ElementTree as ET
import tempfile

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to path for imports
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from core.controller import BicycleController
from core.odometry import Odometry
from teleop.keyboard_teleop import MujocoTeleop


def merge_maze_and_robot_xml(maze_xml_path, robot_xml_path):
    """Merge maze XML with robot XML to create combined model"""
    # Parse maze XML
    maze_tree = ET.parse(maze_xml_path)
    maze_root = maze_tree.getroot()
    
    # Parse robot XML
    robot_tree = ET.parse(str(robot_xml_path))
    robot_root = robot_tree.getroot()
    
    # Override maze compiler options with robot's settings
    robot_compiler = robot_root.find('compiler')
    maze_compiler = maze_root.find('compiler')
    if robot_compiler is not None:
        if maze_compiler is None:
            maze_compiler = ET.SubElement(maze_root, 'compiler')
        for attr, value in robot_compiler.attrib.items():
            maze_compiler.set(attr, value)
    
    # Override maze option settings with robot's settings
    robot_option = robot_root.find('option')
    maze_option = maze_root.find('option')
    if robot_option is not None:
        if maze_option is None:
            maze_option = ET.SubElement(maze_root, 'option')
        for attr, value in robot_option.attrib.items():
            if attr in ['gravity', 'timestep']:
                maze_option.set(attr, value)
    
    # Get robot model directory for resolving relative paths
    robot_model_dir = robot_xml_path.parent.resolve()
    
    # Helper function to convert relative paths to absolute
    def fix_path(elem, attr_name='file'):
        """Convert relative paths to absolute paths"""
        if attr_name in elem.attrib:
            path = elem.attrib[attr_name]
            if not Path(path).is_absolute():
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
            existing = maze_assets.find(f".//{item.tag}[@name='{item.get('name', '')}']")
            if existing is None:
                item_copy = ET.fromstring(ET.tostring(item))
                fix_path(item_copy, 'file')
                maze_assets.append(item_copy)
    
    # Remove Ant robot body from maze (if present) - we're replacing it with our robot
    # Ant robot typically has bodies named "torso" or contains joints like "hip_1", "ankle_1", etc.
    ant_bodies_to_remove = []
    for body in maze_worldbody.findall('body'):
        body_name = body.get('name', '')
        # Check if this body is part of the Ant robot
        if body_name in ['torso'] or body.find('.//joint[@name="hip_1"]') is not None:
            ant_bodies_to_remove.append(body)
    
    for body in ant_bodies_to_remove:
        maze_worldbody.remove(body)
        print(f"Removed Ant robot body: {body.get('name', 'unknown')}")
    
    # Also remove Ant actuators and sensors if present
    maze_actuators = maze_root.find('actuator')
    if maze_actuators is not None:
        ant_actuators = []
        for actuator in maze_actuators.findall('motor'):
            ant_actuators.append(actuator)
        for actuator in ant_actuators:
            maze_actuators.remove(actuator)
            print(f"Removed Ant actuator: {actuator.get('name', 'unknown')}")
    
    maze_sensors = maze_root.find('sensor')
    if maze_sensors is not None:
        ant_sensors = []
        for sensor in maze_sensors.findall('*'):
            if sensor.tag in ['touch', 'accelerometer', 'velocimeter', 'gyro']:
                ant_sensors.append(sensor)
        for sensor in ant_sensors:
            maze_sensors.remove(sensor)
            print(f"Removed Ant sensor: {sensor.get('name', 'unknown')}")
    
    # Set maze ground/floor to a fixed low position and ensure good friction and collision
    ground_geom = maze_worldbody.find('geom[@name="ground"]')
    if ground_geom is not None:
        pos_parts = ground_geom.get('pos', '0 0 -0.1').split()
        if len(pos_parts) >= 3:
            ground_geom.set('pos', f"{pos_parts[0]} {pos_parts[1]} -0.5")
        # Ensure ground has good friction for robot wheels
        if 'friction' not in ground_geom.attrib:
            ground_geom.set('friction', '5.0 0.3 0.02')
        else:
            # Update friction to ensure good grip - match wheel friction
            ground_geom.set('friction', '5.0 0.3 0.02')
        # Ensure ground collision settings match wheels (contype="1", conaffinity="1" or higher)
        if 'contype' not in ground_geom.attrib:
            ground_geom.set('contype', '1')
        else:
            ground_geom.set('contype', '1')  # Ensure it's 1 to match wheels
        if 'conaffinity' not in ground_geom.attrib:
            ground_geom.set('conaffinity', '1')
        else:
            # Ensure conaffinity includes bit 0 (value 1) to match wheel contype="1"
            current_affinity = ground_geom.get('conaffinity', '1')
            try:
                affinity_val = int(current_affinity)
                # Ensure bit 0 is set (value 1 or higher)
                if affinity_val == 0:
                    ground_geom.set('conaffinity', '1')
                # If already >= 1, keep it but ensure it's at least 1
                elif affinity_val < 1:
                    ground_geom.set('conaffinity', '1')
            except ValueError:
                ground_geom.set('conaffinity', '1')
    
    # Lower all maze block geoms (walls/obstacles) to sit on the ground
    for geom in maze_worldbody.findall('geom'):
        geom_name = geom.get('name', '')
        if 'block' in geom_name.lower():
            current_pos = geom.get('pos', '0 0 0')
            pos_parts = current_pos.split()
            if len(pos_parts) >= 3:
                size_str = geom.get('size', '0.2 0.2 0.2')
                size_parts = size_str.split()
                if len(size_parts) >= 3:
                    block_height = float(size_parts[2])
                    new_z = -0.5 + block_height
                    geom.set('pos', f"{pos_parts[0]} {pos_parts[1]} {new_z}")
                else:
                    geom.set('pos', f"{pos_parts[0]} {pos_parts[1]} -0.3")
    
    # Copy robot chassis body to maze worldbody
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
    
    # Ensure all joint ranges are valid (range[0] < range[1])
    for joint in maze_root.findall('.//joint'):
        if 'range' in joint.attrib:
            range_str = joint.attrib['range']
            try:
                range_parts = range_str.split()
                if len(range_parts) == 2:
                    range_min = float(range_parts[0])
                    range_max = float(range_parts[1])
                    if range_min >= range_max:
                        # Remove invalid range
                        del joint.attrib['range']
                        print(f"Warning: Removed invalid range from joint {joint.get('name', 'unknown')}")
            except (ValueError, IndexError):
                # Remove malformed range
                del joint.attrib['range']
                print(f"Warning: Removed malformed range from joint {joint.get('name', 'unknown')}")
    
    # Save merged XML to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
    merged_xml_content = ET.tostring(maze_root, encoding='unicode')
    temp_file.write(merged_xml_content)
    temp_file.close()
    
    # Also save to a debug file for inspection
    debug_file = project_root / 'merged_debug.xml'
    with open(debug_file, 'w') as f:
        f.write(merged_xml_content)
    print(f"Debug: Saved merged XML to {debug_file}")
    
    return temp_file.name


def find_valid_spawn_position(model, data, start_pos, robot_spawn_z, clearance=0.2):
    """
    Find a valid spawn position that's not inside walls.
    Robot footprint is roughly 0.275m x 0.15m, so we need clearance around it.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        start_pos: [x, y] starting position from maze
        robot_spawn_z: z height for robot
        clearance: minimum clearance from walls (meters)
    
    Returns:
        [x, y] valid position
    """
    # Get freejoint address
    freejoint_addr = None
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            freejoint_addr = model.jnt_qposadr[i]
            break
    
    if freejoint_addr is None:
        return start_pos
    
    # Try the original position first
    test_positions = [
        start_pos,  # Original position
        start_pos + np.array([0.1, 0.0]),  # Move right
        start_pos + np.array([-0.1, 0.0]),  # Move left
        start_pos + np.array([0.0, 0.1]),  # Move up
        start_pos + np.array([0.0, -0.1]),  # Move down
        start_pos + np.array([0.1, 0.1]),  # Move right-up
        start_pos + np.array([-0.1, 0.1]),  # Move left-up
        start_pos + np.array([0.1, -0.1]),  # Move right-down
        start_pos + np.array([-0.1, -0.1]),  # Move left-down
        start_pos + np.array([0.2, 0.0]),  # Move further right
        start_pos + np.array([-0.2, 0.0]),  # Move further left
        start_pos + np.array([0.0, 0.2]),  # Move further up
        start_pos + np.array([0.0, -0.2]),  # Move further down
    ]
    
    # Try each position and check for collisions
    for test_pos in test_positions:
        # Reset data to initial state
        mujoco.mj_resetData(model, data)
        
        # Set position
        data.qpos[freejoint_addr:freejoint_addr+2] = test_pos[0:2]
        data.qpos[freejoint_addr+2] = robot_spawn_z
        data.qpos[freejoint_addr+3:freejoint_addr+7] = [1, 0, 0, 0]
        
        # Forward kinematics to compute contacts
        mujoco.mj_forward(model, data)
        
        # Check for collisions (ncon is number of contacts)
        # We want positions with only ground contacts (no wall contacts)
        # Ground contact is expected: 4 wheels + possibly Base = up to 5-6 contacts
        # If we have many more contacts, likely colliding with walls
        # Use a reasonable threshold: if contacts > 8, probably hitting walls
        initial_contacts = data.ncon
        
        # Run a couple steps to ensure stability
        for _ in range(2):
            mujoco.mj_step(model, data)
        
        # Check again after settling
        mujoco.mj_forward(model, data)
        final_contacts = data.ncon
        
        # If contacts are reasonable (just ground), this position is valid
        # Allow up to 8 contacts to account for wheels and base on ground
        if final_contacts <= 8:
            return test_pos
    
    # If all positions failed, return original (user will see the issue)
    print(f"Warning: Could not find collision-free position near {start_pos}")
    return start_pos


def load_robot_in_maze(maze_xml_path, robot_xml_path, start_pos, spawn_override=None):
    """
    Load merged maze+robot model and position robot at start_pos
    
    Args:
        maze_xml_path: Path to maze XML file
        robot_xml_path: Path to robot XML file
        start_pos: [x, y] starting position from maze (or override)
        spawn_override: Optional [x, y] to override start_pos
    """
    # Merge XMLs
    merged_xml_path = merge_maze_and_robot_xml(maze_xml_path, robot_xml_path)
    
    # Load merged model
    try:
        model = mujoco.MjModel.from_xml_path(merged_xml_path)
    except Exception as e:
        print(f"Error loading merged XML from {merged_xml_path}")
        print(f"Error details: {e}")
        print(f"Please check merged_debug.xml for the merged XML content")
        raise
    data = mujoco.MjData(model)
    
    # Calculate robot spawn height (ground is at z=-0.5)
    # Robot chassis has pos="0 0 0.065" relative to freejoint
    # Wheels are at z=-0.0325 relative to chassis, with radius 0.0325
    # For wheel centers to be at ground level + radius:
    #   wheel_center = ground_z + wheel_radius = -0.5 + 0.0325 = -0.4675
    #   wheel_center = chassis_center - 0.0325 (wheels are below chassis)
    #   So: chassis_center = wheel_center + 0.0325 = -0.4675 + 0.0325 = -0.435
    # But we need wheels to press slightly into ground for proper contact
    # So lower by 0.01m to ensure contact: chassis_center = -0.435 - 0.01 = -0.445
    # The freejoint position directly sets the chassis body's world position
    robot_spawn_z = 0.3
    
    # Use spawn override if provided, otherwise use maze start position
    if spawn_override is not None:
        adjusted_pos = np.array(spawn_override)
        print(f"Using override spawn position: ({adjusted_pos[0]:.2f}, {adjusted_pos[1]:.2f})")
    else:
        # Use the maze's suggested start position
        adjusted_pos = np.array(start_pos)
        print(f"Using maze start position: ({adjusted_pos[0]:.2f}, {adjusted_pos[1]:.2f})")
    
    # Set robot position using freejoint
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            joint_qpos_addr = model.jnt_qposadr[i]
            # Set position: x, y from adjusted_pos, z from calculated height
            data.qpos[joint_qpos_addr:joint_qpos_addr+2] = adjusted_pos[0:2]
            data.qpos[joint_qpos_addr+2] = robot_spawn_z
            # Set orientation to identity quaternion [w, x, y, z] = [1, 0, 0, 0]
            data.qpos[joint_qpos_addr+3:joint_qpos_addr+7] = [1, 0, 0, 0]
            break
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Run a few physics steps to stabilize the robot
    for _ in range(3):
        mujoco.mj_step(model, data)
    
    return model, data, merged_xml_path


def main():
    parser = argparse.ArgumentParser(description='Keyboard Teleop Control in Maze')
    parser.add_argument('--maze', type=str, default='umaze',
                        choices=['umaze', 'large', 'medium'],
                        help='Maze type')
    parser.add_argument('--maze-id', type=str, default='PointMaze_UMaze-v3',
                        help='Gymnasium Robotics maze environment ID')
    parser.add_argument('--max-velocity', type=float, default=1.0,
                        help='Maximum linear velocity (m/s)')
    parser.add_argument('--max-angular-velocity', type=float, default=1.0,
                        help='Maximum angular velocity (rad/s)')
    parser.add_argument('--linear-increment', type=float, default=0.1,
                        help='Linear velocity increment per key press')
    parser.add_argument('--angular-increment', type=float, default=0.3,
                        help='Angular velocity increment per key press')
    parser.add_argument('--spawn-x', type=float, default=None,
                        help='X coordinate for robot spawn position (overrides maze start position)')
    parser.add_argument('--spawn-y', type=float, default=None,
                        help='Y coordinate for robot spawn position (overrides maze start position)')
    
    args = parser.parse_args()
    
    # Map maze type to environment ID
    # Available Gymnasium Robotics AntMaze environments (v5):
    # - AntMaze_UMaze-v5, AntMaze_UMazeDense-v5
    # - AntMaze_Medium-v5, AntMaze_MediumDense-v5
    # - AntMaze_Large-v5, AntMaze_LargeDense-v5
    # - AntMaze_Open-v5, AntMaze_OpenDense-v5
    # - PointMaze_UMaze-v3, PointMaze_Open-v3, PointMaze_Medium-v3, PointMaze_Large-v3
    maze_map = {
        'umaze': 'AntMaze_UMaze-v5',
        'medium': 'AntMaze_Medium-v5',  # Large is the biggest available
        'large': 'AntMaze_Large-v5',  # Large is the most complex available
    }
    
    # Use --maze-id if explicitly provided (not the default), otherwise use --maze mapping
    # Check if maze_id was explicitly set via command line
    import sys
    if '--maze-id' in sys.argv:
        maze_id = args.maze_id
    else:
        maze_id = maze_map.get(args.maze, args.maze_id)
    
    print(f"\n{'='*60}")
    print("Keyboard Teleop Control in Maze")
    print(f"{'='*60}")
    print(f"Maze: {maze_id}")
    print(f"Max velocity: {args.max_velocity} m/s")
    print(f"Max angular velocity: {args.max_angular_velocity} rad/s")
    print(f"\nControls (focus MuJoCo viewer window):")
    print(f"  Arrow Keys / NumPad 8/2/4/6 - Control robot")
    print(f"  WASD - Alternative controls")
    print(f"  Space / NumPad 5 - Stop")
    print(f"  R - Reset")
    print(f"  Q / ESC - Quit")
    print(f"{'='*60}\n")
    
    # Create Gymnasium maze environment to get maze XML
    try:
        import gymnasium as gym
        import gymnasium_robotics  # This import automatically registers all gymnasium-robotics environments
        
        # Now you can use any gymnasium-robotics environment ID directly
        maze_env = gym.make(maze_id, render_mode=None)
        maze_env.reset()
        maze_xml_path = maze_env.unwrapped.tmp_xml_file_path
        
    except ImportError:
        print("Error: gymnasium-robotics not installed!")
        print("Install with: pip install gymnasium-robotics")
        return
    except Exception as e:
        print(f"Error creating maze environment: {e}")
        return
    
    # Get robot model path
    robot_model_path = project_root / "models" / "ackermann_robot_v2.xml"
    if not robot_model_path.exists():
        print(f"Error: Robot model not found at {robot_model_path}")
        return
    
    # Get start position from maze environment
    maze_obs, maze_info = maze_env.reset()
    if isinstance(maze_obs, dict):
        start_pos_2d = np.array(maze_obs.get('observation', maze_obs.get('state', [0, 0]))[:2])
    else:
        start_pos_2d = np.array(maze_obs[:2])
    
    print(f"Maze suggested spawn position: ({start_pos_2d[0]:.2f}, {start_pos_2d[1]:.2f})")
    
    # Determine spawn position (use override if provided, otherwise use maze suggestion)
    if args.spawn_x is not None or args.spawn_y is not None:
        spawn_pos = [
            args.spawn_x if args.spawn_x is not None else start_pos_2d[0],
            args.spawn_y if args.spawn_y is not None else start_pos_2d[1]
        ]
    else:
        spawn_pos = None  # Use maze start position
    
    # Load merged model
    try:
        model, data, merged_xml_path = load_robot_in_maze(
            maze_xml_path, robot_model_path, start_pos_2d, spawn_override=spawn_pos
        )
    except Exception as e:
        print(f"Error loading robot in maze: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize controller and odometry
    controller = BicycleController(model, data)
    odometry = Odometry(model, data, robot_body_name="chassis")

    print("Actuators in merged model:")
    for i in range(model.nu):
        print(i, model.actuator(i).name)

    
    # Create teleop controller
    teleop = MujocoTeleop(
        linear_increment=args.linear_increment,
        angular_increment=args.angular_increment,
        linear_limit=args.max_velocity,
        angular_limit=args.max_angular_velocity
    )
    
    # Control state
    should_reset = False
    should_quit = False
    
    # Keyboard callback for MuJoCo viewer
    def key_callback(keycode):
        nonlocal should_reset, should_quit
        
        # Handle teleop keys
        teleop.key_callback(keycode)
        
        # Additional controls
        if keycode == glfw.KEY_R:
            should_reset = True
            print("Reset requested...")
        elif keycode in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            should_quit = True
            print("Quit requested...")
        elif keycode == glfw.KEY_SPACE:
            teleop.cmd_vel["linear_x"] = 0.0
            teleop.cmd_vel["angular_z"] = 0.0
            print("Stop")
        
        # WASD controls (alternative to arrow keys)
        elif keycode == glfw.KEY_W:
            teleop.cmd_vel["linear_x"] += teleop.linear_increment
            teleop.cmd_vel["linear_x"] = min(teleop.cmd_vel["linear_x"], teleop.linear_limit)
            print(f"Forward: linear={teleop.cmd_vel['linear_x']:.2f}")
        elif keycode == glfw.KEY_S:
            teleop.cmd_vel["linear_x"] -= teleop.linear_increment
            teleop.cmd_vel["linear_x"] = max(teleop.cmd_vel["linear_x"], -teleop.linear_limit)
            print(f"Backward: linear={teleop.cmd_vel['linear_x']:.2f}")
        elif keycode == glfw.KEY_A:
            teleop.cmd_vel["angular_z"] += teleop.angular_increment
            teleop.cmd_vel["angular_z"] = min(teleop.cmd_vel["angular_z"], teleop.angular_limit)
            print(f"Turn Left: angular={teleop.cmd_vel['angular_z']:.2f}")
        elif keycode == glfw.KEY_D:
            teleop.cmd_vel["angular_z"] -= teleop.angular_increment
            teleop.cmd_vel["angular_z"] = max(teleop.cmd_vel["angular_z"], -teleop.angular_limit)
            print(f"Turn Right: angular={teleop.cmd_vel['angular_z']:.2f}")
    
    # Create viewer with keyboard callback
    print("\nStarting keyboard control...")
    print("Focus the MuJoCo viewer window to control the robot.\n")
    
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    
    # Main control loop
    step_count = 0
    last_print_time = time.time()
    
    try:
        while not should_quit and viewer.is_running():
            # Handle reset
            if should_reset:
                print("\nResetting environment...")
                # Get new start position from maze
                maze_obs, maze_info = maze_env.reset()
                if isinstance(maze_obs, dict):
                    start_pos_2d = np.array(maze_obs.get('observation', maze_obs.get('state', [0, 0]))[:2])
                else:
                    start_pos_2d = np.array(maze_obs[:2])
                
                # Reload model with new start position (use override if set)
                viewer.close()
                if args.spawn_x is not None or args.spawn_y is not None:
                    spawn_pos = [
                        args.spawn_x if args.spawn_x is not None else start_pos_2d[0],
                        args.spawn_y if args.spawn_y is not None else start_pos_2d[1]
                    ]
                else:
                    spawn_pos = None
                model, data, merged_xml_path = load_robot_in_maze(
                    maze_xml_path, robot_model_path, start_pos_2d, spawn_override=spawn_pos
                )
                controller = BicycleController(model, data)
                odometry = Odometry(model, data, robot_body_name="chassis")
                viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
                
                should_reset = False
                step_count = 0
                print(f"Robot reset at: ({start_pos_2d[0]:.2f}, {start_pos_2d[1]:.2f})\n")
                continue
            
            # Get command velocities from teleop
            cmd_vel = teleop.get_cmd_vel()
            
            # Apply control
            controller.apply_cmd_vel(cmd_vel["linear_x"], cmd_vel["angular_z"])
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            step_count += 1
            
            # Print status every 1 second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                odom = odometry.calculate_odom()
                actual_linear = cmd_vel["linear_x"]
                actual_angular = cmd_vel["angular_z"]
                
                print(f"Step {step_count}: "
                      f"Linear={actual_linear:.2f} m/s, "
                      f"Angular={np.degrees(actual_angular):.1f} deg/s, "
                      f"Pos=({odom['position'][0]:.2f}, {odom['position'][1]:.2f})")
                
                last_print_time = current_time
            
            # Small delay for visualization
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        viewer.close()
        maze_env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
