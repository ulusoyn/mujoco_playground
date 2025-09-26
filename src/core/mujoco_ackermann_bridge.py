"""
MuJoCo bridge that connects cmd_vel controller to the Ackermann robot simulation
"""
import sys
from pathlib import Path
import numpy as np
import time

# Add project root to path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import mujoco
from core.cmd_vel_message import cmd_vel_publisher
from ackermann_cmd_vel_controller import AckermannCmdVelController, AckermannParams


class MuJoCoAckermannBridge:
    """
    Bridge between cmd_vel controller and MuJoCo Ackermann robot simulation
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Initialize cmd_vel controller
        self.controller = AckermannCmdVelController(AckermannParams())
        
        # Get actuator IDs
        self.actuator_ids = self._get_actuator_ids()
        
        # Control gains (can be tuned)
        self.wheel_torque_gain = 0.5  # Convert rad/s to torque
        self.steer_position_gain = 1.0  # Steering position control gain
        
        # State tracking
        self.last_status_print = 0.0
        self.status_print_interval = 1.0  # Print status every N seconds
        
        print("MuJoCo Ackermann Bridge initialized")
        print(f"Found actuators: {list(self.actuator_ids.keys())}")
    
    def _get_actuator_ids(self) -> dict:
        """Get MuJoCo actuator IDs"""
        actuators = {}
        
        # Required actuators from the XML
        required_actuators = [
            'rear_left_drive',
            'rear_right_drive', 
            'front_steer_left',
            'front_steer_right'
        ]
        
        for name in required_actuators:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id == -1:
                raise ValueError(f"Actuator '{name}' not found in model")
            actuators[name] = actuator_id
        
        return actuators
    
    def step(self):
        """
        Update the robot control based on current cmd_vel commands
        Call this every simulation step
        """
        # Get wheel commands from controller
        left_vel, right_vel, left_steer, right_steer = self.controller.get_wheel_commands()
        
        # Apply commands to MuJoCo actuators
        self._apply_wheel_velocities(left_vel, right_vel)
        self._apply_steering_angles(left_steer, right_steer)
        
        # Print status periodically
        current_time = time.time()
        if current_time - self.last_status_print > self.status_print_interval:
            self._print_status()
            self.last_status_print = current_time
    
    def _apply_wheel_velocities(self, left_vel: float, right_vel: float):
        """Apply wheel velocities to rear drive motors"""
        # Convert desired angular velocity to torque commands
        # This is a simple P-controller approach
        
        # Get current wheel velocities from sensors
        left_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'rear_left_wheel')
        right_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'rear_right_wheel')
        
        if left_joint_id != -1 and right_joint_id != -1:
            current_left_vel = self.data.qvel[self.model.jnt_dofadr[left_joint_id]]
            current_right_vel = self.data.qvel[self.model.jnt_dofadr[right_joint_id]]
            
            # Simple velocity control using torque
            left_error = left_vel - current_left_vel
            right_error = right_vel - current_right_vel
            
            left_torque = self.wheel_torque_gain * left_error
            right_torque = self.wheel_torque_gain * right_error
            
            # Clamp torques to actuator limits
            torque_limit = 0.3  # From XML forcerange
            left_torque = np.clip(left_torque, -torque_limit, torque_limit)
            right_torque = np.clip(right_torque, -torque_limit, torque_limit)
        else:
            # Fallback: direct torque from velocity command
            left_torque = np.clip(left_vel * 0.1, -0.3, 0.3)
            right_torque = np.clip(right_vel * 0.1, -0.3, 0.3)
        
        # Apply torques
        self.data.ctrl[self.actuator_ids['rear_left_drive']] = left_torque
        self.data.ctrl[self.actuator_ids['rear_right_drive']] = right_torque
    
    def _apply_steering_angles(self, left_steer: float, right_steer: float):
        """Apply steering angles to front wheel position actuators"""
        # The XML uses position actuators for steering
        self.data.ctrl[self.actuator_ids['front_steer_left']] = left_steer
        self.data.ctrl[self.actuator_ids['front_steer_right']] = right_steer
    
    def _print_status(self):
        """Print current status for debugging"""
        if self.controller.is_cmd_valid():
            print(f"[Bridge] {self.controller.get_status_string()}")
        else:
            print("[Bridge] No valid cmd_vel (timeout)")
    
    def get_robot_state(self) -> dict:
        """Get current robot state"""
        # Get chassis position and orientation
        chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        
        if chassis_id != -1:
            pos = self.data.xpos[chassis_id].copy()
            quat = self.data.xquat[chassis_id].copy()
            
            # Convert quaternion to yaw angle
            yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                            1 - 2 * (quat[2]**2 + quat[3]**2))
        else:
            pos = np.zeros(3)
            yaw = 0.0
        
        # Get wheel states
        state = {
            'position': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
            'yaw': yaw,
            'wheel_commands': self.controller.get_wheel_commands(),
            'cmd_valid': self.controller.is_cmd_valid(),
            'linear_cmd': self.controller.current_linear_vel,
            'angular_cmd': self.controller.current_angular_vel
        }
        
        return state


def set_spawn_pose(model: mujoco.MjModel, data: mujoco.MjData, 
                  xy=(0.0, 0.0), z=0.09, yaw_rad=0.0) -> None:
    """Spawn chassis freejoint at specified pose"""
    x, y = float(xy[0]), float(xy[1])
    qw = float(np.cos(0.5 * yaw_rad))
    qz = float(np.sin(0.5 * yaw_rad))
    data.qpos[0:7] = [x, y, float(z), qw, 0.0, 0.0, qz]
    mujoco.mj_forward(model, data)


def main():
    """Test the bridge with keyboard or joystick input"""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    xml_path = str(project_root / "models" / "ackermann_robot.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as exc:
        print(f"Failed to load XML at {xml_path}: {exc}")
        sys.exit(1)
    
    data = mujoco.MjData(model)
    
    # Initialize robot position
    set_spawn_pose(model, data, xy=(0.0, 0.0), z=0.09, yaw_rad=0.0)
    
    # Create bridge
    bridge = MuJoCoAckermannBridge(model, data)
    
    # Start teleop in separate process/thread (choose one)
    import subprocess
    import threading
    
    def run_keyboard_teleop():
        """Run keyboard teleop in thread"""
        try:
            from keyboard_teleop import KeyboardTeleop
            teleop = KeyboardTeleop(max_linear_vel=1.5, max_angular_vel=3.0)
            teleop.start()
        except ImportError:
            print("Could not import keyboard teleop")
    
    # Start keyboard teleop in background thread
    teleop_thread = threading.Thread(target=run_keyboard_teleop, daemon=True)
    teleop_thread.start()
    
    print("\nBridge test running. Use keyboard controls:")
    print("W/S: Forward/Backward, A/D: Left/Right, Space: Stop, Q: Quit")
    print("Starting simulation loop...")
    
    # Simple simulation loop for testing
    dt = model.opt.timestep
    steps = 0
    
    try:
        while True:
            # Update bridge (applies cmd_vel to robot)
            bridge.step()
            
            # Step physics
            mujoco.mj_step(model, data)
            steps += 1
            
            # Print robot state occasionally
            if steps % 1000 == 0:
                state = bridge.get_robot_state()
                pos = state['position']
                print(f"Robot at ({pos['x']:.2f}, {pos['y']:.2f}), "
                      f"yaw={np.degrees(state['yaw']):.1f}°")
            
            # Simple real-time simulation
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nBridge test stopped.")
    
    # Send stop command
    from core.cmd_vel_message import Twist
    cmd_vel_publisher.publish(Twist())


if __name__ == "__main__":
    main()