"""
Wheel velocity and steering angle monitoring utility.
This script helps diagnose issues with Ackermann steering and wheel velocities.
"""

import mujoco
import numpy as np
import time
from pathlib import Path
import sys

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.controller import BicycleController
from src.teleop.keyboard_teleop import MujocoTeleop


class WheelMonitor:
    """Monitor wheel velocities and steering angles for diagnostics."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Get joint IDs
        self.joint_ids = {
            'rear_left_wheel': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rear_left_wheel"),
            'rear_right_wheel': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rear_right_wheel"),
            'front_left_steer': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "front_left_steer"),
            'front_right_steer': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "front_right_steer"),
            'front_left_wheel': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "front_left_wheel"),
            'front_right_wheel': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "front_right_wheel"),
        }
        
        # Get actuator IDs
        self.actuator_ids = {
            'front_steer_left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_steer_left"),
            'front_steer_right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_steer_right"),
            'rear_left_drive': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_drive"),
            'rear_right_drive': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_drive"),
        }
        
        # Robot parameters
        self.wheel_radius = 0.0325
        self.wheelbase = 0.20
        self.track_width = 0.174
        
    def get_wheel_data(self):
        """Get current wheel velocities and steering angles."""
        data = {}
        
        # Get joint velocities (angular velocities)
        data['rear_left_vel'] = self.data.qvel[self.joint_ids['rear_left_wheel']]
        data['rear_right_vel'] = self.data.qvel[self.joint_ids['rear_right_wheel']]
        data['front_left_vel'] = self.data.qvel[self.joint_ids['front_left_wheel']]
        data['front_right_vel'] = self.data.qvel[self.joint_ids['front_right_wheel']]
        
        # Get steering angles
        data['front_left_steer'] = self.data.qpos[self.joint_ids['front_left_steer']]
        data['front_right_steer'] = self.data.qpos[self.joint_ids['front_right_steer']]
        
        # Get actuator control values
        data['steering_ctrl_left'] = self.data.ctrl[self.actuator_ids['front_steer_left']]
        data['steering_ctrl_right'] = self.data.ctrl[self.actuator_ids['front_steer_right']]
        data['rear_left_ctrl'] = self.data.ctrl[self.actuator_ids['rear_left_drive']]
        data['rear_right_ctrl'] = self.data.ctrl[self.actuator_ids['rear_right_drive']]
        
        # Calculate linear velocities
        data['rear_left_linear'] = data['rear_left_vel'] * self.wheel_radius
        data['rear_right_linear'] = data['rear_right_vel'] * self.wheel_radius
        data['front_left_linear'] = data['front_left_vel'] * self.wheel_radius
        data['front_right_linear'] = data['front_right_vel'] * self.wheel_radius
        
        return data
    
    def print_wheel_data(self, wheel_data):
        """Print formatted wheel data."""
        print("\n" + "="*60)
        print("WHEEL DIAGNOSTICS")
        print("="*60)
        
        print(f"STEERING ANGLES:")
        print(f"  Front Left:  {np.degrees(wheel_data['front_left_steer']):6.2f}° (ctrl: {wheel_data['steering_ctrl_left']:6.3f})")
        print(f"  Front Right: {np.degrees(wheel_data['front_right_steer']):6.2f}° (ctrl: {wheel_data['steering_ctrl_right']:6.3f})")
        
        print(f"\nANGULAR VELOCITIES:")
        print(f"  Rear Left:   {wheel_data['rear_left_vel']:6.2f} rad/s (ctrl: {wheel_data['rear_left_ctrl']:6.2f})")
        print(f"  Rear Right:  {wheel_data['rear_right_vel']:6.2f} rad/s (ctrl: {wheel_data['rear_right_ctrl']:6.2f})")
        print(f"  Front Left:  {wheel_data['front_left_vel']:6.2f} rad/s")
        print(f"  Front Right: {wheel_data['front_right_vel']:6.2f} rad/s")
        
        print(f"\nLINEAR VELOCITIES:")
        print(f"  Rear Left:   {wheel_data['rear_left_linear']:6.2f} m/s")
        print(f"  Rear Right:  {wheel_data['rear_right_linear']:6.2f} m/s")
        print(f"  Front Left:  {wheel_data['front_left_linear']:6.2f} m/s")
        print(f"  Front Right: {wheel_data['front_right_linear']:6.2f} m/s")
        
        # Check for issues
        self.check_issues(wheel_data)
    
    def check_issues(self, wheel_data):
        """Check for common issues."""
        print(f"\nISSUE DETECTION:")
        
        # Check steering angle difference
        steer_diff = abs(wheel_data['front_left_steer'] - wheel_data['front_right_steer'])
        if steer_diff > 0.01:  # More than ~0.6 degrees
            print(f"  ⚠️  Steering angle mismatch: {np.degrees(steer_diff):.2f}°")
        else:
            print(f"  ✅ Steering angles synchronized")
        
        # Check wheel velocity differences
        rear_vel_diff = abs(wheel_data['rear_left_vel'] - wheel_data['rear_right_vel'])
        if rear_vel_diff > 0.1:  # More than 0.1 rad/s difference
            print(f"  ⚠️  Rear wheel velocity mismatch: {rear_vel_diff:.2f} rad/s")
        else:
            print(f"  ✅ Rear wheel velocities synchronized")
        
        # Check for excessive velocities
        max_vel = max(abs(wheel_data['rear_left_vel']), abs(wheel_data['rear_right_vel']))
        if max_vel > 100:  # More than 100 rad/s
            print(f"  ⚠️  Excessive wheel velocity: {max_vel:.2f} rad/s")
        else:
            print(f"  ✅ Wheel velocities within normal range")
        
        # Check Ackermann geometry
        if abs(wheel_data['front_left_steer']) > 0.01:
            self.check_ackermann_geometry(wheel_data)
    
    def check_ackermann_geometry(self, wheel_data):
        """Check if Ackermann geometry is correct."""
        delta_left = wheel_data['front_left_steer']
        delta_right = wheel_data['front_right_steer']
        
        # Calculate expected Ackermann angles
        if abs(delta_left) > 0.01:
            R = self.wheelbase / np.tan(delta_left)
            R_inner = abs(R) - self.track_width / 2.0
            R_outer = abs(R) + self.track_width / 2.0
            
            if R > 0:
                expected_left = np.arctan(self.wheelbase / R_inner)
                expected_right = np.arctan(self.wheelbase / R_outer)
            else:
                expected_left = -np.arctan(self.wheelbase / R_outer)
                expected_right = -np.arctan(self.wheelbase / R_inner)
            
            print(f"\nACKERMANN GEOMETRY CHECK:")
            print(f"  Current Left:  {np.degrees(delta_left):6.2f}°")
            print(f"  Expected Left: {np.degrees(expected_left):6.2f}°")
            print(f"  Current Right: {np.degrees(delta_right):6.2f}°")
            print(f"  Expected Right:{np.degrees(expected_right):6.2f}°")
            
            left_error = abs(delta_left - expected_left)
            right_error = abs(delta_right - expected_right)
            
            if left_error > 0.05 or right_error > 0.05:  # More than ~3 degrees
                print(f"  ⚠️  Ackermann geometry incorrect!")
            else:
                print(f"  ✅ Ackermann geometry correct")


def main():
    """Run wheel monitoring diagnostics."""
    # Load model
    model_path = project_root / "models" / "ackermann_robot.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Initialize components
    teleop = MujocoTeleop()
    controller = BicycleController(model, data)
    monitor = WheelMonitor(model, data)
    
    print("Wheel Monitor Started!")
    print("Use WASD keys to control the robot")
    print("Press Ctrl+C to exit")
    
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=teleop.key_callback) as viewer:
            step_count = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # Get teleop command
                cmd = teleop.get_cmd_vel()
                
                # Send to controller
                controller.apply_cmd_vel(cmd["linear_x"], cmd["angular_z"])
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Print diagnostics every 50 steps (about 1 second)
                if step_count % 50 == 0:
                    wheel_data = monitor.get_wheel_data()
                    monitor.print_wheel_data(wheel_data)
                
                step_count += 1
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
                
    except KeyboardInterrupt:
        print("\nWheel Monitor Stopped!")


if __name__ == "__main__":
    main()
