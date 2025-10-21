"""
Odometry utility for tracking robot position and providing realistic odometry data.
This class tracks the robot's position changes from an initial reference point,
providing odometry data suitable for reinforcement learning applications.
"""

import numpy as np
import mujoco


class Odometry:
    """
    Odometry tracker for robot position monitoring.
    
    This class provides realistic odometry data by tracking the robot's position
    changes from an initial reference point. It's designed to be compatible
    with RL environments and real-world robot applications.
    """
    
    def __init__(self, model, data, robot_body_name="chassis"):
        """
        Initialize the odometry tracker.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            robot_body_name: Name of the robot's main body (default: "chassis")
        """
        self.model = model
        self.data = data
        self.robot_body_name = robot_body_name
        
        # Get robot body ID
        try:
            self.robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name)
        except Exception as e:
            raise ValueError(f"Could not find robot body '{robot_body_name}': {e}")
        
        # Initialize reference position (will be set on first call)
        self.reference_position = None
        self.is_initialized = False
        
        # Store initial orientation for heading calculation
        self.reference_quat = None
        
    def initialize(self):
        """
        Initialize the odometry with current robot position as reference.
        This should be called after the robot is placed in its starting position.
        """
        # Get current position and orientation
        current_pos = np.array(self.data.xpos[self.robot_body_id])
        current_quat = np.array(self.data.xquat[self.robot_body_id])
        
        # Set reference position and orientation
        self.reference_position = current_pos.copy()
        self.reference_quat = current_quat.copy()
        self.is_initialized = True
        
        print(f"Odometry initialized with reference position: {self.reference_position}")
        
    def calculate_odom(self):
        """
        Calculate odometry data relative to the reference position.
        
        Returns:
            dict: Dictionary containing odometry data with keys:
                - 'position': 3D position difference (x, y, z) from reference
                - 'orientation': Current quaternion orientation
                - 'heading': Heading angle in radians (yaw)
                - 'distance': Total distance traveled from reference
                - 'is_initialized': Whether odometry has been initialized
        """
        if not self.is_initialized:
            # Auto-initialize if not done manually
            self.initialize()
        
        # Get current position and orientation
        current_pos = np.array(self.data.xpos[self.robot_body_id])
        current_quat = np.array(self.data.xquat[self.robot_body_id])
        
        # Calculate position difference
        position_diff = current_pos - self.reference_position
        
        # Calculate heading (yaw) from quaternion
        # Convert quaternion to euler angles (yaw, pitch, roll)
        heading = self._quat_to_yaw(current_quat)
        
        # Calculate total distance traveled
        distance = np.linalg.norm(position_diff)
        
        # Return odometry data
        odom_data = {
            'position': position_diff,  # 3D position difference (x, y, z)
            'orientation': current_quat,  # Current quaternion
            'heading': heading,  # Heading angle in radians
            'distance': distance,  # Total distance from reference
            'is_initialized': self.is_initialized,
            'reference_position': self.reference_position.copy(),
            'current_position': current_pos.copy()
        }
        
        return odom_data
    
    def reset(self, new_reference_position=None):
        """
        Reset the odometry reference point.
        
        Args:
            new_reference_position: Optional new reference position.
                                   If None, uses current position.
        """
        if new_reference_position is not None:
            self.reference_position = np.array(new_reference_position)
        else:
            # Use current position as new reference
            current_pos = np.array(self.data.xpos[self.robot_body_id])
            self.reference_position = current_pos.copy()
        
        # Update reference orientation
        current_quat = np.array(self.data.xquat[self.robot_body_id])
        self.reference_quat = current_quat.copy()
        
        print(f"Odometry reset with new reference position: {self.reference_position}")
    
    def get_position(self):
        """
        Get current robot position.
        
        Returns:
            np.array: Current 3D position of the robot
        """
        return np.array(self.data.xpos[self.robot_body_id])
    
    def get_orientation(self):
        """
        Get current robot orientation.
        
        Returns:
            np.array: Current quaternion orientation
        """
        return np.array(self.data.xquat[self.robot_body_id])
    
    def get_heading(self):
        """
        Get current robot heading (yaw angle).
        
        Returns:
            float: Heading angle in radians
        """
        current_quat = self.get_orientation()
        return self._quat_to_yaw(current_quat)
    
    def _quat_to_yaw(self, quat):
        """
        Convert quaternion to yaw angle (heading).
        
        Args:
            quat: Quaternion as numpy array [w, x, y, z]
            
        Returns:
            float: Yaw angle in radians
        """
        # MuJoCo quaternions are in [w, x, y, z] format
        w, x, y, z = quat
        
        # Calculate yaw (rotation around Z-axis)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        return yaw
    
    def get_distance_to_point(self, target_point):
        """
        Calculate distance from current position to a target point.
        
        Args:
            target_point: Target 3D position as numpy array or list
            
        Returns:
            float: Distance to target point
        """
        current_pos = self.get_position()
        target_pos = np.array(target_point)
        
        return np.linalg.norm(current_pos - target_pos)
    
    def get_bearing_to_point(self, target_point):
        """
        Calculate bearing (angle) from current position to a target point.
        
        Args:
            target_point: Target 3D position as numpy array or list
            
        Returns:
            float: Bearing angle in radians (0 = forward, positive = right)
        """
        current_pos = self.get_position()
        target_pos = np.array(target_point)
        
        # Calculate vector to target
        to_target = target_pos - current_pos
        
        # Calculate bearing angle (in horizontal plane)
        bearing = np.arctan2(to_target[1], to_target[0])
        
        return bearing
    
    def is_initialized(self):
        """
        Check if odometry has been initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self.is_initialized


# Example usage and testing
if __name__ == "__main__":
    # This would be used in your main simulation
    print("Odometry utility created successfully!")
    print("Usage:")
    print("  odom = Odometry(model, data)")
    print("  odom_data = odom.calculate_odom()")
    print("  print(f'Position: {odom_data[\"position\"]}')")
    print("  print(f'Heading: {odom_data[\"heading\"]}')")
