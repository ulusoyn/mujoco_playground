"""
Simple cmd_vel message structure similar to ROS2 geometry_msgs/Twist
"""
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class Vector3:
    """3D vector representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Twist:
    """Twist message containing linear and angular velocity"""
    linear: Vector3
    angular: Vector3
    
    def __init__(self, linear_x: float = 0.0, linear_y: float = 0.0, linear_z: float = 0.0,
                 angular_x: float = 0.0, angular_y: float = 0.0, angular_z: float = 0.0):
        self.linear = Vector3(linear_x, linear_y, linear_z)
        self.angular = Vector3(angular_x, angular_y, angular_z)


class CmdVelPublisher:
    """Simple publisher for cmd_vel messages"""
    
    def __init__(self):
        self._subscribers = []
        self._last_message: Optional[Twist] = None
        self._last_publish_time: float = 0.0
    
    def subscribe(self, callback):
        """Subscribe to cmd_vel messages"""
        self._subscribers.append(callback)
    
    def publish(self, twist_msg: Twist):
        """Publish a cmd_vel message to all subscribers"""
        self._last_message = twist_msg
        self._last_publish_time = time.time()
        
        for callback in self._subscribers:
            try:
                callback(twist_msg)
            except Exception as e:
                print(f"Error in subscriber callback: {e}")
    
    def get_last_message(self) -> Optional[Twist]:
        """Get the last published message"""
        return self._last_message
    
    def get_last_publish_time(self) -> float:
        """Get timestamp of last published message"""
        return self._last_publish_time


# Global cmd_vel publisher instance
cmd_vel_publisher = CmdVelPublisher()