import mujoco
import time
import glfw

class MujocoTeleop:
    def __init__(self, linear_increment=0.1, angular_increment=0.3,
                 linear_limit=1.0, angular_limit=2.0):
        self.cmd_vel = {"linear_x": 0.0, "angular_z": 0.0}
        self.linear_increment = linear_increment
        self.angular_increment = angular_increment
        self.linear_limit = linear_limit
        self.angular_limit = angular_limit

    def key_callback(self, keycode):

        # Forward / backward
        if keycode in (glfw.KEY_KP_8, glfw.KEY_UP):
            self.cmd_vel["linear_x"] += self.linear_increment
        elif keycode in (glfw.KEY_KP_2, glfw.KEY_DOWN):
            self.cmd_vel["linear_x"] -= self.linear_increment

        # Left / right
        elif keycode in (glfw.KEY_KP_4, glfw.KEY_LEFT):
            self.cmd_vel["angular_z"] += self.angular_increment
        elif keycode in (glfw.KEY_KP_6, glfw.KEY_RIGHT):
            self.cmd_vel["angular_z"] -= self.angular_increment

        # Diagonals
        elif keycode == glfw.KEY_KP_7:
            self.cmd_vel["linear_x"] += self.linear_increment
            self.cmd_vel["angular_z"] += self.angular_increment
        elif keycode == glfw.KEY_KP_9:
            self.cmd_vel["linear_x"] += self.linear_increment
            self.cmd_vel["angular_z"] -= self.angular_increment
        elif keycode == glfw.KEY_KP_1:
            self.cmd_vel["linear_x"] -= self.linear_increment
            self.cmd_vel["angular_z"] += self.angular_increment
        elif keycode == glfw.KEY_KP_3:
            self.cmd_vel["linear_x"] -= self.linear_increment
            self.cmd_vel["angular_z"] -= self.angular_increment

        # Stop
        elif keycode == glfw.KEY_KP_5:
            self.cmd_vel["linear_x"] = 0.0
            self.cmd_vel["angular_z"] = 0.0

        # Clip
        self.cmd_vel["linear_x"] = max(min(self.cmd_vel["linear_x"], self.linear_limit), -self.linear_limit)
        self.cmd_vel["angular_z"] = max(min(self.cmd_vel["angular_z"], self.angular_limit), -self.angular_limit)

    def get_cmd_vel(self):
        """Return the current cmd_vel dict (linear_x, angular_z)."""
        return self.cmd_vel
