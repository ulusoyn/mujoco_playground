import mujoco
import numpy as np

class PID:
    def __init__(self, kp, ki, kd, dt=0.002):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class AckermannController:
    def __init__(self, model, data,
                 wheel_radius=0.065, wheelbase=0.20, track_width=0.174):
        """
        model, data: MuJoCo model and data objects
        wheel_radius: wheel radius in meters
        wheelbase: distance between front and rear axles (m)
        track_width: distance between left and right wheels (m)
        """
        self.model = model
        self.data = data
        self.wheel_radius = wheel_radius
        self.wheelbase = wheelbase
        self.track_width = track_width

        # Actuator IDs
        self.act_steer_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_steer_left")
        self.act_steer_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_steer_right")
        self.act_rear_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_drive")
        self.act_rear_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_drive")

        # Joint IDs for wheel velocity feedback
        self.joint_rear_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rear_left_wheel")
        self.joint_rear_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rear_right_wheel")

        # PID controllers for wheel speeds
        dt = model.opt.timestep
        self.pid_left = PID(kp=2.0, ki=0.1, kd=0.05, dt=dt)
        self.pid_right = PID(kp=2.0, ki=0.1, kd=0.05, dt=dt)

    def cmd_vel_to_controls(self, linear_x, angular_z):
        if abs(linear_x) < 1e-5 and abs(angular_z) < 1e-5:
            return 0.0, 0.0, 0.0, 0.0

        if abs(angular_z) < 1e-4:
            delta_left = delta_right = 0.0
            v_left = v_right = linear_x
        else:
            R = linear_x / angular_z  # turning radius (signed)

            # steering angles
            delta_left = np.arctan2(self.wheelbase, R - (self.track_width / 2))
            delta_right = np.arctan2(self.wheelbase, R + (self.track_width / 2))

            # rear wheel linear velocities
            v_left = angular_z * (R - self.track_width / 2)
            v_right = angular_z * (R + self.track_width / 2)

        if angular_z < -1e-4 and abs(angular_z) > 1e-4:
            v_left = -v_left
            v_right = -v_right

        # Convert to wheel angular velocities
        w_left = v_left / self.wheel_radius
        w_right = v_right / self.wheel_radius

        # Clip steering angles to joint limits
        max_steer = np.deg2rad(35)
        delta_left = np.clip(delta_left, -max_steer, max_steer)
        delta_right = np.clip(delta_right, -max_steer, max_steer)

        return delta_left, delta_right, w_left, w_right



    def apply_cmd_vel(self, linear_x, angular_z):
        delta_left, delta_right, w_left, w_right = self.cmd_vel_to_controls(linear_x, angular_z)

        # Steering (position actuators expect radians)
        self.data.ctrl[self.act_steer_left] = delta_left
        self.data.ctrl[self.act_steer_right] = delta_right

        # Send wheel angular velocities directly to velocity actuators
        self.data.ctrl[self.act_rear_left] = w_left
        self.data.ctrl[self.act_rear_right] = w_right

