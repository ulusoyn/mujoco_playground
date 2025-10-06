import mujoco
import numpy as np

# -------------------------
# Shared utility: PID class
# -------------------------
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


# -------------------------
# Ackermann Controller
# -------------------------
class AckermannController:
    def __init__(self, model, data,
                 wheel_radius=0.0325, wheelbase=0.20, track_width=0.174):
        self.model = model
        self.data = data
        self.wheel_radius = wheel_radius
        self.wheelbase = wheelbase
        self.track_width = track_width

        # Actuators
        self.act_steer_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_steer_left")
        self.act_steer_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_steer_right")
        self.act_rear_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_drive")
        self.act_rear_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_drive")

    def cmd_vel_to_controls(self, linear_x, angular_z):
        v = float(linear_x)
        omega = float(angular_z)

        if abs(omega) < 1e-6:
            delta_left = delta_right = 0.0
            v_left = v_right = v
        else:
            R = self.wheelbase / np.tan(np.clip(omega, -1.0, 1.0))
            R_inner = max(abs(R) - self.track_width / 2.0, 1e-6)
            R_outer = abs(R) + self.track_width / 2.0
            inner_angle = np.arctan(self.wheelbase / R_inner)
            outer_angle = np.arctan(self.wheelbase / R_outer)
            if omega > 0:
                delta_left, delta_right = inner_angle, outer_angle
            else:
                delta_left, delta_right = -outer_angle, -inner_angle

            v_left = v * (R_inner / abs(R))
            v_right = v * (R_outer / abs(R))

        w_left = v_left / self.wheel_radius
        w_right = v_right / self.wheel_radius
        return delta_left, delta_right, w_left, w_right

    def apply_cmd_vel(self, linear_x, angular_z):
        delta_l, delta_r, w_l, w_r = self.cmd_vel_to_controls(linear_x, angular_z)
        self.data.ctrl[self.act_steer_left] = np.clip(delta_l, -0.61, 0.61)
        self.data.ctrl[self.act_steer_right] = np.clip(delta_r, -0.61, 0.61)
        self.data.ctrl[self.act_rear_left] = np.clip(w_l, -50, 50)
        self.data.ctrl[self.act_rear_right] = np.clip(w_r, -50, 50)


# -------------------------
# Bicycle Controller
# -------------------------
class BicycleController:
    def __init__(self, model, data,
                 wheel_radius=0.0325, wheelbase=0.20, track_width=0.174):
        self.model = model
        self.data = data
        self.wheel_radius = wheel_radius
        self.wheelbase = wheelbase
        self.track_width = track_width

        self.act_steer = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steering_servo")
        self.act_rear_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_drive")
        self.act_rear_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_drive")

    def cmd_vel_to_controls(self, linear_x, angular_z):
        v = float(linear_x)
        omega = float(angular_z)
        eps = 1e-5

        if abs(omega) < 1e-6:
            delta = 0.0
        else:
            ratio = (self.wheelbase * omega) / (v if abs(v) > eps else np.sign(omega) * eps)
            delta = np.arctan(ratio)

        delta = np.clip(delta, -np.deg2rad(35), np.deg2rad(35))

        if abs(delta) < 1e-6:
            v_left = v_right = v
        else:
            R = self.wheelbase / np.tan(delta)
            R_inner = max(abs(R) - self.track_width / 2.0, 1e-6)
            R_outer = abs(R) + self.track_width / 2.0
            if delta > 0:
                v_left = v * (R_inner / abs(R))
                v_right = v * (R_outer / abs(R))
            else:
                v_left = v * (R_outer / abs(R))
                v_right = v * (R_inner / abs(R))

        w_left = v_left / self.wheel_radius
        w_right = v_right / self.wheel_radius
        return delta, w_left, w_right

    def apply_cmd_vel(self, linear_x, angular_z):
        delta, w_l, w_r = self.cmd_vel_to_controls(linear_x, angular_z)
        self.data.ctrl[self.act_steer] = np.clip(delta, -0.61, 0.61)
        self.data.ctrl[self.act_rear_left] = np.clip(w_l, -50, 50)
        self.data.ctrl[self.act_rear_right] = np.clip(w_r, -50, 50)
