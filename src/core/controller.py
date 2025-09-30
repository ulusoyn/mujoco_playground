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
                 wheel_radius=0.0325, wheelbase=0.20, track_width=0.174):
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
        v_cmd = float(linear_x)
        yaw_rate = float(angular_z)

        # If no yaw, go straight
        if abs(yaw_rate) < 1e-6:
            delta_left = 0.0
            delta_right = 0.0
            v_left = v_cmd
            v_right = v_cmd
        else:
            # Bicycle steering from v and yaw_rate: tan(delta) = wb * yaw_rate / v
            # Avoid division blowups at low speeds
            eps_v = 1e-5
            ratio = (self.wheelbase * yaw_rate) / (v_cmd if abs(v_cmd) > eps_v else np.sign(yaw_rate) * eps_v)
            delta_bicycle = float(np.arctan(ratio))

            # Clamp to actuator ctrlrange (±0.5 rad) for stability
            delta_max = 0.5
            delta_bicycle = float(np.clip(delta_bicycle, -delta_max, delta_max))

            # Derive Ackermann inner/outer angles from clamped bicycle delta
            R = self.wheelbase / np.tan(delta_bicycle)
            R_abs = abs(R)
            inner_R = max(R_abs - 0.5 * self.track_width, 1e-6)
            outer_R = R_abs + 0.5 * self.track_width
            inner_mag = float(np.arctan(self.wheelbase / inner_R))
            outer_mag = float(np.arctan(self.wheelbase / outer_R))
            if yaw_rate >= 0.0:  # left turn
                delta_left = +inner_mag
                delta_right = +outer_mag
            else:  # right turn
                delta_left = -outer_mag
                delta_right = -inner_mag

            # Wheel linear speeds scaled from commanded linear speed
            scale_inner = inner_R / R_abs
            scale_outer = outer_R / R_abs
            if yaw_rate >= 0.0:  # left turn
                v_left = v_cmd * scale_inner
                v_right = v_cmd * scale_outer
            else:  # right turn
                v_left = v_cmd * scale_outer
                v_right = v_cmd * scale_inner

        # Convert linear wheel speeds to angular wheel speeds
        w_left = v_left / self.wheel_radius
        w_right = v_right / self.wheel_radius

        # Clip steering to physical joint limits
        max_steer = float(np.deg2rad(35.0))
        delta_left = float(np.clip(delta_left, -max_steer, max_steer))
        delta_right = float(np.clip(delta_right, -max_steer, max_steer))

        return float(delta_left), float(delta_right), float(w_left), float(w_right)

    def apply_cmd_vel(self, linear_x, angular_z):
        delta_left, delta_right, w_left, w_right = self.cmd_vel_to_controls(linear_x, angular_z)

        # Clamp to actuator ctrlranges for safety
        try:
            steer_min_l, steer_max_l = self.model.actuator_ctrlrange[self.act_steer_left]
            steer_min_r, steer_max_r = self.model.actuator_ctrlrange[self.act_steer_right]
            vel_min_l, vel_max_l = self.model.actuator_ctrlrange[self.act_rear_left]
            vel_min_r, vel_max_r = self.model.actuator_ctrlrange[self.act_rear_right]
        except Exception:
            steer_min_l = steer_min_r = -0.5
            steer_max_l = steer_max_r = 0.5
            vel_min_l = vel_min_r = -50.0
            vel_max_l = vel_max_r = 50.0

        self.data.ctrl[self.act_steer_left] = float(np.clip(delta_left, steer_min_l, steer_max_l))
        self.data.ctrl[self.act_steer_right] = float(np.clip(delta_right, steer_min_r, steer_max_r))
        self.data.ctrl[self.act_rear_left] = float(np.clip(w_left, vel_min_l, vel_max_l))
        self.data.ctrl[self.act_rear_right] = float(np.clip(w_right, vel_min_r, vel_max_r))

