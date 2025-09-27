import math
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np


@dataclass
class AckermannParams:
    wheel_radius_m: float = 0.0325
    wheelbase_m: float = 0.20
    track_width_m: float = 0.18
    max_velocity_mps: float = 0.8
    max_steer_rad: float = math.radians(35.0)
    max_delta_rate_radps: float = 3.0
    torque_limit: float = 1.5
    kp_w: float = 0.6
    ki_w: float = 1.2
    cmd_filter_tau_s: float = 0.06


class AckermannController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, params: Optional[AckermannParams] = None):
        self.model = model
        self.data = data
        self.params = params or AckermannParams()

        # Actuators
        self.act_id_steer_left = model.actuator("front_steer_left").id
        self.act_id_steer_right = model.actuator("front_steer_right").id
        self.act_id_left = model.actuator("rear_left_drive").id
        self.act_id_right = model.actuator("rear_right_drive").id

        # Wheel velocity indices
        jid_left = model.joint("rear_left_wheel").id
        jid_right = model.joint("rear_right_wheel").id
        self.qvel_adr_left = int(model.jnt_dofadr[jid_left])
        self.qvel_adr_right = int(model.jnt_dofadr[jid_right])

        # Commands and state
        self.velocity_cmd = 0.0
        self.steering_cmd = 0.0
        self._steering_cmd_prev = 0.0
        self._velocity_cmd_filt = 0.0
        self._steering_cmd_filt = 0.0
        self._integ_left = 0.0
        self._integ_right = 0.0

    def set_velocity(self, v_mps: float) -> None:
        self.velocity_cmd = float(np.clip(v_mps, -self.params.max_velocity_mps, self.params.max_velocity_mps))

    def set_steering(self, delta_rad: float) -> None:
        self.steering_cmd = float(np.clip(delta_rad, -self.params.max_steer_rad, self.params.max_steer_rad))

    def stop(self) -> None:
        self.velocity_cmd = 0.0
        self.steering_cmd = 0.0
        self._integ_left = 0.0
        self._integ_right = 0.0

    def step(self) -> None:
        dt = float(self.model.opt.timestep)
        p = self.params

        # Slew limit on steering
        delta_target = float(np.clip(self.steering_cmd, -p.max_steer_rad, p.max_steer_rad))
        delta_step = float(np.clip(delta_target - self._steering_cmd_prev, -p.max_delta_rate_radps * dt, p.max_delta_rate_radps * dt))
        self._steering_cmd_prev = self._steering_cmd_prev + delta_step

        # Low-pass filter commands
        alpha = float(np.clip(dt / max(p.cmd_filter_tau_s, 1e-6), 0.0, 1.0))
        self._steering_cmd_filt = (1 - alpha) * self._steering_cmd_filt + alpha * self._steering_cmd_prev
        self._velocity_cmd_filt = (1 - alpha) * self._velocity_cmd_filt + alpha * float(self.velocity_cmd)

        # Ackermann geometry for left/right steering commands
        delta = self._steering_cmd_filt
        if abs(delta) < 1e-6:
            delta_left = 0.0
            delta_right = 0.0
        else:
            R = p.wheelbase_m / math.tan(delta)
            R_abs = abs(R)
            R_inner = max(R_abs - 0.5 * p.track_width_m, 1e-6)
            R_outer = R_abs + 0.5 * p.track_width_m
            inner_mag = float(math.atan(p.wheelbase_m / R_inner))
            outer_mag = float(math.atan(p.wheelbase_m / R_outer))
            if delta > 0.0:  # left turn: left is inner
                delta_left = inner_mag
                delta_right = outer_mag
            else:  # right turn: right is inner
                delta_left = -outer_mag
                delta_right = -inner_mag

        # Apply front steering (radians)
        self.data.ctrl[self.act_id_steer_left] = float(np.clip(delta_left, -p.max_steer_rad, p.max_steer_rad))
        self.data.ctrl[self.act_id_steer_right] = float(np.clip(delta_right, -p.max_steer_rad, p.max_steer_rad))

        # Wheel speed targets based on turn geometry
        if abs(delta) < 1e-6:
            v_left = float(self._velocity_cmd_filt)
            v_right = float(self._velocity_cmd_filt)
        else:
            R = p.wheelbase_m / math.tan(delta)
            R_abs = abs(R)
            R_inner = max(R_abs - 0.5 * p.track_width_m, 1e-3)
            R_outer = R_abs + 0.5 * p.track_width_m
            scale_inner = R_inner / R_abs
            scale_outer = R_outer / R_abs
            if delta > 0.0:  # left turn
                v_left = float(self._velocity_cmd_filt * scale_inner)
                v_right = float(self._velocity_cmd_filt * scale_outer)
            else:  # right turn
                v_left = float(self._velocity_cmd_filt * scale_outer)
                v_right = float(self._velocity_cmd_filt * scale_inner)
        w_left_target = v_left / p.wheel_radius_m
        w_right_target = v_right / p.wheel_radius_m

        # Command wheel angular velocities directly to velocity actuators
        # (these actuators interpret ctrl as desired qvel)
        try:
            ctrl_min_l = float(self.model.actuator_ctrlrange[self.act_id_left][0])
            ctrl_max_l = float(self.model.actuator_ctrlrange[self.act_id_left][1])
            ctrl_min_r = float(self.model.actuator_ctrlrange[self.act_id_right][0])
            ctrl_max_r = float(self.model.actuator_ctrlrange[self.act_id_right][1])
        except Exception:
            ctrl_min_l = -50.0
            ctrl_max_l = 50.0
            ctrl_min_r = -50.0
            ctrl_max_r = 50.0

        self.data.ctrl[self.act_id_left] = float(np.clip(w_left_target, ctrl_min_l, ctrl_max_l))
        self.data.ctrl[self.act_id_right] = float(np.clip(w_right_target, ctrl_min_r, ctrl_max_r))
