import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# --- Resolve model path relative to repo root ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # .../scripts/interactive -> repo root
xml_path = str(project_root / "models" / "ackermann_robot.xml")

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
except Exception as exc:  # noqa: BLE001
    print(f"Failed to load XML at {xml_path}: {exc}")
    sys.exit(1)

data = mujoco.MjData(model)

# Actuator IDs
act_id_steer_left = model.actuator("front_steer_left").id
act_id_steer_right = model.actuator("front_steer_right").id
act_id_left = model.actuator("rear_left_drive").id
act_id_right = model.actuator("rear_right_drive").id

# Rear wheel joint velocity indices (for PI control)
jid_left = model.joint("rear_left_wheel").id
jid_right = model.joint("rear_right_wheel").id
# Use joint's DOF address to index into qvel
qvel_adr_left = model.jnt_dofadr[jid_left]
qvel_adr_right = model.jnt_dofadr[jid_right]

# Robot parameters and control gains
wheel_radius = 0.0325  # meters
# Simple geometry (approx): wheelbase and track
wheelbase_L = 0.20  # distance between front and rear axle (m)
track_W = 0.18      # distance between left and right wheels (m)
dt = model.opt.timestep

# Desired commands (stationary by default)
velocity_cmd = 0.0  # m/s
steering_cmd = 0.0  # rad (commanded bicycle angle)
steering_cmd_prev = 0.0  # rad (for slew limiting)
velocity_cmd_filt = 0.0  # m/s (filtered)
steering_cmd_filt = 0.0  # rad (filtered)

# Limits and steps
max_velocity = 0.4  # m/s
# Steering limits (radians); MuJoCo runtime uses radians regardless of XML angle setting
max_steer = float(np.deg2rad(35.0))  # rad
vel_step = 0.03     # m/s per key press
steer_step = 0.04   # rad per key press

# Simple PI controller for wheel angular velocity -> motor torque
kp_w = 0.25
ki_w = 0.6
integ_left = 0.0
integ_right = 0.0

def apply_controls():
    """Apply Ackermann steering and rear wheel torques via PI velocity control."""
    global integ_left, integ_right, steering_cmd_prev

    # Ackermann steering: compute inner/outer angles from a single steering_cmd (virtual center angle)
    # Interpret steering_cmd as the steering angle of a virtual single track (bicycle model) delta.
    # Slew-limit steering command for smoothness
    max_delta_rate = 2.0  # rad/s
    delta_target_raw = float(np.clip(steering_cmd, -max_steer, max_steer))
    delta_step = np.clip(delta_target_raw - steering_cmd_prev, -max_delta_rate * dt, max_delta_rate * dt)
    delta_slewed = steering_cmd_prev + float(delta_step)
    steering_cmd_prev = delta_slewed

    # Low-pass filter both steering and velocity commands
    global steering_cmd_filt, velocity_cmd_filt
    tau_cmd = 0.08  # s
    alpha = float(np.clip(dt / max(tau_cmd, 1e-6), 0.0, 1.0))
    steering_cmd_filt = (1 - alpha) * steering_cmd_filt + alpha * delta_slewed
    velocity_cmd_filt = (1 - alpha) * velocity_cmd_filt + alpha * float(velocity_cmd)
    delta = steering_cmd_filt
    if abs(delta) < 1e-6:
        delta_left = 0.0
        delta_right = 0.0
    else:
        # Compute turning radius from bicycle model: R = L / tan(delta)
        R = wheelbase_L / np.tan(delta)
        sgn = 1.0 if delta > 0 else -1.0
        # Inner/outer wheel radii (positive magnitudes)
        R_abs = abs(R)
        R_inner = max(R_abs - 0.5 * track_W, 1e-6)
        R_outer = R_abs + 0.5 * track_W
        # Magnitudes of steering angles (positive)
        inner_mag = float(np.arctan(wheelbase_L / R_inner))
        outer_mag = float(np.arctan(wheelbase_L / R_outer))
        if delta > 0:  # turning left: left is inner
            delta_left = sgn * inner_mag
            delta_right = sgn * outer_mag
        else:  # turning right: right is inner
            delta_left = sgn * outer_mag
            delta_right = sgn * inner_mag

    # Apply left/right steering actuators in radians
    # Per-wheel sign to account for joint axis direction in model
    steer_sign_left = 1.0
    steer_sign_right = 1.0
    cmd_left = float(np.clip(steer_sign_left * delta_left, -max_steer, max_steer))
    cmd_right = float(np.clip(steer_sign_right * delta_right, -max_steer, max_steer))
    data.ctrl[act_id_steer_left] = cmd_left
    data.ctrl[act_id_steer_right] = cmd_right

    # Target rear wheel angular velocities from linear velocity and turn geometry
    # Base forward speed is along the vehicle reference path (bicycle model radius R)
    # Outer wheel travels a larger arc; inner wheel travels a smaller arc
    if abs(delta) < 1e-6:
        v_left = float(velocity_cmd_filt)
        v_right = float(velocity_cmd_filt)
    else:
        R = wheelbase_L / np.tan(delta)
        R_abs = abs(R)
        # Avoid division by near-zero radii
        R_inner = max(R_abs - 0.5 * track_W, 1e-3)
        R_outer = R_abs + 0.5 * track_W
        # Speed scale factors for inner/outer relative to bicycle path radius R_abs
        scale_inner = R_inner / R_abs
        scale_outer = R_outer / R_abs
        if delta > 0:  # turning left: left is inner
            v_left = float(velocity_cmd_filt * scale_inner)
            v_right = float(velocity_cmd_filt * scale_outer)
        else:  # turning right: right is inner
            v_left = float(velocity_cmd_filt * scale_outer)
            v_right = float(velocity_cmd_filt * scale_inner)
    target_w_left = v_left / wheel_radius
    target_w_right = v_right / wheel_radius

    # Measure current wheel angular velocities
    w_left = float(data.qvel[qvel_adr_left])
    w_right = float(data.qvel[qvel_adr_right])

    # PI control with small deadband to avoid drift at zero command
    err_l = target_w_left - w_left
    err_r = target_w_right - w_right
    if abs(velocity_cmd_filt) < 1e-3 and abs(err_l) < 0.5 and abs(err_r) < 0.5:
        integ_left = 0.0
        integ_right = 0.0
        tau_left = 0.0
        tau_right = 0.0
    else:
        integ_left = float(np.clip(integ_left + err_l * dt, -2.0, 2.0))
        integ_right = float(np.clip(integ_right + err_r * dt, -2.0, 2.0))
        tau_left = kp_w * err_l + ki_w * integ_left
        tau_right = kp_w * err_r + ki_w * integ_right

    # Clip to actuator force limits (XML currently ±0.3)
    data.ctrl[act_id_left] = float(np.clip(tau_left, -0.3, 0.3))
    data.ctrl[act_id_right] = float(np.clip(tau_right, -0.3, 0.3))


def key_callback(keycode: int):
    """Handle keyboard input from MuJoCo viewer (GLFW key codes)."""
    global velocity_cmd, steering_cmd, integ_left, integ_right

    # GLFW key codes: LEFT=263, RIGHT=262, UP=265, DOWN=264, SPACE=32, R=82
    if keycode in (265,):  # Up
        velocity_cmd = min(velocity_cmd + vel_step, max_velocity)
    elif keycode in (264,):  # Down
        velocity_cmd = max(velocity_cmd - vel_step, -max_velocity)
    elif keycode in (263,):  # Left
        steering_cmd = min(steering_cmd + steer_step, max_steer)
    elif keycode in (262,):  # Right
        steering_cmd = max(steering_cmd - steer_step, -max_steer)
    elif keycode == 32:  # Space: stop
        velocity_cmd = 0.0
        steering_cmd = 0.0
        integ_left = 0.0
        integ_right = 0.0

    # Also support WASD as alternative
    elif keycode in (ord('W'), ord('w')):
        velocity_cmd = min(velocity_cmd + vel_step, max_velocity)
    elif keycode in (ord('S'), ord('s')):
        velocity_cmd = max(velocity_cmd - vel_step, -max_velocity)
    elif keycode in (ord('A'), ord('a')):
        steering_cmd = min(steering_cmd + steer_step, max_steer)
    elif keycode in (ord('D'), ord('d')):
        steering_cmd = max(steering_cmd - steer_step, -max_steer)


# --- Run simulation with viewer (stationary by default) ---
print("Ackermann control: Click viewer window to focus. Use Arrow keys or WASD. Space to stop.")
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        apply_controls()
        mujoco.mj_step(model, data)
        viewer.sync()
