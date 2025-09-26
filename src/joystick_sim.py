import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


# Ensure repo root and src on sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent


if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


try:
    from core.controller import AckermannController
    from core.cmd_vel_message import cmd_vel_publisher, Twist
    from teleop.joystick_teleop import JoystickTeleop
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def main() -> None:
    # Load model
    model_path = project_root / "models" / "ackermann_robot.xml"
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load XML at {model_path}: {exc}")
        sys.exit(1)

    # Subscribe to cmd_vel messages
    cmd_state = {"vx": 0.0, "wz": 0.0, "t": 0.0}

    def on_cmd_vel(msg: Twist) -> None:
        cmd_state["vx"] = float(msg.linear.x)
        cmd_state["wz"] = float(msg.angular.z)
        cmd_state["t"] = time.time()
        # Debug output - show ALL messages
        print(f"Received cmd_vel: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")

    cmd_vel_publisher.subscribe(on_cmd_vel)
    
    # Test the cmd_vel system
    print("Testing cmd_vel system...")
    test_msg = Twist(linear_x=0.5, angular_z=0.3)
    cmd_vel_publisher.publish(test_msg)
    print("Test message sent")

    # Start joystick teleop in a background thread
    teleop = JoystickTeleop(max_linear_vel=1.5, max_angular_vel=3.0, deadzone=0.12)

    import threading

    teleop_thread = threading.Thread(target=teleop.start, daemon=True)
    teleop_thread.start()
    
    print("Joystick teleop thread started")
    print("Waiting for joystick input...")

    # Launch viewer and control loop
    timeout_s = 0.5  # if no cmd for this long, command zero
    with mujoco.viewer.launch_passive(model, data) as viewer:
        controller = AckermannController(model, data)
        while viewer.is_running():
            step_start = time.time()

            # Timeout to stop robot if stale
            if step_start - cmd_state["t"] > timeout_s:
                vx = 0.0
                wz = 0.0
                print(f"TIMEOUT: No cmd_vel for {step_start - cmd_state['t']:.1f}s, stopping robot")
            else:
                vx = cmd_state["vx"]
                wz = cmd_state["wz"]
                # Debug: show commands being sent to controller
                if abs(vx) > 0.01 or abs(wz) > 0.01:
                    print(f"Sending to controller: vx={vx:.3f}, wz={wz:.3f}")

            controller.apply_cmd_vel(vx, wz)
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            time.sleep(max(0.0, model.opt.timestep - (time.time() - step_start)))

    # Ensure teleop stops
    try:
        teleop.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()


