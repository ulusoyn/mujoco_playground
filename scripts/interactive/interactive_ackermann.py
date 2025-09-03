import sys
from pathlib import Path

# Ensure repo root on sys.path for `utils` package import
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import mujoco
import mujoco.viewer
import numpy as np

from utils.ackermann_controller import AckermannController, AckermannParams


def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    xml_path = str(project_root / "models" / "ackermann_in_mushr_maze.xml")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load XML at {xml_path}: {exc}")
        sys.exit(1)

    data = mujoco.MjData(model)

    controller = AckermannController(model, data, AckermannParams())

    # Keyboard rates
    vel_step = 0.05
    steer_step = 0.06

    def key_callback(keycode: int):
        # GLFW: LEFT=263, RIGHT=262, UP=265, DOWN=264, SPACE=32
        if keycode == 265 or keycode in (ord('W'), ord('w')):  # up/W
            controller.set_velocity(controller.velocity_cmd + vel_step)
        elif keycode == 264 or keycode in (ord('S'), ord('s')):  # down/S
            controller.set_velocity(controller.velocity_cmd - vel_step)
        elif keycode == 263 or keycode in (ord('A'), ord('a')):  # left/A -> counterclockwise steer
            controller.set_steering(controller.steering_cmd + steer_step)
        elif keycode == 262 or keycode in (ord('D'), ord('d')):  # right/D -> clockwise steer
            controller.set_steering(controller.steering_cmd - steer_step)
        elif keycode == 32:  # space
            controller.stop()

    print("Interactive Ackermann in maze. Click viewer and use Arrow keys or WASD. Space to stop.")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            controller.step()
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
