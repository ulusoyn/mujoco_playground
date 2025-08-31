import sys
from pathlib import Path

import mujoco
import mujoco.viewer as mj_viewer


def main():
    repo_root = Path(__file__).resolve().parent
    xml_path = str(repo_root / "ackermann_robot.xml")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load XML at {xml_path}: {exc}")
        sys.exit(1)

    data = mujoco.MjData(model)

    # Launch interactive viewer; ESC to quit
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()


