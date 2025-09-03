import sys
from pathlib import Path

import mujoco
import mujoco.viewer as mj_viewer


def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent  # .../scripts/examples -> repo root
    # Use flattened scene (no external includes) to avoid missing env files
    xml_path = str(project_root / "models" / "ackermann_maze_flat.xml")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load XML at {xml_path}: {exc}")
        sys.exit(1)

    data = mujoco.MjData(model)

    # Optional: spawn the robot at a specific pose to avoid immediate collisions
    data.qpos[0:7] = [-2.9, -3.4, 0.08, 1.0, 0.0, 0.0, 0.0]

    print(f"Loaded scene: {xml_path}")
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()


