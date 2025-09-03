import sys
from pathlib import Path

import mujoco
import mujoco.viewer as mj_viewer

# Add utils to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "utils"))
from path_helpers import get_model_path


def main():
    # Use utility function to get model path
    xml_path = str(get_model_path("ackermann_robot"))

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load XML at {xml_path}: {exc}")
        sys.exit(1)

    data = mujoco.MjData(model)
    
    # Get actuator ID for the single steering control
    front_steer_id = model.actuator("front_steer_act").id
    
    print(f"Single steering actuator ID: {front_steer_id}")
    print("Both front wheels will now steer together!")

    # Initialize controls to neutral position
    data.ctrl[front_steer_id] = 0.0  # Straight ahead
    
    # Launch interactive viewer; ESC to quit
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Wheels will stay still - no automatic steering input
            # You can manually modify data.ctrl[front_steer_id] to test steering
            
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()


