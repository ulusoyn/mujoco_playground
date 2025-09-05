import mujoco
import mujoco.viewer
import time
import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# Fix: use project_root instead of _project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Alternative: Add current directory and src directory to path
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


try:
    from src.teleop.keyboard_teleop import MujocoTeleop
    from src.core.controller import AckermannController
    print("Successfully imported modules!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check your project structure and file paths")
    sys.exit(1)

# Load model
try:
    model_path = project_root / "models" / "ackermann_robot.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

teleop = MujocoTeleop()

# Here we pass teleop.key_callback → Mujoco will call it with keycodes automatically
with mujoco.viewer.launch_passive(model, data, key_callback=teleop.key_callback) as viewer:
    controller = AckermannController(model, data)

    while viewer.is_running():
        step_start = time.time()

        # Get teleop command
        cmd = teleop.get_cmd_vel()
        print(cmd)

        # Send to controller
        controller.apply_cmd_vel(cmd["linear_x"], cmd["angular_z"])

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

