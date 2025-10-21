import mujoco
import mujoco.viewer
import time
import sys
import numpy as np
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
    from src.core.controller import BicycleController
    from src.core.odometry import Odometry
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
    controller = BicycleController(model, data)
    
    # Initialize odometry tracker
    odometry = Odometry(model, data, robot_body_name="chassis")
    # Prepare lidar sensor and site indices (72 beams)
    beam_count = 72
    rf_sensor_ids = []
    rf_sensor_addrs = []
    rf_site_ids = []
    for i in range(beam_count):
        sname = f"rf_360_s{i:02d}"
        site_name = f"lidar_360_s{i:02d}"
        try:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            rf_sensor_ids.append(sid)
            rf_sensor_addrs.append(model.sensor_adr[sid])
        except Exception:
            rf_sensor_ids.append(-1)
            rf_sensor_addrs.append(-1)
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            rf_site_ids.append(site_id)
        except Exception:
            rf_site_ids.append(-1)

    while viewer.is_running():
        # Ensure built-in sensor visualization is off so only our rays show
        try:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SENSOR] = 0
        except Exception:
            pass
        step_start = time.time()

        # Get teleop command
        cmd = teleop.get_cmd_vel()


        # Send to controller
        controller.apply_cmd_vel(cmd["linear_x"], cmd["angular_z"])

        mujoco.mj_step(model, data)

        # Print first 36 values for brevity
        first_addr = rf_sensor_addrs[0] if rf_sensor_addrs and rf_sensor_addrs[0] >= 0 else 0
        lidar_values = data.sensordata[first_addr:first_addr + 72]
        # Safely read lidar data using individual sensor addresses
        lidar_values = [
            data.sensordata[addr] for addr in rf_sensor_addrs if addr >= 0
        ]
        print("Lidar scan:", np.round(lidar_values, 2))
        
        # Get and print odometry data
        odom_data = odometry.calculate_odom()
        print(f"Odometry - Position: {np.round(odom_data['position'], 3)}, "
              f"Heading: {np.round(odom_data['heading'], 3)}, "
              f"Distance: {np.round(odom_data['distance'], 3)}")

        # Draw lidar rays starting from each site
        viewer.user_scn.ngeom = 0
        ray_width = 0.01
        max_len = 12.0
        ray_rgba = np.array([1.0, 1.0, 0.0, 0.9])  # Yellow rays for hits
        for addr, site_id in zip(rf_sensor_addrs, rf_site_ids):
            if addr < 0 or site_id < 0:
                continue
            start = np.array(data.site_xpos[site_id])
            xmat = np.array(data.site_xmat[site_id]).reshape(3, 3)
            # Use the z-axis of the site's rotation matrix as the ray direction
            direction = xmat[:, 2]
            # Use the -Z axis of the site's rotation matrix as the ray direction
            direction = -xmat[:, 2]
            distance = float(data.sensordata[addr])
            length = float(min(max_len, max(0.0, distance)))
            end = start + direction * length

            scn = viewer.user_scn
            if scn.ngeom < scn.maxgeom:
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, ray_width, start, end)
                geom.rgba[:] = ray_rgba
                scn.ngeom += 1

        # Optional: draw tiny tick marks at starts to verify origin
        debug_rgba = np.array([0.9, 0.1, 0.1, 0.9])
        eps = 0.002
        for site_id in rf_site_ids:
            if site_id < 0:
                continue
            start = np.array(data.site_xpos[site_id])
            end = start + np.array([0, 0, eps])
            scn = viewer.user_scn
            if scn.ngeom < scn.maxgeom:
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 0.006, start, end)
                geom.rgba[:] = debug_rgba
                scn.ngeom += 1

        viewer.sync()
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

