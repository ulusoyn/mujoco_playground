import mujoco
import mujoco.viewer
import time
import sys
import numpy as np
from pathlib import Path

# ---- Setup paths ----
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_dir = project_root / "src"

for p in [project_root, src_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---- Imports ----
try:
    from src.teleop.keyboard_teleop import MujocoTeleop
    from src.core.controller import BicycleController
    print("Successfully imported modules!")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ---- Load model ----
try:
    model_path = project_root / "models" / "ackermann_robot.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

teleop = MujocoTeleop()
controller = BicycleController(model, data)

# ---- Lidar setup ----
beam_count = 72
rf_sensor_addrs = []
rf_site_ids = []

for i in range(beam_count):
    sname = f"rf_360_s{i:02d}"
    site_name = f"lidar_360_s{i:02d}"
    try:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
        rf_sensor_addrs.append(model.sensor_adr[sid])
    except Exception:
        rf_sensor_addrs.append(-1)
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        rf_site_ids.append(site_id)
    except Exception:
        rf_site_ids.append(-1)

# ---- Viewer loop ----
with mujoco.viewer.launch_passive(model, data, key_callback=teleop.key_callback) as viewer:
    ray_width = 0.01
    max_len = 12.0
    ray_rgba = np.array([0.1, 0.8, 0.1, 0.9])
    debug_rgba = np.array([0.9, 0.1, 0.1, 0.9])
    eps = 0.002

    while viewer.is_running():
        step_start = time.time()

        if hasattr(viewer, "opt") and hasattr(viewer.opt, "flags"):
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SENSOR] = 0

        # Teleop input
        try:
            cmd = teleop.get_cmd_vel() or {"linear_x": 0.0, "angular_z": 0.0}
        except Exception:
            cmd = {"linear_x": 0.0, "angular_z": 0.0}

        controller.apply_cmd_vel(cmd["linear_x"], cmd["angular_z"])
        mujoco.mj_step(model, data)

        # Get lidar scan safely
        lidar_values = [data.sensordata[addr] for addr in rf_sensor_addrs if addr >= 0]

        # Optional rate-limited print
        if int(data.time * 10) % 10 == 0:
            print("Lidar scan:", np.round(lidar_values[:36], 2))

        # ---- Draw lidar rays ----
        viewer.user_scn.ngeom = 0
        scn = viewer.user_scn
        for addr, site_id in zip(rf_sensor_addrs, rf_site_ids):
            if addr < 0 or site_id < 0:
                continue
            start = np.array(data.site_xpos[site_id])
            xmat = np.array(data.site_xmat[site_id]).reshape(3, 3)
            direction = -xmat[:, 2]
            distance = float(data.sensordata[addr])
            length = min(max_len, max(0.0, distance))
            end = start + direction * length

            if scn.ngeom < scn.maxgeom:
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, ray_width, start, end)
                geom.rgba[:] = ray_rgba
                scn.ngeom += 1

        # Draw origin ticks
        for site_id in rf_site_ids:
            if site_id < 0:
                continue
            start = np.array(data.site_xpos[site_id])
            end = start + np.array([0, 0, eps])
            if scn.ngeom < scn.maxgeom:
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 0.006, start, end)
                geom.rgba[:] = debug_rgba
                scn.ngeom += 1

        viewer.sync()
        # viewer handles timing; sleep is optional
