import sys
from pathlib import Path

# Ensure repo root on sys.path for `utils` package import
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np

from utils.ackermann_controller import AckermannController, AckermannParams


def set_spawn_pose(model: mujoco.MjModel, data: mujoco.MjData, xy=( -2.9, -3.3 ), z=0.09, yaw_rad=0.0) -> None:
    """Spawn chassis freejoint at a known empty cell with desired yaw.
    qpos layout for a freejoint: x, y, z, qw, qx, qy, qz
    """
    x, y = float(xy[0]), float(xy[1])
    qw = float(np.cos(0.5 * yaw_rad))
    qz = float(np.sin(0.5 * yaw_rad))
    data.qpos[0:7] = [x, y, float(z), qw, 0.0, 0.0, qz]
    mujoco.mj_forward(model, data)


def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent  # .../scripts/examples -> repo root
    # Use flattened maze scene (self-contained)
    xml_path = str(project_root / "models" / "ackermann_robot.xml")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load XML at {xml_path}: {exc}")
        sys.exit(1)

    data = mujoco.MjData(model)

    # Place robot at world origin, facing +X (yaw=0)
    set_spawn_pose(model, data, xy=(0.0, 0.0), z=0.09, yaw_rad=0.0)

    controller = AckermannController(model, data, AckermannParams())

    # Keyboard increments
    vel_step = 0.05
    steer_step = 0.06

    # Build rangefinder (sensor, site) lookup for visualization
    rf_count = 36
    rf_sensor_addrs = []  # index of each sensor in sensordata
    rf_site_ids = []
    for i in range(rf_count):
        sensor_name = f"rf_360_s{i:02d}"
        site_name = f"lidar_360_s{i:02d}"
        s_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if s_id == -1:
            continue
        rf_sensor_addrs.append(model.sensor_adr[s_id])
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        rf_site_ids.append(site_id)

    show_lidar = True

    def key_callback(keycode: int):
        nonlocal show_lidar
        # GLFW: LEFT=263, RIGHT=262, UP=265, DOWN=264, SPACE=32
        if keycode == 265 or keycode in (ord('W'), ord('w')):
            controller.set_velocity(controller.velocity_cmd + vel_step)
        elif keycode == 264 or keycode in (ord('S'), ord('s')):
            controller.set_velocity(controller.velocity_cmd - vel_step)
        elif keycode == 263 or keycode in (ord('A'), ord('a')):
            controller.set_steering(controller.steering_cmd + steer_step)
        elif keycode == 262 or keycode in (ord('D'), ord('d')):
            controller.set_steering(controller.steering_cmd - steer_step)
        elif keycode == 32:
            controller.stop()
        elif keycode in (ord('L'), ord('l')):
            show_lidar = not show_lidar

    print("Ackermann controlled example. Click viewer, use Arrow keys or WASD. Space to stop.")
    with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Center and track the camera on the chassis body
        chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        viewer.cam.trackbodyid = chassis_id
        viewer.cam.fixedcamid = -1
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        while viewer.is_running():
            controller.step()
            mujoco.mj_step(model, data)
            # Visualize rangefinder rays
            if show_lidar and rf_sensor_addrs and rf_site_ids:
                viewer.user_scn.ngeom = 0  # clear previous user geoms
                ray_width = 0.01
                max_len = 5.0
                ray_rgba = np.array([0.1, 0.8, 0.1, 0.9])
                for addr, site_id in zip(rf_sensor_addrs, rf_site_ids):
                    start = np.array(data.site_xpos[site_id])
                    xmat = data.site_xmat[site_id]
                    z_axis = np.array([xmat[2], xmat[5], xmat[8]])
                    direction = -z_axis  # rangefinder casts along -Z
                    distance = float(data.sensordata[addr])
                    length = float(min(max_len, max(0.0, distance)))
                    end = start + direction * length

                    scn = viewer.user_scn
                    geom = scn.geoms[scn.ngeom]
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, ray_width, start, end)
                    geom.rgba[:] = ray_rgba
                    scn.ngeom += 1
            viewer.sync()


if __name__ == "__main__":
    main()
