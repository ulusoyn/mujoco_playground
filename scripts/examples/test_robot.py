import test_robot as mj
import numpy as np

# Load model
model = mj.MjModel.from_xml_path('ackermann_robot.xml')
data = mj.MjData(model)

# Get LiDAR data
lidar_data = []
for i in range(24):  # 24 LiDAR sensors
    sensor_name = f"lidar_{i*15:03d}"
    sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, sensor_name)
    distance = data.sensordata[sensor_id]
    lidar_data.append(distance)

# Get IMU data
imu_accel = data.sensordata[model.sensor_adr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "imu_accel")]:
                            model.sensor_adr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "imu_accel")] + 3]