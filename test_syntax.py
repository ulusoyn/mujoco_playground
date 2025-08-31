#!/usr/bin/env python3
"""
Test script for Ackermann robot in MuJoCo
Make sure to rename your Python file to something other than 'mujoco.py'
"""

import mujoco as mj
import numpy as np
import time

def simulate_lidar_with_raycasting(model, data, site_name, num_rays=24, max_range=15.0):
    """
    Simulate LiDAR using MuJoCo's raycasting functionality
    This is an alternative to rangefinder sensors
    """
    # Get site ID
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        print(f"Site '{site_name}' not found")
        return []
    
    # Get site position and orientation
    site_pos = data.site_xpos[site_id].copy()
    site_mat = data.site_xmat[site_id].reshape(3, 3)
    
    distances = []
    
    for i in range(num_rays):
        # Calculate ray direction
        angle = 2 * np.pi * i / num_rays
        local_direction = np.array([np.cos(angle), np.sin(angle), 0])
        world_direction = site_mat @ local_direction
        
        # Perform raycast
        geomid = np.array([-1], dtype=np.int32)
        distance = mj.mj_ray(model, data, site_pos, world_direction, geomid, 
                            bodyexclude=-1, geomgroup=None, flg_static=1, 
                            bodyid=None, geomid_out=geomid)
        
        # Limit range
        if distance < 0 or distance > max_range:
            distance = max_range
            
        distances.append(distance)
    
    return np.array(distances)

def test_basic_model():
    """Test the basic robot model without rangefinders"""
    try:
        # Load the simple model first
        model = mj.MjModel.from_xml_path('simple_test.xml')
        data = mj.MjData(model)
        
        print("✅ Basic model loaded successfully!")
        print(f"Model has {model.nsensor} sensors")
        
        # Print available sensors
        print("\nAvailable sensors:")
        for i in range(model.nsensor):
            sensor_name = model.names[model.name_sensoradr[i]:].decode('utf-8').split('\x00')[0]
            sensor_type = model.sensor_type[i]
            print(f"  {i}: {sensor_name} (type: {sensor_type})")
        
        # Run a few simulation steps
        for i in range(100):
            mj.mj_step(model, data)
        
        # Test LiDAR simulation with raycasting
        print("\nTesting LiDAR simulation with raycasting...")
        lidar_data = simulate_lidar_with_raycasting(model, data, 'lidar_site', num_rays=12)
        print(f"LiDAR distances: {lidar_data}")
        
        # Test IMU data
        print("\nTesting IMU sensors...")
        imu_accel_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, 'imu_accel')
        if imu_accel_id >= 0:
            accel_data = data.sensordata[model.sensor_adr[imu_accel_id]:model.sensor_adr[imu_accel_id]+3]
            print(f"IMU Acceleration: {accel_data}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading basic model: {e}")
        return False

def test_rangefinder_model():
    """Test the model with rangefinder sensors"""
    try:
        # Try to load the rangefinder model
        model = mj.MjModel.from_xml_path('ackermann_robot.xml')
        data = mj.MjData(model)
        
        print("✅ Rangefinder model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading rangefinder model: {e}")
        print("This suggests rangefinder sensors are not available in your MuJoCo version")
        return False

if __name__ == "__main__":
    print("MuJoCo Robot Test")
    print("================")
    
    # Test 1: Basic model
    print("\n1. Testing basic model...")
    basic_success = test_basic_model()
    
    # Test 2: Rangefinder model
    print("\n2. Testing rangefinder model...")
    rangefinder_success = test_rangefinder_model()
    
    if basic_success and not rangefinder_success:
        print("\n💡 Recommendation:")
        print("Use the raycasting-based LiDAR simulation shown above")
        print("It provides the same functionality as rangefinder sensors")
    
    print("\nDone!")