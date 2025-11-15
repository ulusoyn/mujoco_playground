# Ackermann Robot Simulation - System Summary

## ✅ Completed Components

### 1. **Robot Model** (`models/ackermann_robot_v2.xml`)
- ✓ Ackermann steering with bicycle model
- ✓ 4 wheels with proper physics (wheelbase: 0.20m, track: 0.174m, radius: 0.0325m)
- ✓ 360° Lidar with 72 beams arranged in 3.5cm radius circle
- ✓ Robot geometry in collision group 2 (prevents lidar self-detection)
- ✓ Proper mass distribution and friction parameters

### 2. **Controller** (`src/core/controller.py`)
- ✓ **BicycleController**: Single steering servo (matches v2 model)
- ✓ **AckermannController**: Independent left/right steering
- ✓ Perfectly aligned with robot dimensions
- ✓ Differential wheel velocity calculation for turning
- ✓ Steering angle limits: ±35°

### 3. **Odometry** (`src/core/odometry.py`)
- ✓ Position tracking (x, y, z)
- ✓ Orientation (quaternion and Euler angles)
- ✓ Heading angle
- ✓ Distance traveled
- ✓ Velocity estimation

### 4. **Sensor Readings** (`src/main_sim.py`)
- ✓ 72 rangefinder sensors (lidar-0 to lidar-71)
- ✓ Real-time lidar visualization (yellow rays)
- ✓ Wheel encoders (position and velocity)
- ✓ Steering angle feedback
- ✓ No -1 values issue (fixed with collision groups)

### 5. **Map Spawner** (`src/environments/map_spawner.py`) ⭐ NEW
- ✓ Automatic map discovery from `mobile-mujoco-environments`
- ✓ Random map loading
- ✓ Specific map selection (by index or name)
- ✓ Dynamic XML merging (map + robot)
- ✓ Smart robot spawning in empty spaces
- ✓ Custom spawn position/orientation support

### 6. **Teleoperation** (`src/teleop/`)
- ✓ Keyboard control (W/A/S/D)
- ✓ Joystick support (optional)
- ✓ Cmd_vel message format

## 📁 Project Structure

```
mujoco_playground/
├── models/
│   ├── ackermann_robot_v2.xml          ⭐ Main robot model (WORKING)
│   ├── ackermann_robot.xml             (old version)
│   └── environments/
│       ├── ackermann_in_mushr_maze.xml
│       └── ackermann_maze_flat.xml
├── src/
│   ├── core/
│   │   ├── controller.py               ✓ Robot controller
│   │   ├── odometry.py                 ✓ Odometry tracking
│   │   └── cmd_vel_message.py          ✓ Message format
│   ├── teleop/
│   │   ├── keyboard_teleop.py          ✓ Keyboard control
│   │   └── joystick_teleop.py          ✓ Joystick control
│   ├── environments/                   ⭐ NEW
│   │   ├── map_spawner.py              ✓ Dynamic map loading
│   │   ├── demo_map_spawner.py         ✓ Demo script
│   │   └── README.md                   ✓ Documentation
│   └── main_sim.py                     ✓ Main simulation
├── mobile-mujoco-environments/         📦 Map library
│   └── envs/assets/
│       ├── maze.xml
│       ├── mushr_maze.xml
│       ├── mushr_elevation.xml
│       └── ... (more maps)
├── CAD Models/
│   ├── Base.stl                        ✓ Robot chassis mesh
│   └── Ceiling.stl                     ✓ Robot top mesh
└── requirements.txt                    ✓ Dependencies
```

## 🎮 How to Use

### Basic Simulation (Single Map)
```bash
python3 src/main_sim.py
```
Controls: W/A/S/D for movement, Space to stop

### Random Map Simulation
```bash
python3 src/environments/demo_map_spawner.py
```

### Custom Integration
```python
from src.environments.map_spawner import MapSpawner
from src.core.controller import BicycleController
from src.core.odometry import Odometry

# Load random map
spawner = MapSpawner()
model, data, map_name = spawner.load_random_environment(
    robot_pos=[0, 0, 0.1],
    robot_quat=None  # Random orientation
)

# Initialize systems
controller = BicycleController(model, data)
odometry = Odometry(model, data, robot_body_name="chassis")

# Control loop
controller.apply_cmd_vel(linear_x=0.5, angular_z=0.1)
odom = odometry.calculate_odom()
```

## 🔧 Robot Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Wheelbase** | 0.20 m | Distance between front and rear axles |
| **Track Width** | 0.174 m | Distance between left and right wheels |
| **Wheel Radius** | 0.0325 m | 3.25 cm |
| **Chassis Length** | 0.275 m | 27.5 cm |
| **Chassis Width** | 0.15 m | 15 cm |
| **Steering Range** | ±35° | ±0.61 rad |
| **Lidar Beams** | 72 | 5° spacing, 360° coverage |
| **Lidar Radius** | 0.035 m | 3.5 cm circle |
| **Lidar Cutoff** | 12 m | Maximum detection range |

## 🎯 Next Steps (TODO)

### Goal Position System
- [ ] Goal marker visualization
- [ ] Distance to goal calculation
- [ ] Path planning integration
- [ ] Goal reached detection

### Advanced Features
- [ ] Collision detection and avoidance
- [ ] Occupancy grid mapping
- [ ] SLAM integration
- [ ] Trajectory recording/playback
- [ ] Multiple robot support

### Map Improvements
- [ ] Better empty space detection algorithm
- [ ] Collision checking for spawn positions
- [ ] Map bounds detection
- [ ] Custom map creation tools

## 🐛 Known Issues & Solutions

### ✅ SOLVED: Lidar values becoming -1
**Solution**: Robot geometry moved to collision group 2

### ✅ SOLVED: All lidar beams showing same value
**Solution**: Fixed sensor naming (lidar-0 to lidar-71)

### ✅ SOLVED: Lidar beams not radiating in circle
**Solution**: Used `pos="0.035 0 0"` with `euler="0 0 5"` in replicate

### ✅ SOLVED: Controller not matching robot dimensions
**Solution**: Verified and aligned all parameters

## 📊 System Status

| Component | Status | Notes |
|-----------|--------|-------|
| Robot Model | ✅ Working | v2 is production-ready |
| Controller | ✅ Working | BicycleController recommended |
| Odometry | ✅ Working | Accurate tracking |
| Lidar | ✅ Working | 72 beams, no -1 values |
| Teleop | ✅ Working | Keyboard + Joystick |
| Map Spawner | ✅ Working | Dynamic loading ready |
| Goal System | ⏳ TODO | Next priority |

## 🚀 Performance

- **Timestep**: 0.002s (500 Hz)
- **Real-time Factor**: ~1.0 (matches real-time)
- **Lidar Update Rate**: 500 Hz (same as physics)
- **Visualization**: Smooth 60 FPS

## 📝 Notes

- Always use `ackermann_robot_v2.xml` (not v1)
- Robot spawns at z=0.065m (chassis center height)
- Lidar at z=0.03m relative to chassis (z=0.095m world)
- Collision groups: 0=environment, 2=robot, prevents self-detection
- Sensor naming: `lidar-{i}` and `rf-{i}` (i=0 to 71)



