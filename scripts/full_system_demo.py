"""
Complete system demonstration with MuJoCo viewer and cmd_vel control
This integrates everything: teleop input -> cmd_vel -> Ackermann controller -> MuJoCo robot
"""
import sys
from pathlib import Path
import numpy as np
import time
import threading

# Add project root to path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import mujoco
import mujoco.viewer as mj_viewer
from cmd_vel_message import cmd_vel_publisher, Twist
from ackermann_cmd_vel_controller import AckermannCmdVelController, AckermannParams
from mujoco_ackermann_bridge import MuJoCoAckermannBridge, set_spawn_pose


class FullSystemDemo:
    """
    Complete system demonstration with visualization
    """
    
    def __init__(self, xml_path: str, use_joystick: bool = False):
        self.use_joystick = use_joystick
        self.running = True
        
        # Load MuJoCo model
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as exc:
            print(f"Failed to load XML at {xml_path}: {exc}")
            sys.exit(1)
        
        # Initialize robot position
        set_spawn_pose(self.model, self.data, xy=(0.0, 0.0), z=0.09, yaw_rad=0.0)
        
        # Create bridge
        self.bridge = MuJoCoAckermannBridge(self.model, self.data)
        
        # Setup rangefinder visualization
        self.setup_lidar_visualization()
        
        print("Full System Demo initialized")
        print(f"Using {'joystick' if use_joystick else 'keyboard'} input")
    
    def setup_lidar_visualization(self):
        """Setup lidar rangefinder visualization data"""
        rf_count = 36
        self.rf_sensor_addrs = []
        self.rf_site_ids = []
        
        for i in range(rf_count):
            sensor_name = f"rf_360_s{i:02d}"
            site_name = f"lidar_360_s{i:02d}"
            
            s_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if s_id == -1:
                continue
                
            self.rf_sensor_addrs.append(self.model.sensor_adr[s_id])
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            self.rf_site_ids.append(site_id)
    
    def start_teleop(self):
        """Start teleoperation in background thread"""
        if self.use_joystick:
            teleop_thread = threading.Thread(target=self._start_joystick_teleop, daemon=True)
        else:
            teleop_thread = threading.Thread(target=self._start_keyboard_teleop, daemon=True)
        
        teleop_thread.start()
        time.sleep(0.5)  # Give teleop time to initialize
    
    def _start_keyboard_teleop(self):
        """Start keyboard teleop"""
        try:
            from keyboard_teleop import KeyboardTeleop
            teleop = KeyboardTeleop(max_linear_vel=1.5, max_angular_vel=3.0)
            teleop.start()
        except ImportError as e:
            print(f"Could not start keyboard teleop: {e}")
            print("Install keyboard module: pip install keyboard")
    
    def _start_joystick_teleop(self):
        """Start joystick teleop"""
        try:
            from joystick_teleop import JoystickTeleop
            teleop = JoystickTeleop(max_linear_vel=1.5, max_angular_vel=3.0)
            teleop.start()
        except ImportError as e:
            print(f"Could not start joystick teleop: {e}")
            print("Install pygame: pip install pygame")
    
    def run_with_viewer(self):
        """Run the demo with MuJoCo viewer"""
        show_lidar = True
        
        def key_callback(keycode: int):
            nonlocal show_lidar
            # Toggle lidar visualization with 'L'
            if keycode in (ord('L'), ord('l')):
                show_lidar = not show_lidar
                print(f"Lidar visualization: {'ON' if show_lidar else 'OFF'}")
            # Emergency stop with Space (if keyboard teleop not running)
            elif keycode == 32:  # Space
                cmd_vel_publisher.publish(Twist())
                print("Emergency stop!")
        
        print("\n" + "="*60)
        print("FULL SYSTEM DEMO")
        print("="*60)
        if self.use_joystick:
            print("Joystick Controls:")
            print("  Left stick: Move robot (vertical=forward/back, horizontal=left/right)")
            print("  Any button: Emergency stop")
        else:
            print("Keyboard Controls:")
            print("  W/S or ↑/↓: Forward/Backward")
            print("  A/D or ←/→: Turn Left/Right") 
            print("  Space: Stop")
            print("  Q/Esc: Quit teleop")
        
        print("\nViewer Controls:")
        print("  L: Toggle lidar visualization")
        print("  Space: Emergency stop (backup)")
        print("  Mouse: Camera control")
        print("="*60)
        
        # Start teleop
        self.start_teleop()
        
        with mj_viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
            # Setup camera to track robot
            chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
            viewer.cam.trackbodyid = chassis_id
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            
            last_state_print = 0.0
            state_print_interval = 2.0
            
            print("Demo running! Move the robot with your input device.")
            
            while viewer.is_running():
                # Update bridge (cmd_vel -> robot control)
                self.bridge.step()
                
                # Step physics
                mujoco.mj_step(self.model, self.data)
                
                # Visualize lidar
                if show_lidar and self.rf_sensor_addrs and self.rf_site_ids:
                    self._visualize_lidar(viewer)
                
                # Print robot state occasionally
                current_time = time.time()
                if current_time - last_state_print > state_print_interval:
                    self._print_robot_state()
                    last_state_print = current_time
                
                # Sync viewer
                viewer.sync()
        
        # Clean shutdown
        self.stop()
    
    def _visualize_lidar(self, viewer):
        """Add lidar rays to viewer"""
        viewer.user_scn.ngeom = 0  # Clear previous rays
        
        ray_width = 0.008
        max_len = 4.0
        ray_rgba = np.array([0.2, 0.9, 0.2, 0.8])
        
        for addr, site_id in zip(self.rf_sensor_addrs, self.rf_site_ids):
            if site_id == -1:
                continue
                
            # Get ray start position and direction
            start = np.array(self.data.site_xpos[site_id])
            xmat = self.data.site_xmat[site_id].reshape(3, 3)
            direction = -xmat[:, 2]  # -Z axis of site
            
            # Get distance reading
            distance = float(self.data.sensordata[addr])
            length = min(max_len, max(0.1, distance))
            end = start + direction * length
            
            # Add ray to viewer
            scn = viewer.user_scn
            if scn.ngeom < scn.maxgeom:
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, ray_width, start, end)
                geom.rgba[:] = ray_rgba
                scn.ngeom += 1
    
    def _print_robot_state(self):
        """Print current robot state"""
        state = self.bridge.get_robot_state()
        pos = state['position']
        yaw_deg = np.degrees(state['yaw'])
        
        print(f"\n[State] Position: ({pos['x']:+.2f}, {pos['y']:+.2f}), "
              f"Yaw: {yaw_deg:+.1f}°")
        
        if state['cmd_valid']:
            print(f"[State] Commands: v={state['linear_cmd']:+.2f}m/s, "
                  f"ω={state['angular_cmd']:+.2f}rad/s")
        else:
            print("[State] No valid command (timeout)")
    
    def stop(self):
        """Stop the demo"""
        self.running = False
        # Send final stop command
        cmd_vel_publisher.publish(Twist())
        print("Demo stopped.")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Ackermann Robot Demo")
    parser.add_argument("--joystick", action="store_true", 
                       help="Use joystick instead of keyboard")
    parser.add_argument("--list-joysticks", action="store_true",
                       help="List available joysticks and exit")
    args = parser.parse_args()
    
    if args.list_joysticks:
        try:
            from joystick_teleop import list_joysticks
            list_joysticks()
        except ImportError:
            print("pygame not installed. Install with: pip install pygame")
        return
    
    # Find robot XML file
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    xml_path = str(project_root / "models" / "ackermann_robot.xml")
    
    if not Path(xml_path).exists():
        print(f"Robot XML file not found at: {xml_path}")
        sys.exit(1)
    
    # Create and run demo
    demo = FullSystemDemo(xml_path, use_joystick=args.joystick)
    
    try:
        demo.run_with_viewer()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        demo.stop()


if __name__ == "__main__":
    main()