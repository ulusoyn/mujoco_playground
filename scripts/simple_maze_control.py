"""
Simple Robot Control in Maze Environment with Keyboard Teleop

This script spawns the robot in a Gymnasium maze environment and allows
you to control it with keyboard teleop or automatic movement patterns.

Usage:
    # Keyboard teleop mode (default - use keys in MuJoCo viewer window)
    python scripts/simple_maze_control.py --maze umaze
    
    # Automatic forward movement
    python scripts/simple_maze_control.py --maze umaze --auto
    
    # Custom velocity
    python scripts/simple_maze_control.py --maze umaze --linear 0.5 --angular 0.3

Keyboard Controls (when viewer window is focused):
    Arrow Keys / WASD:
        ↑/W - Forward
        ↓/S - Backward
        ←/A - Turn Left
        →/D - Turn Right
    Space - Stop
    R - Reset environment
    Q/ESC - Quit
"""
import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
import sys
import glfw

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to path for imports
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from rl.envs import AckermannGymnasiumMazeEnv


def main():
    parser = argparse.ArgumentParser(description='Simple Robot Control in Maze')
    parser.add_argument('--maze', type=str, default='umaze',
                        choices=['umaze', 'open', 'medium', 'large'],
                        help='Maze type')
    parser.add_argument('--maze-id', type=str, default='PointMaze_UMaze-v3',
                        help='Gymnasium Robotics maze environment ID')
    parser.add_argument('--max-velocity', type=float, default=1.0,
                        help='Maximum linear velocity (m/s)')
    parser.add_argument('--max-angular-velocity', type=float, default=1.0,
                        help='Maximum angular velocity (rad/s)')
    parser.add_argument('--linear', type=float, default=0.5,
                        help='Linear velocity command (-1 to 1)')
    parser.add_argument('--angular', type=float, default=0.0,
                        help='Angular velocity command (-1 to 1)')
    parser.add_argument('--auto', action='store_true',
                        help='Run automatic forward movement (disables keyboard)')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Number of steps to run')
    
    args = parser.parse_args()
    
    # Map maze type to environment ID
    maze_map = {
        'umaze': 'PointMaze_UMaze-v3',
        'open': 'PointMaze-Open-v3',
        'medium': 'PointMaze-Medium-v3',
        'large': 'PointMaze-Large-v3'
    }
    maze_id = maze_map.get(args.maze, args.maze_id)
    
    print(f"\n{'='*60}")
    print("Simple Robot Control in Maze")
    print(f"{'='*60}")
    print(f"Maze: {maze_id}")
    print(f"Max velocity: {args.max_velocity} m/s")
    print(f"Max angular velocity: {args.max_angular_velocity} rad/s")
    print(f"{'='*60}\n")
    
    # Create environment
    try:
        env = AckermannGymnasiumMazeEnv(
            maze_env_id=maze_id,
            render_mode="human",
            max_linear_velocity=args.max_velocity,
            max_angular_velocity=args.max_angular_velocity,
            goal_distance_threshold=0.5,
            max_episode_steps=args.steps * 2,  # Long episode
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    # Control state (shared with callback)
    linear_vel = args.linear
    angular_vel = args.angular
    step_size = 0.1
    should_reset = False
    should_quit = False
    
    # Keyboard callback for MuJoCo viewer
    # MuJoCo's key_callback receives just the keycode (glfw key constant)
    def key_callback(keycode):
        nonlocal linear_vel, angular_vel, should_reset, should_quit
        
        # Forward / backward
        if keycode in (glfw.KEY_UP, glfw.KEY_W, glfw.KEY_KP_8):
            linear_vel = min(linear_vel + step_size, 1.0)
            print(f"Forward: linear={linear_vel:.2f}")
        elif keycode in (glfw.KEY_DOWN, glfw.KEY_S, glfw.KEY_KP_2):
            linear_vel = max(linear_vel - step_size, -1.0)
            print(f"Backward: linear={linear_vel:.2f}")
        
        # Left / right turn
        elif keycode in (glfw.KEY_LEFT, glfw.KEY_A, glfw.KEY_KP_4):
            angular_vel = min(angular_vel + step_size, 1.0)
            print(f"Turn Left: angular={angular_vel:.2f}")
        elif keycode in (glfw.KEY_RIGHT, glfw.KEY_D, glfw.KEY_KP_6):
            angular_vel = max(angular_vel - step_size, -1.0)
            print(f"Turn Right: angular={angular_vel:.2f}")
        
        # Stop
        elif keycode == glfw.KEY_SPACE or keycode == glfw.KEY_KP_5:
            linear_vel = 0.0
            angular_vel = 0.0
            print("Stop")
        
        # Reset
        elif keycode == glfw.KEY_R:
            should_reset = True
            print("Reset requested...")
        
        # Quit
        elif keycode in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            should_quit = True
            print("Quit requested...")
    
    # Reset environment first
    obs, info = env.reset()
    
    # Close viewer created by environment and recreate with keyboard callback
    if not args.auto and env.viewer is not None:
        env.viewer.close()
        env.viewer = mujoco.viewer.launch_passive(
            env.model, 
            env.data, 
            key_callback=key_callback
        )
    
    print(f"Robot spawned at: {info.get('start_position', 'N/A')}")
    print(f"Goal at: {info.get('goal_position', 'N/A')}")
    
    if args.auto:
        print("\nRunning automatic forward movement...")
        print("Press Ctrl+C to stop\n")
    else:
        print("\nKeyboard Teleop Mode:")
        print("  Focus the MuJoCo viewer window to use keyboard controls")
        print("  Arrow Keys / WASD / Numpad: Move robot")
        print("  Space / Numpad 5: Stop")
        print("  R: Reset")
        print("  Q/ESC: Quit")
        print("  (Make sure the viewer window is focused!)\n")
    
    # Main control loop
    try:
        step_count = 0
        while step_count < args.steps and not should_quit:
            # Handle reset
            if should_reset:
                print("\nResetting...")
                # Close existing viewer
                if env.viewer is not None:
                    env.viewer.close()
                    env.viewer = None
                obs, info = env.reset()
                # Recreate viewer with keyboard callback
                if not args.auto and env.viewer is not None:
                    env.viewer.close()
                    env.viewer = mujoco.viewer.launch_passive(
                        env.model, 
                        env.data, 
                        key_callback=key_callback
                    )
                linear_vel = args.linear
                angular_vel = args.angular
                step_count = 0
                should_reset = False
                print(f"Robot reset at: {info.get('start_position', 'N/A')}")
                print(f"Goal at: {info.get('goal_position', 'N/A')}\n")
            
            # Check if viewer is still running (for keyboard teleop)
            if env.viewer is not None and not args.auto:
                if not env.viewer.is_running():
                    print("\nViewer closed. Exiting...")
                    break
            
            # Auto mode: forward movement
            if args.auto:
                linear_vel = 0.5
                angular_vel = 0.0
            
            # Create action
            action = np.array([linear_vel, angular_vel], dtype=np.float32)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                odom = env.odometry.calculate_odom()
                goal_distance = info.get('goal_distance', float('inf'))
                actual_linear = linear_vel * args.max_velocity
                actual_angular = angular_vel * args.max_angular_velocity
                print(f"Step {step_count}: "
                      f"Linear={actual_linear:.2f} m/s, "
                      f"Angular={np.degrees(actual_angular):.1f} deg/s, "
                      f"Distance={goal_distance:.2f}m, "
                      f"Pos=({odom['position'][0]:.2f}, {odom['position'][1]:.2f})")
            
            # Small delay for visualization
            time.sleep(0.01)
            
            # Check if episode ended
            if terminated:
                print(f"\n✓ Goal reached! Reward: {reward:.2f}")
                print("Resetting...")
                # Close existing viewer
                if env.viewer is not None:
                    env.viewer.close()
                    env.viewer = None
                obs, info = env.reset()
                # Recreate viewer with keyboard callback
                if not args.auto and env.viewer is not None:
                    env.viewer.close()
                    env.viewer = mujoco.viewer.launch_passive(
                        env.model, 
                        env.data, 
                        key_callback=key_callback
                    )
                step_count = 0
            elif truncated:
                print(f"\nEpisode truncated. Resetting...")
                # Close existing viewer
                if env.viewer is not None:
                    env.viewer.close()
                    env.viewer = None
                obs, info = env.reset()
                # Recreate viewer with keyboard callback
                if not args.auto and env.viewer is not None:
                    env.viewer.close()
                    env.viewer = mujoco.viewer.launch_passive(
                        env.model, 
                        env.data, 
                        key_callback=key_callback
                    )
                step_count = 0
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()

