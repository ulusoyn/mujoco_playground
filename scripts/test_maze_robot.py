"""
Interactive Robot Control in Maze Environment

This script spawns the robot in a Gymnasium maze environment and allows
you to manually control it using keyboard commands.

Controls:
    W/S - Forward/Backward
    A/D - Turn Left/Right
    Space - Stop
    R - Reset
    Q - Quit

Usage:
    python scripts/test_maze_robot.py --maze umaze
"""
import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
import sys
import select
import tty
import termios

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to path for imports
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from rl.envs import AckermannGymnasiumMazeEnv
from core.controller import BicycleController
from core.odometry import Odometry


def main():
    parser = argparse.ArgumentParser(description='Interactive Robot Control in Maze')
    parser.add_argument('--maze', type=str, default='umaze',
                        choices=['umaze', 'open', 'medium', 'large'],
                        help='Maze type')
    parser.add_argument('--maze-id', type=str, default='PointMaze_UMaze-v3',
                        help='Gymnasium Robotics maze environment ID')
    parser.add_argument('--max-velocity', type=float, default=1.0,
                        help='Maximum linear velocity (m/s)')
    parser.add_argument('--max-angular-velocity', type=float, default=1.0,
                        help='Maximum angular velocity (rad/s)')
    
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
    print("Interactive Robot Control in Maze")
    print(f"{'='*60}")
    print(f"Maze: {maze_id}")
    print(f"Max velocity: {args.max_velocity} m/s")
    print(f"Max angular velocity: {args.max_angular_velocity} rad/s")
    print(f"\nControls:")
    print(f"  W/S - Forward/Backward")
    print(f"  A/D - Turn Left/Right")
    print(f"  Space - Stop")
    print(f"  R - Reset")
    print(f"  Q - Quit")
    print(f"{'='*60}\n")
    
    # Create environment
    try:
        env = AckermannGymnasiumMazeEnv(
            maze_env_id=maze_id,
            render_mode="human",
            max_linear_velocity=args.max_velocity,
            max_angular_velocity=args.max_angular_velocity,
            goal_distance_threshold=0.5,
            max_episode_steps=10000,  # Long episode for manual control
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"Robot spawned at: {info.get('start_position', 'N/A')}")
    print(f"Goal at: {info.get('goal_position', 'N/A')}")
    print("\nStarting control loop...\n")
    
    # Control state
    linear_vel = 0.0
    angular_vel = 0.0
    step_size = 0.1  # Velocity increment per key press
    
    # Main control loop
    try:
        while True:
            # Check if input is available (non-blocking)
            if select.select([sys.stdin], [], [], 0.01)[0]:
                key = sys.stdin.read(1).lower()
                
                if key == 'w':
                    linear_vel = min(linear_vel + step_size, 1.0)
                elif key == 's':
                    linear_vel = max(linear_vel - step_size, -1.0)
                elif key == 'a':
                    angular_vel = min(angular_vel + step_size, 1.0)
                elif key == 'd':
                    angular_vel = max(angular_vel - step_size, -1.0)
                elif key == ' ':  # Space
                    linear_vel = 0.0
                    angular_vel = 0.0
                elif key == 'r':
                    print("\nResetting environment...")
                    obs, info = env.reset()
                    linear_vel = 0.0
                    angular_vel = 0.0
                    print(f"Robot reset at: {info.get('start_position', 'N/A')}")
                    print(f"Goal at: {info.get('goal_position', 'N/A')}\n")
                elif key == 'q':
                    print("\nQuitting...")
                    break
            
            # Convert normalized velocities to actual velocities
            actual_linear = linear_vel * args.max_linear_velocity
            actual_angular = angular_vel * args.max_angular_velocity
            
            # Create action
            action = np.array([linear_vel, angular_vel], dtype=np.float32)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print status occasionally
            if env.step_count % 50 == 0:
                odom = env.odometry.calculate_odom()
                goal_distance = info.get('goal_distance', float('inf'))
                print(f"Step {env.step_count}: "
                      f"Linear={actual_linear:.2f} m/s, "
                      f"Angular={np.degrees(actual_angular):.1f} deg/s, "
                      f"Distance to goal={goal_distance:.2f}m, "
                      f"Position=({odom['position'][0]:.2f}, {odom['position'][1]:.2f})")
            
            # Small delay for visualization
            time.sleep(0.01)
            
            # Check if episode ended
            if terminated:
                print(f"\n✓ Goal reached! Reward: {reward:.2f}")
                print("Press 'R' to reset or 'Q' to quit")
            elif truncated:
                print(f"\nEpisode truncated at step {env.step_count}")
                print("Press 'R' to reset or 'Q' to quit")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    # Set up terminal for non-blocking input
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore terminal settings
        if old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass

