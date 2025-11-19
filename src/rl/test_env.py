"""
Test and Visualize Environment Behavior

This script allows you to test the environment behavior with different agents
before committing to longer training sessions.

Example usage:
    # Test with random actions
    python src/rl/test_env.py --episodes 5
    
    # Test with a trained model
    python src/rl/test_env.py --model-path rl_logs/ppo/ppo_final --episodes 5
    
    # Test on maze environment
    python src/rl/test_env.py --maze umaze --episodes 5
"""
import argparse
import numpy as np
import time
from pathlib import Path
import sys
import mujoco

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to path for imports
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from rl.envs import AckermannRobotEnv

# Try to import Gymnasium maze environment
try:
    from rl.envs import AckermannGymnasiumMazeEnv
    HAS_GYMNASIUM_MAZE = True
except ImportError:
    HAS_GYMNASIUM_MAZE = False


def create_environment(args):
    """Create and return the appropriate environment"""
    render_mode = "human"  # Always render for visualization
    
    if args.maze is not None and HAS_GYMNASIUM_MAZE:
        print(f"Using Gymnasium Robotics maze: {args.maze_id}")
        env = AckermannGymnasiumMazeEnv(
            maze_env_id=args.maze_id,
            render_mode=render_mode,
            max_linear_velocity=args.max_velocity,
            max_angular_velocity=args.max_angular_velocity,
            goal_distance_threshold=args.goal_threshold,
            max_episode_steps=args.max_episode_steps,
        )
    elif args.maze is not None:
        print("Warning: gymnasium-robotics not installed. Using default environment.")
        print("Install with: pip install gymnasium-robotics")
        env = AckermannRobotEnv(
            render_mode=render_mode,
            max_linear_velocity=args.max_velocity,
            max_angular_velocity=args.max_angular_velocity,
            goal_distance_threshold=args.goal_threshold,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        env = AckermannRobotEnv(
            render_mode=render_mode,
            max_linear_velocity=args.max_velocity,
            max_angular_velocity=args.max_angular_velocity,
            goal_distance_threshold=args.goal_threshold,
            max_episode_steps=args.max_episode_steps,
        )
    
    return env


def load_model(model_path, algo=None):
    """Load a trained model"""
    try:
        from stable_baselines3 import PPO, SAC, TD3
        
        # Try to infer algorithm from path if not provided
        if algo is None:
            if 'ppo' in str(model_path).lower():
                algo = 'ppo'
            elif 'sac' in str(model_path).lower():
                algo = 'sac'
            elif 'td3' in str(model_path).lower():
                algo = 'td3'
            else:
                # Try PPO first (most common)
                try:
                    model = PPO.load(str(model_path))
                    print(f"Loaded PPO model from {model_path}")
                    return model
                except:
                    pass
                
                # Try SAC
                try:
                    model = SAC.load(str(model_path))
                    print(f"Loaded SAC model from {model_path}")
                    return model
                except:
                    pass
                
                # Try TD3
                try:
                    model = TD3.load(str(model_path))
                    print(f"Loaded TD3 model from {model_path}")
                    return model
                except:
                    raise ValueError(f"Could not determine algorithm. Please specify --algo")
        
        # Load with specified algorithm
        if algo.lower() == 'ppo':
            model = PPO.load(str(model_path))
        elif algo.lower() == 'sac':
            model = SAC.load(str(model_path))
        elif algo.lower() == 'td3':
            model = TD3.load(str(model_path))
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        print(f"Loaded {algo.upper()} model from {model_path}")
        return model
        
    except ImportError:
        print("Error: stable-baselines3 not installed!")
        print("Install with: pip install stable-baselines3")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def test_with_random_agent(env, num_episodes=5, step_delay=0.01):
    """Test environment with random actions"""
    print(f"\n{'='*60}") 
    print(f"Testing with Random Agent ({num_episodes} episodes)")
    print(f"{'='*60}\n")
    
    episode_stats = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 60)
        
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        min_distance = float('inf')
        collisions = 0
        
        print(f"Start position: {info.get('start_position', 'N/A')}")
        print(f"Goal position: {info.get('goal_position', 'N/A')}")
        
        while True:
            # Random action
            action = env.action_space.sample()
            
            # Debug: Print action and control values for first few steps
            if steps < 1000:
                print(f"  Action: linear={action[0]:.3f}, angular={action[1]:.3f}")
                if hasattr(env, 'controller') and hasattr(env.controller, 'data'):
                    ctrl = env.controller.data.ctrl
                    if len(ctrl) > 0:
                        print(f"  Control values: steer={ctrl[env.controller.act_steer]:.3f}, "
                              f"rear_left={ctrl[env.controller.act_rear_left]:.3f}, "
                              f"rear_right={ctrl[env.controller.act_rear_right]:.3f}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Debug: Print control values after step
            if steps < 1000:
                if hasattr(env, 'controller') and hasattr(env.controller, 'data'):
                    ctrl = env.controller.data.ctrl
                    if len(ctrl) > 0:
                        print(f"  After step - Control: steer={ctrl[env.controller.act_steer]:.3f}, "
                              f"rear_left={ctrl[env.controller.act_rear_left]:.3f}, "
                              f"rear_right={ctrl[env.controller.act_rear_right]:.3f}")
                        # Check robot actual world position (not odometry relative)
                        if hasattr(env, 'data') and hasattr(env, 'model'):
                            try:
                                chassis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
                                world_pos = env.data.xpos[chassis_id]
                                print(f"  Robot world position: x={world_pos[0]:.3f}, y={world_pos[1]:.3f}, z={world_pos[2]:.3f}")
                                # Check wheel positions
                                try:
                                    rear_left_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "rear_left_wheel")
                                    rear_right_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "rear_right_wheel")
                                    rear_left_pos = env.data.xpos[rear_left_id]
                                    rear_right_pos = env.data.xpos[rear_right_id]
                                    print(f"  Rear wheels z: left={rear_left_pos[2]:.3f}, right={rear_right_pos[2]:.3f}")
                                except:
                                    pass
                            except Exception as e:
                                print(f"  Error getting world position: {e}")
                        # Check odometry relative position
                        if hasattr(env, 'odometry'):
                            odom = env.odometry.calculate_odom()
                            print(f"  Odometry relative: x={odom['position'][0]:.3f}, y={odom['position'][1]:.3f}")
            
            total_reward += reward
            steps += 1
            
            # Track statistics
            goal_distance = info.get('goal_distance', float('inf'))
            if goal_distance < min_distance:
                min_distance = goal_distance
            
            if info.get('collision', False):
                collisions += 1
            
            # Small delay for visualization
            time.sleep(step_delay)
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"  Step {steps}: Reward={total_reward:.2f}, "
                      f"Distance={goal_distance:.2f}m, "
                      f"Collisions={collisions}")
            
            if terminated or truncated:
                success = "✓ SUCCESS" if terminated else "✗ TIMEOUT"
                print(f"\n  {success}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Steps: {steps}")
                print(f"  Final distance to goal: {goal_distance:.2f}m")
                print(f"  Min distance reached: {min_distance:.2f}m")
                print(f"  Collisions: {collisions}")
                
                episode_stats.append({
                    'episode': episode + 1,
                    'reward': total_reward,
                    'steps': steps,
                    'success': terminated,
                    'final_distance': goal_distance,
                    'min_distance': min_distance,
                    'collisions': collisions
                })
                break
        
        # Pause between episodes
        if episode < num_episodes - 1:
            print("\nPress Enter to continue to next episode (or Ctrl+C to stop)...")
            try:
                input()
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Episodes completed: {len(episode_stats)}")
    print(f"Success rate: {np.mean([s['success'] for s in episode_stats])*100:.1f}%")
    print(f"Average reward: {np.mean([s['reward'] for s in episode_stats]):.2f} ± {np.std([s['reward'] for s in episode_stats]):.2f}")
    print(f"Average steps: {np.mean([s['steps'] for s in episode_stats]):.1f} ± {np.std([s['steps'] for s in episode_stats]):.1f}")
    print(f"Average final distance: {np.mean([s['final_distance'] for s in episode_stats]):.2f}m")
    print(f"Average min distance: {np.mean([s['min_distance'] for s in episode_stats]):.2f}m")
    print(f"Average collisions: {np.mean([s['collisions'] for s in episode_stats]):.1f}")
    print(f"{'='*60}\n")


def test_with_trained_agent(env, model, num_episodes=5, deterministic=True, step_delay=0.01):
    """Test environment with a trained model"""
    print(f"\n{'='*60}")
    print(f"Testing with Trained Agent ({num_episodes} episodes)")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*60}\n")
    
    episode_stats = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 60)
        
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        min_distance = float('inf')
        collisions = 0
        
        print(f"Start position: {info.get('start_position', 'N/A')}")
        print(f"Goal position: {info.get('goal_position', 'N/A')}")
        
        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Track statistics
            goal_distance = info.get('goal_distance', float('inf'))
            if goal_distance < min_distance:
                min_distance = goal_distance
            
            if info.get('collision', False):
                collisions += 1
            
            # Small delay for visualization
            time.sleep(step_delay)
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"  Step {steps}: Reward={total_reward:.2f}, "
                      f"Distance={goal_distance:.2f}m, "
                      f"Collisions={collisions}")
            
            if terminated or truncated:
                success = "✓ SUCCESS" if terminated else "✗ TIMEOUT"
                print(f"\n  {success}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Steps: {steps}")
                print(f"  Final distance to goal: {goal_distance:.2f}m")
                print(f"  Min distance reached: {min_distance:.2f}m")
                print(f"  Collisions: {collisions}")
                
                episode_stats.append({
                    'episode': episode + 1,
                    'reward': total_reward,
                    'steps': steps,
                    'success': terminated,
                    'final_distance': goal_distance,
                    'min_distance': min_distance,
                    'collisions': collisions
                })
                break
        
        # Pause between episodes
        if episode < num_episodes - 1:
            print("\nPress Enter to continue to next episode (or Ctrl+C to stop)...")
            try:
                input()
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Episodes completed: {len(episode_stats)}")
    print(f"Success rate: {np.mean([s['success'] for s in episode_stats])*100:.1f}%")
    print(f"Average reward: {np.mean([s['reward'] for s in episode_stats]):.2f} ± {np.std([s['reward'] for s in episode_stats]):.2f}")
    print(f"Average steps: {np.mean([s['steps'] for s in episode_stats]):.1f} ± {np.std([s['steps'] for s in episode_stats]):.1f}")
    print(f"Average final distance: {np.mean([s['final_distance'] for s in episode_stats]):.2f}m")
    print(f"Average min distance: {np.mean([s['min_distance'] for s in episode_stats]):.2f}m")
    print(f"Average collisions: {np.mean([s['collisions'] for s in episode_stats]):.1f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test and Visualize Environment Behavior',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Agent selection
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model (if None, uses random agent)')
    parser.add_argument('--algo', type=str, default=None,
                        choices=['ppo', 'sac', 'td3'],
                        help='Algorithm type (auto-detected if not specified)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic policy (for trained models)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (overrides --deterministic)')
    
    # Test parameters
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--step-delay', type=float, default=0.01,
                        help='Delay between steps (seconds) for visualization')
    
    # Environment parameters
    parser.add_argument('--maze', type=str, default=None,
                        choices=[None, 'umaze', 'open', 'medium', 'large'],
                        help='Use Gymnasium Robotics maze')
    parser.add_argument('--maze-id', type=str, default='PointMaze_UMaze-v3',
                        help='Gymnasium Robotics maze environment ID')
    parser.add_argument('--max-velocity', type=float, default=1.0,
                        help='Maximum linear velocity (m/s)')
    parser.add_argument('--max-angular-velocity', type=float, default=1.0,
                        help='Maximum angular velocity (rad/s)')
    parser.add_argument('--goal-threshold', type=float, default=0.5,
                        help='Goal distance threshold (m)')
    parser.add_argument('--max-episode-steps', type=int, default=1000,
                        help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    # Handle deterministic/stochastic
    if args.stochastic:
        args.deterministic = False
    
    # Create environment
    print("Creating environment...")
    env = create_environment(args)
    
    print(f"\n{'='*60}")
    print("Environment Information")
    print(f"{'='*60}")
    print(f"Environment: {type(env).__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"{'='*60}\n")
    
    # Load model if provided
    model = None
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            # Try relative to project root
            model_path = project_root / args.model_path
            if not model_path.exists():
                print(f"Error: Model path not found: {args.model_path}")
                return
        
        model = load_model(model_path, args.algo)
        if model is None:
            print("Failed to load model. Exiting.")
            return
    
    # Run tests
    try:
        if model is not None:
            test_with_trained_agent(
                env, model, 
                num_episodes=args.episodes,
                deterministic=args.deterministic,
                step_delay=args.step_delay
            )
        else:
            test_with_random_agent(
                env,
                num_episodes=args.episodes,
                step_delay=args.step_delay
            )
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()

