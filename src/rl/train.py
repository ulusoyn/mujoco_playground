"""
Full Training Script for Ackermann Robot RL

This script provides a complete training pipeline for the Ackermann robot
using Stable-Baselines3 algorithms.

Example usage:
    # Train with PPO on maze environment
    python src/rl/train.py --algo ppo --maze umaze --timesteps 100000 --render
    
    # Train with SAC
    python src/rl/train.py --algo sac --maze umaze --timesteps 200000
    
    # Train with custom hyperparameters
    python src/rl/train.py --algo ppo --maze umaze --timesteps 500000 --learning-rate 1e-4
"""
import argparse
import numpy as np
import os
from pathlib import Path
import sys

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # src/rl -> src -> project_root
os.chdir(project_root)
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


def train_with_stable_baselines3(env, algo='ppo', total_timesteps=100000, render=False, 
                                  learning_rate=None, save_freq=10000, eval_freq=10000, 
                                  eval_episodes=10, **kwargs):
    """Train using Stable-Baselines3"""
    try:
        from stable_baselines3 import PPO, SAC, TD3
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("Error: stable-baselines3 not installed!")
        print("Install with: pip install stable-baselines3")
        return None
    
    # Store reference to unwrapped env for viewer access BEFORE wrapping
    unwrapped_env = env
    
    # Reset environment first to initialize viewer if rendering
    if render:
        print("Initializing environment and viewer...")
        obs, info = env.reset()
        print(f"Viewer initialized: {hasattr(env, 'viewer') and env.viewer is not None}")
        if hasattr(env, 'viewer') and env.viewer is not None:
            print("Viewer is ready for visualization")
    
    # Wrap environment
    env = Monitor(env)
    
    # Create environment factory for DummyVecEnv
    def make_env():
        return env
    
    vec_env = DummyVecEnv([make_env])
    
    # Store reference to vec_env for viewer access in callback
    # The vec_env contains the wrapped environment, but we need to access the unwrapped one
    if render:
        # Try to get viewer from vec_env's environment
        try:
            vec_env_env = vec_env.envs[0]
            # Unwrap Monitor
            while hasattr(vec_env_env, 'env'):
                vec_env_env = vec_env_env.env
            # Check if viewer is accessible
            if hasattr(vec_env_env, 'viewer') and vec_env_env.viewer is not None:
                print(f"Viewer accessible from vec_env: {vec_env_env.viewer is not None}")
        except Exception as e:
            print(f"Warning: Could not access viewer from vec_env: {e}")
    
    # Set default hyperparameters
    algo_kwargs = {}
    if learning_rate is not None:
        algo_kwargs['learning_rate'] = learning_rate
    
    # Algorithm-specific defaults
    if algo.lower() == 'ppo':
        algo_kwargs.setdefault('learning_rate', 3e-4)
        algo_kwargs.setdefault('n_steps', 2048)
        algo_kwargs.setdefault('batch_size', 64)
        algo_kwargs.setdefault('n_epochs', 10)
        algo_kwargs.setdefault('gamma', 0.99)
        algo_kwargs.setdefault('gae_lambda', 0.95)
        algo_kwargs.setdefault('clip_range', 0.2)
        algo_kwargs.setdefault('ent_coef', 0.01)
    elif algo.lower() == 'sac':
        algo_kwargs.setdefault('learning_rate', 3e-4)
        algo_kwargs.setdefault('buffer_size', 100000)
        algo_kwargs.setdefault('learning_starts', 1000)
        algo_kwargs.setdefault('batch_size', 256)
        algo_kwargs.setdefault('tau', 0.005)
        algo_kwargs.setdefault('gamma', 0.99)
    elif algo.lower() == 'td3':
        algo_kwargs.setdefault('learning_rate', 3e-4)
        algo_kwargs.setdefault('buffer_size', 100000)
        algo_kwargs.setdefault('learning_starts', 1000)
        algo_kwargs.setdefault('batch_size', 256)
        algo_kwargs.setdefault('tau', 0.005)
        algo_kwargs.setdefault('gamma', 0.99)
    
    # Merge with user-provided kwargs
    algo_kwargs.update(kwargs)
    
    # Create model
    if algo.lower() == 'ppo':
        model = PPO('MlpPolicy', vec_env, verbose=1, **algo_kwargs)
    elif algo.lower() == 'sac':
        model = SAC('MlpPolicy', vec_env, verbose=1, **algo_kwargs)
    elif algo.lower() == 'td3':
        model = TD3('MlpPolicy', vec_env, verbose=1, **algo_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Setup callbacks
    log_dir = project_root / "rl_logs" / algo
    log_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(log_dir),
        name_prefix=f"{algo}_model"
    )
    
    # Add evaluation callback if eval_freq is set
    eval_callback = None
    if eval_freq > 0:
        eval_log_dir = log_dir / "eval"
        eval_callback = EvalCallback(
            vec_env,
            best_model_save_path=str(eval_log_dir),
            log_path=str(eval_log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False
        )
    
    # Combine callbacks
    callbacks = [checkpoint_callback]
    if eval_callback is not None:
        callbacks.append(eval_callback)
    # Note: No render callback needed - environment handles syncing automatically in each step
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training {algo.upper()} agent")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Log directory: {log_dir}")
    if render:
        print("(Visualization enabled - you can see the robot training)")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = log_dir / f"{algo}_final"
    model.save(str(final_model_path))
    print(f"\n✓ Model saved to: {final_model_path}")
    
    return model


def train_with_custom_algo(env, episodes=1000, render=False):
    """Simple random baseline for testing"""
    print(f"\nTraining with random policy ({episodes} episodes)...")
    if render:
        print("(Visualization enabled - you can see the robot training)")
    
    import time
    
    returns = []
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Note: Viewer syncing is handled automatically by environment.step()
            # Small delay for visualization if rendering
            if render:
                time.sleep(0.01)
            
            if terminated or truncated:
                break
        
        returns.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_return = np.mean(returns[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Return (last 100): {avg_return:.2f}")
    
    print(f"\n✓ Training complete!")
    print(f"Average return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Best return: {np.max(returns):.2f}")
    
    return returns


def main():
    parser = argparse.ArgumentParser(description='Train Ackermann Robot RL Agent')
    parser.add_argument('--algo', type=str, default='random',
                        choices=['random', 'ppo', 'sac', 'td3'],
                        help='RL algorithm to use')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes (for random)')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Number of timesteps (for SB3 algorithms)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--max-velocity', type=float, default=1.0,
                        help='Maximum linear velocity (m/s)')
    parser.add_argument('--goal-threshold', type=float, default=0.5,
                        help='Goal distance threshold (m)')
    parser.add_argument('--maze', type=str, default=None,
                        choices=[None, 'umaze', 'open', 'medium', 'large'],
                        help='Use Gymnasium Robotics maze (requires gymnasium-robotics)')
    parser.add_argument('--maze-id', type=str, default='PointMaze_UMaze-v3',
                        help='Gymnasium Robotics maze environment ID')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate for the algorithm')
    parser.add_argument('--save-freq', type=int, default=10000,
                        help='Frequency to save model checkpoints')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Frequency to evaluate the model')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    # Create environment
    if args.maze is not None and HAS_GYMNASIUM_MAZE:
        print(f"Using Gymnasium Robotics maze: {args.maze_id}")
        env = AckermannGymnasiumMazeEnv(
            maze_env_id=args.maze_id,
            render_mode="human" if args.render else None,
            max_linear_velocity=args.max_velocity,
            goal_distance_threshold=args.goal_threshold,
        )
    elif args.maze is not None:
        print("Warning: gymnasium-robotics not installed. Using default environment.")
        print("Install with: pip install gymnasium-robotics")
        env = AckermannRobotEnv(
            render_mode="human" if args.render else None,
            max_linear_velocity=args.max_velocity,
            goal_distance_threshold=args.goal_threshold,
        )
    else:
        env = AckermannRobotEnv(
            render_mode="human" if args.render else None,
            max_linear_velocity=args.max_velocity,
            goal_distance_threshold=args.goal_threshold,
        )
    
    print(f"\n{'='*60}")
    print("Ackermann Robot RL Training")
    print(f"{'='*60}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"{'='*60}\n")
    
    # Train
    if args.algo == 'random':
        train_with_custom_algo(env, episodes=args.episodes, render=args.render)
    else:
        model = train_with_stable_baselines3(
            env,
            algo=args.algo,
            total_timesteps=args.timesteps,
            render=args.render,
            learning_rate=args.learning_rate,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes
        )
        
        # Test trained model
        if model is not None:
            print("\n" + "="*60)
            print("Testing trained model...")
            print("="*60)
            
            from rl.utils import evaluate_agent
            stats = evaluate_agent(env, model, num_episodes=args.eval_episodes)
            print(f"\nEvaluation Results:")
            print(f"  Mean Return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
            print(f"  Mean Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
            print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
            
            # Visual test if rendering
            if args.render:
                print("\nRunning visual test (close viewer to stop)...")
                import time
                obs, info = env.reset()
                episode_count = 0
                step_count = 0
                while episode_count < 5 and step_count < 1000:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_count += 1
                    
                    # Note: Viewer syncing is handled automatically by environment.step()
                    # Small delay for visualization
                    time.sleep(0.01)
                    
                    if terminated or truncated:
                        print(f"Episode {episode_count + 1} finished! Reward: {reward:.2f}")
                        episode_count += 1
                        obs, info = env.reset()
    
    env.close()


if __name__ == "__main__":
    main()

