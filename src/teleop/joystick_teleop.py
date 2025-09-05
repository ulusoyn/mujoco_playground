"""
Joystick teleoperation node that publishes cmd_vel messages
"""
import threading
import time
import sys
from pathlib import Path

# Add project root to path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from cmd_vel_message import Twist, cmd_vel_publisher

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    print("Warning: 'pygame' module not installed. Install with: pip install pygame")
    HAS_PYGAME = False


class JoystickTeleop:
    """Joystick teleoperation for publishing cmd_vel messages"""
    
    def __init__(self, max_linear_vel: float = 1.5, max_angular_vel: float = 3.0,
                 deadzone: float = 0.1):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.deadzone = deadzone
        
        # Joystick configuration
        self.joystick = None
        self.axis_linear = 1   # Usually left stick vertical (forward/backward)
        self.axis_angular = 0  # Usually left stick horizontal (left/right)
        
        # Publishing
        self.publish_rate = 20  # Hz
        self.running = True
        self.publisher_thread = None
        
        # Initialize pygame
        if HAS_PYGAME:
            pygame.init()
            pygame.joystick.init()
        
        print("Joystick Teleoperation Started!")
        print(f"Max linear velocity: {max_linear_vel:.1f} m/s")
        print(f"Max angular velocity: {max_angular_vel:.1f} rad/s")
        print(f"Deadzone: {deadzone:.2f}")
    
    def start(self):
        """Start the joystick teleop node"""
        if not HAS_PYGAME:
            print("Cannot start joystick teleop without pygame")
            return
        
        # Check for joysticks
        if pygame.joystick.get_count() == 0:
            print("No joystick detected. Please connect a joystick and try again.")
            return
        
        # Initialize first joystick
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"Connected to joystick: {self.joystick.get_name()}")
        print(f"Number of axes: {self.joystick.get_numaxes()}")
        print(f"Number of buttons: {self.joystick.get_numbuttons()}")
        print("Controls:")
        print(f"  Left stick vertical (axis {self.axis_linear}): Linear velocity")
        print(f"  Left stick horizontal (axis {self.axis_angular}): Angular velocity") 
        print("  Any button: Emergency stop")
        print("-" * 50)
        
        # Start publisher thread
        self.publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        self.publisher_thread.start()
        
        try:
            self._joystick_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the teleop node"""
        self.running = False
        # Send stop command
        self._publish_cmd_vel(0.0, 0.0)
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        print("Joystick teleop stopped.")
    
    def _joystick_loop(self):
        """Main joystick reading loop"""
        clock = pygame.time.Clock()
        emergency_stop = False
        
        print("Joystick ready. Press Ctrl+C to quit.")
        
        while self.running:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button {event.button} pressed - Emergency stop activated!")
                    emergency_stop = True
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"Button {event.button} released - Emergency stop deactivated")
                    emergency_stop = False
                elif event.type == pygame.QUIT:
                    self.running = False
            
            # Read joystick axes
            if not emergency_stop and self.joystick:
                try:
                    # Get axis values (-1.0 to 1.0)
                    raw_linear = -self.joystick.get_axis(self.axis_linear)  # Invert for intuitive control
                    raw_angular = -self.joystick.get_axis(self.axis_angular)  # Invert for intuitive control
                    
                    # Apply deadzone
                    linear_vel = self._apply_deadzone(raw_linear) * self.max_linear_vel
                    angular_vel = self._apply_deadzone(raw_angular) * self.max_angular_vel
                    
                    # Store current velocities for publisher thread
                    self.current_linear_vel = linear_vel
                    self.current_angular_vel = angular_vel
                    
                except pygame.error as e:
                    print(f"Joystick error: {e}")
                    break
            else:
                # Emergency stop or no joystick
                self.current_linear_vel = 0.0
                self.current_angular_vel = 0.0
            
            clock.tick(60)  # 60 FPS for responsive control
    
    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick input"""
        if abs(value) < self.deadzone:
            return 0.0
        else:
            # Scale the remaining range to [0, 1] or [-1, 0]
            sign = 1.0 if value > 0 else -1.0
            scaled = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
            return sign * scaled
    
    def _publish_cmd_vel(self, linear_x: float, angular_z: float):
        """Publish cmd_vel message"""
        twist_msg = Twist(linear_x=linear_x, angular_z=angular_z)
        cmd_vel_publisher.publish(twist_msg)
    
    def _publisher_loop(self):
        """Main publisher loop"""
        rate = 1.0 / self.publish_rate
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        
        while self.running:
            self._publish_cmd_vel(self.current_linear_vel, self.current_angular_vel)
            time.sleep(rate)


def list_joysticks():
    """List all available joysticks"""
    if not HAS_PYGAME:
        print("pygame not available")
        return
    
    pygame.init()
    pygame.joystick.init()
    
    joystick_count = pygame.joystick.get_count()
    print(f"Found {joystick_count} joystick(s):")
    
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        print(f"  {i}: {joystick.get_name()} ({joystick.get_numaxes()} axes, {joystick.get_numbuttons()} buttons)")
        joystick.quit()
    
    pygame.quit()


def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_joysticks()
        return
    
    if not HAS_PYGAME:
        print("Please install pygame: pip install pygame")
        return
    
    teleop = JoystickTeleop(
        max_linear_vel=1.5,
        max_angular_vel=3.0,
        deadzone=0.15
    )
    
    teleop.start()


if __name__ == "__main__":
    main()