"""
Joystick teleoperation node that publishes cmd_vel messages
"""
import threading
import time
import sys
from pathlib import Path

# Add project root and src to path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
_src_dir = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from core.cmd_vel_message import Twist, cmd_vel_publisher

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
        
        # Axis mapping for Xbox One controller
        self.axis_left_x = 0      # Left stick horizontal (left/right)
        self.axis_left_y = 1      # Left stick vertical (forward/backward)
        self.axis_right_x = 2     # Right stick horizontal (rotation left/right)
        self.axis_right_y = 3     # Right stick vertical (not used)
        
        # Button configuration for Xbox One controller
        # Based on our test: A=0, B=1, X=2, Y=3, LB=4, RB=5, Back=6, Start=7, etc.
        self.button_safety = 5    # RB button - must be pressed to allow movement  
        self.button_emergency = 6 # Back button - emergency stop
        
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
        print(f"  Left stick (axis {self.axis_left_x},{self.axis_left_y}): Linear movement (forward/back, left/right)")
        print(f"  Right stick (axis {self.axis_right_x}): Rotation (left/right)")
        print("  Safety:")
        print(f"    RB (button {self.button_safety}): HOLD to enable movement")
        print(f"    Back (button {self.button_emergency}): Emergency stop")
        print("  IMPORTANT: Robot only moves when RB is pressed!")
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
        safety_enabled = False
        
        print("Joystick ready. Press Ctrl+C to quit.")
        print("Hold RB to enable movement, use Back button for emergency stop.")
        print("Starting joystick input loop...")
        
        while self.running:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == self.button_safety:
                        safety_enabled = True
                        print("RB pressed - Movement ENABLED")
                    elif event.button == self.button_emergency:
                        emergency_stop = True
                        print("Back button pressed - Emergency stop activated!")
                    else:
                        print(f"Button {event.button} pressed")
                elif event.type == pygame.JOYBUTTONUP:
                    if event.button == self.button_safety:
                        safety_enabled = False
                        print("RB released - Movement DISABLED")
                    elif event.button == self.button_emergency:
                        emergency_stop = False
                        print("Back button released - Emergency stop deactivated")
                    else:
                        print(f"Button {event.button} released")
                elif event.type == pygame.QUIT:
                    self.running = False
            
            # Read joystick input
            if not emergency_stop and self.joystick:
                try:
                    # Get axis values (-1.0 to 1.0)
                    # Left stick: linear movement
                    raw_left_x = self.joystick.get_axis(self.axis_left_x)   # Left/right
                    raw_left_y = -self.joystick.get_axis(self.axis_left_y)  # Forward/back (inverted)
                    
                    # Right stick: rotation
                    raw_right_x = self.joystick.get_axis(self.axis_right_x)  # Left/right rotation
                    
                    # Apply deadzone
                    left_x = self._apply_deadzone(raw_left_x)
                    left_y = self._apply_deadzone(raw_left_y)
                    right_x = self._apply_deadzone(raw_right_x)
                    
                    # Convert to robot commands
                    # For Ackermann robot, we use:
                    # - linear_x: forward/backward speed
                    # - angular_z: rotation speed
                    
                    linear_x = left_y * self.max_linear_vel  # Forward/backward
                    angular_z = right_x * self.max_angular_vel  # Rotation
                    
                    # Apply safety button - only move if R1 is pressed
                    if not safety_enabled:
                        linear_x = 0.0
                        angular_z = 0.0
                    
                    # Store current velocities for publisher thread
                    self.current_linear_vel = linear_x
                    self.current_angular_vel = angular_z
                    
                    # Debug output - show raw values and processed values
                    # Show all input, even small values
                    if abs(raw_left_y) > 0.001 or abs(raw_right_x) > 0.001:
                        print(f"Raw: left_y={raw_left_y:.3f}, right_x={raw_right_x:.3f}")
                        print(f"Processed: linear={linear_x:.3f}, angular={angular_z:.3f}, safety={safety_enabled}")
                    
                    # Also show if safety button is pressed
                    if safety_enabled and (abs(raw_left_y) > 0.001 or abs(raw_right_x) > 0.001):
                        print(f"SAFETY ENABLED: RB pressed, processing input")
                    
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
        # Debug: show what we're publishing
        if abs(linear_x) > 0.01 or abs(angular_z) > 0.01:
            print(f"PUBLISHING: linear={linear_x:.3f}, angular={angular_z:.3f}")
    
    def _publisher_loop(self):
        """Main publisher loop"""
        rate = 1.0 / self.publish_rate
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        
        print(f"Publisher loop started, rate: {self.publish_rate} Hz")
        
        while self.running:
            # Debug: show what we're about to publish
            if abs(self.current_linear_vel) > 0.01 or abs(self.current_angular_vel) > 0.01:
                print(f"Publisher loop: linear={self.current_linear_vel:.3f}, angular={self.current_angular_vel:.3f}")
            
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