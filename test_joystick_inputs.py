#!/usr/bin/env python3
"""
Joystick Input Testing Script
This script will help us analyze joystick inputs to debug the control issue.
"""

import sys
import time
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    print("Error: pygame not installed. Install with: pip install pygame")
    sys.exit(1)

def test_joystick_inputs():
    """Test and display all joystick inputs in real-time"""
    
    # Initialize pygame
    pygame.init()
    pygame.joystick.init()
    
    # Check for joysticks
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No joystick detected!")
        return
    
    print(f"Found {joystick_count} joystick(s)")
    
    # Initialize first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"Connected to: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}")
    print(f"Buttons: {joystick.get_numbuttons()}")
    print(f"Hats: {joystick.get_numhats()}")
    print()
    
    # Display controls
    print("=== JOYSTICK INPUT TEST ===")
    print("Press buttons and move sticks to see values")
    print("Press Ctrl+C to quit")
    print("=" * 50)
    
    clock = pygame.time.Clock()
    running = True
    
    try:
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"BUTTON {event.button} PRESSED")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"BUTTON {event.button} RELEASED")
                elif event.type == pygame.JOYAXISMOTION:
                    if abs(event.value) > 0.1:  # Only show significant movement
                        print(f"AXIS {event.axis}: {event.value:.3f}")
                elif event.type == pygame.JOYHATMOTION:
                    if event.value != (0, 0):  # Only show non-zero hat values
                        print(f"HAT {event.hat}: {event.value}")
            
            # Display current state every 0.5 seconds
            current_time = time.time()
            if hasattr(test_joystick_inputs, 'last_display_time'):
                if current_time - test_joystick_inputs.last_display_time < 0.5:
                    clock.tick(60)
                    continue
            test_joystick_inputs.last_display_time = current_time
            
            # Clear screen (simple approach)
            print("\n" + "="*60)
            print(f"TIME: {time.strftime('%H:%M:%S')}")
            print("="*60)
            
            # Display all axis values
            print("AXES:")
            for i in range(joystick.get_numaxes()):
                value = joystick.get_axis(i)
                # Color coding for significant values
                if abs(value) > 0.1:
                    print(f"  Axis {i:2d}: {value:7.3f} {'***' if abs(value) > 0.5 else '**' if abs(value) > 0.2 else '*'}")
                else:
                    print(f"  Axis {i:2d}: {value:7.3f}")
            
            # Display all button states
            print("\nBUTTONS:")
            button_states = []
            for i in range(joystick.get_numbuttons()):
                pressed = joystick.get_button(i)
                button_states.append("1" if pressed else "0")
                if pressed:
                    print(f"  Button {i:2d}: PRESSED")
            
            if not any(joystick.get_button(i) for i in range(joystick.get_numbuttons())):
                print("  All buttons: RELEASED")
            
            # Display hat states
            if joystick.get_numhats() > 0:
                print("\nHATS:")
                for i in range(joystick.get_numhats()):
                    hat_value = joystick.get_hat(i)
                    if hat_value != (0, 0):
                        print(f"  Hat {i}: {hat_value}")
                    else:
                        print(f"  Hat {i}: (0, 0)")
            
            # Test the specific mapping we're using
            print("\n=== CURRENT MAPPING TEST ===")
            axis_linear = 1   # Left stick vertical
            axis_angular = 0  # Left stick horizontal
            
            raw_linear = -joystick.get_axis(axis_linear)
            raw_angular = -joystick.get_axis(axis_angular)
            
            print(f"Raw Linear (axis {axis_linear}): {raw_linear:.3f}")
            print(f"Raw Angular (axis {axis_angular}): {raw_angular:.3f}")
            
            # Apply deadzone
            deadzone = 0.12
            def apply_deadzone(value):
                if abs(value) < deadzone:
                    return 0.0
                else:
                    sign = 1.0 if value > 0 else -1.0
                    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
                    return sign * scaled
            
            linear_vel = apply_deadzone(raw_linear) * 1.5  # max_linear_vel
            angular_vel = apply_deadzone(raw_angular) * 3.0  # max_angular_vel
            
            print(f"After deadzone Linear: {linear_vel:.3f}")
            print(f"After deadzone Angular: {angular_vel:.3f}")
            
            # Test button mapping
            print("\n=== BUTTON MAPPING TEST ===")
            button_forward = 0   # A button
            button_backward = 1  # B button  
            button_left = 2      # X button
            button_right = 3     # Y button
            button_speed = 4     # Left bumper
            
            buttons = {
                "Forward (A)": joystick.get_button(button_forward),
                "Backward (B)": joystick.get_button(button_backward),
                "Left (X)": joystick.get_button(button_left),
                "Right (Y)": joystick.get_button(button_right),
                "Speed (LB)": joystick.get_button(button_speed)
            }
            
            for name, pressed in buttons.items():
                status = "PRESSED" if pressed else "released"
                print(f"  {name}: {status}")
            
            # Calculate final commands
            button_linear = 0.0
            button_angular = 0.0
            speed_multiplier = 2.0 if joystick.get_button(button_speed) else 1.0
            
            if joystick.get_button(button_forward):
                button_linear = 1.5 * speed_multiplier
            elif joystick.get_button(button_backward):
                button_linear = -1.5 * speed_multiplier
            
            if joystick.get_button(button_left):
                button_angular = 3.0 * speed_multiplier
            elif joystick.get_button(button_right):
                button_angular = -3.0 * speed_multiplier
            
            print(f"\nButton Linear: {button_linear:.3f}")
            print(f"Button Angular: {button_angular:.3f}")
            
            # Final commands (what would be sent to robot)
            if (joystick.get_button(button_forward) or 
                joystick.get_button(button_backward) or
                joystick.get_button(button_left) or 
                joystick.get_button(button_right)):
                final_linear = button_linear
                final_angular = button_angular
                print(f"\n*** USING BUTTON INPUT ***")
            else:
                final_linear = linear_vel
                final_angular = angular_vel
                print(f"\n*** USING AXIS INPUT ***")
            
            print(f"FINAL Linear: {final_linear:.3f}")
            print(f"FINAL Angular: {final_angular:.3f}")
            
            clock.tick(60)
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        joystick.quit()
        pygame.quit()

if __name__ == "__main__":
    test_joystick_inputs()
