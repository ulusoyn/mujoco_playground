#!/usr/bin/env python3
"""
Debug script to test joystick integration step by step
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
    from core.cmd_vel_message import cmd_vel_publisher, Twist
    from teleop.joystick_teleop import JoystickTeleop
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_cmd_vel_system():
    """Test if cmd_vel system works"""
    print("\n=== Testing cmd_vel System ===")
    
    received_messages = []
    
    def test_callback(msg):
        received_messages.append((msg.linear.x, msg.angular.z))
        print(f"  Received: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")
    
    cmd_vel_publisher.subscribe(test_callback)
    
    # Send test messages
    test_messages = [
        Twist(linear_x=0.5, angular_z=0.0),
        Twist(linear_x=0.0, angular_z=0.3),
        Twist(linear_x=0.2, angular_z=0.1),
    ]
    
    for i, msg in enumerate(test_messages):
        print(f"  Sending test message {i+1}: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")
        cmd_vel_publisher.publish(msg)
        time.sleep(0.1)
    
    print(f"✓ cmd_vel system working: received {len(received_messages)} messages")
    return len(received_messages) > 0

def test_joystick_teleop():
    """Test joystick teleop in isolation"""
    print("\n=== Testing Joystick Teleop ===")
    
    try:
        teleop = JoystickTeleop(max_linear_vel=1.5, max_angular_vel=3.0, deadzone=0.12)
        print("✓ JoystickTeleop created successfully")
        
        # Check if joystick is detected
        import pygame
        pygame.init()
        pygame.joystick.init()
        
        joystick_count = pygame.joystick.get_count()
        print(f"  Joysticks detected: {joystick_count}")
        
        if joystick_count == 0:
            print("✗ No joystick detected!")
            return False
        
        # Initialize joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"✓ Connected to: {joystick.get_name()}")
        print(f"  Axes: {joystick.get_numaxes()}")
        print(f"  Buttons: {joystick.get_numbuttons()}")
        
        pygame.quit()
        return True
        
    except Exception as e:
        print(f"✗ Joystick teleop error: {e}")
        return False

def test_joystick_publishing():
    """Test if joystick teleop publishes messages"""
    print("\n=== Testing Joystick Publishing ===")
    
    received_messages = []
    
    def test_callback(msg):
        received_messages.append((msg.linear.x, msg.angular.z))
        print(f"  Received from joystick: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")
    
    cmd_vel_publisher.subscribe(test_callback)
    
    try:
        teleop = JoystickTeleop(max_linear_vel=1.5, max_angular_vel=3.0, deadzone=0.12)
        
        # Start teleop in a thread
        import threading
        teleop_thread = threading.Thread(target=teleop.start, daemon=True)
        teleop_thread.start()
        
        print("  Joystick teleop started in thread")
        print("  Press RB button and move joysticks for 5 seconds...")
        
        # Wait and collect messages
        start_time = time.time()
        while time.time() - start_time < 5.0:
            time.sleep(0.1)
        
        # Stop teleop
        teleop.stop()
        
        print(f"✓ Test completed: received {len(received_messages)} messages")
        return len(received_messages) > 0
        
    except Exception as e:
        print(f"✗ Publishing test error: {e}")
        return False

def main():
    print("Joystick Integration Debug Tool")
    print("=" * 40)
    
    # Test 1: cmd_vel system
    cmd_vel_ok = test_cmd_vel_system()
    
    # Test 2: Joystick detection
    joystick_ok = test_joystick_teleop()
    
    # Test 3: Joystick publishing (only if joystick is available)
    publishing_ok = False
    if joystick_ok:
        publishing_ok = test_joystick_publishing()
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"cmd_vel system: {'✓' if cmd_vel_ok else '✗'}")
    print(f"Joystick detection: {'✓' if joystick_ok else '✗'}")
    print(f"Joystick publishing: {'✓' if publishing_ok else '✗'}")
    
    if cmd_vel_ok and joystick_ok and publishing_ok:
        print("\n✓ All tests passed! The issue might be in the integration.")
    else:
        print("\n✗ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main()
