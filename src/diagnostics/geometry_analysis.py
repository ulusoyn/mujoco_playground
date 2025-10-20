"""
Robot Geometry Analysis and Mismatch Identification
This document identifies the mismatches between controller parameters and actual robot geometry.
"""

import numpy as np

def analyze_robot_geometry():
    """
    Analyze the actual robot geometry from the XML file and compare with controller parameters.
    """
    
    print("="*60)
    print("ROBOT GEOMETRY ANALYSIS")
    print("="*60)
    
    # Actual geometry from XML file
    print("\nACTUAL ROBOT GEOMETRY (from XML):")
    print("-" * 40)
    
    # Wheel positions from XML
    rear_left_pos = [-0.10, 0.087, -0.0325]
    rear_right_pos = [-0.10, -0.087, -0.0325]
    front_left_pos = [0.10, 0.087, -0.0325]
    front_right_pos = [0.10, -0.087, -0.0325]
    
    print(f"Rear Left Wheel:  {rear_left_pos}")
    print(f"Rear Right Wheel: {rear_right_pos}")
    print(f"Front Left Wheel: {front_left_pos}")
    print(f"Front Right Wheel:{front_right_pos}")
    
    # Calculate actual dimensions
    wheelbase_actual = front_left_pos[0] - rear_left_pos[0]  # X distance
    track_width_actual = rear_left_pos[1] - rear_right_pos[1]  # Y distance
    wheel_radius_actual = 0.0325  # From geom size
    
    print(f"\nCALCULATED DIMENSIONS:")
    print(f"Wheelbase (X):     {wheelbase_actual:.3f} m")
    print(f"Track Width (Y):   {track_width_actual:.3f} m")
    print(f"Wheel Radius:      {wheel_radius_actual:.3f} m")
    
    # Controller parameters
    print(f"\nCONTROLLER PARAMETERS:")
    print("-" * 40)
    wheelbase_controller = 0.20
    track_width_controller = 0.174
    wheel_radius_controller = 0.0325
    
    print(f"Wheelbase:         {wheelbase_controller:.3f} m")
    print(f"Track Width:       {track_width_controller:.3f} m")
    print(f"Wheel Radius:      {wheel_radius_controller:.3f} m")
    
    # Calculate mismatches
    print(f"\nMISMATCHES:")
    print("-" * 40)
    
    wheelbase_error = abs(wheelbase_actual - wheelbase_controller)
    track_width_error = abs(track_width_actual - track_width_controller)
    wheel_radius_error = abs(wheel_radius_actual - wheel_radius_controller)
    
    print(f"Wheelbase Error:    {wheelbase_error:.3f} m ({wheelbase_error*100:.1f} cm)")
    print(f"Track Width Error:  {track_width_error:.3f} m ({track_width_error*100:.1f} cm)")
    print(f"Wheel Radius Error: {wheel_radius_error:.3f} m ({wheel_radius_error*100:.1f} cm)")
    
    # Calculate percentage errors
    wheelbase_pct_error = (wheelbase_error / wheelbase_actual) * 100
    track_width_pct_error = (track_width_error / track_width_actual) * 100
    
    print(f"\nPERCENTAGE ERRORS:")
    print(f"Wheelbase:          {wheelbase_pct_error:.1f}%")
    print(f"Track Width:        {track_width_pct_error:.1f}%")
    
    # Impact analysis
    print(f"\nIMPACT ANALYSIS:")
    print("-" * 40)
    
    if wheelbase_pct_error > 5:
        print(f"⚠️  HIGH IMPACT: Wheelbase error ({wheelbase_pct_error:.1f}%) affects:")
        print(f"   - Steering angle calculations")
        print(f"   - Turn radius calculations")
        print(f"   - Robot turning behavior")
    else:
        print(f"✅ LOW IMPACT: Wheelbase error ({wheelbase_pct_error:.1f}%) is acceptable")
    
    if track_width_pct_error > 5:
        print(f"⚠️  HIGH IMPACT: Track width error ({track_width_pct_error:.1f}%) affects:")
        print(f"   - Differential wheel speed calculations")
        print(f"   - Turning smoothness")
        print(f"   - Ackermann geometry")
    else:
        print(f"✅ LOW IMPACT: Track width error ({track_width_pct_error:.1f}%) is acceptable")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print("-" * 40)
    print(f"1. Update controller wheelbase: {wheelbase_actual:.3f} m")
    print(f"2. Update controller track width: {track_width_actual:.3f} m")
    print(f"3. Wheel radius is correct: {wheel_radius_actual:.3f} m")
    
    return {
        'actual': {
            'wheelbase': wheelbase_actual,
            'track_width': track_width_actual,
            'wheel_radius': wheel_radius_actual
        },
        'controller': {
            'wheelbase': wheelbase_controller,
            'track_width': track_width_controller,
            'wheel_radius': wheel_radius_controller
        },
        'errors': {
            'wheelbase': wheelbase_error,
            'track_width': track_width_error,
            'wheel_radius': wheel_radius_error
        }
    }

def calculate_corrected_parameters():
    """Calculate the corrected controller parameters."""
    
    print(f"\n" + "="*60)
    print("CORRECTED CONTROLLER PARAMETERS")
    print("="*60)
    
    # Corrected parameters
    wheelbase_corrected = 0.20  # 0.10 - (-0.10) = 0.20
    track_width_corrected = 0.174  # 0.087 - (-0.087) = 0.174
    wheel_radius_corrected = 0.0325
    
    print(f"Corrected Wheelbase:   {wheelbase_corrected:.3f} m")
    print(f"Corrected Track Width: {track_width_corrected:.3f} m")
    print(f"Corrected Wheel Radius:{wheel_radius_corrected:.3f} m")
    
    print(f"\nCONTROLLER CODE UPDATE:")
    print("-" * 40)
    print(f"def __init__(self, model, data,")
    print(f"             wheel_radius={wheel_radius_corrected:.3f}, wheelbase={wheelbase_corrected:.3f}, track_width={track_width_corrected:.3f}):")
    
    return {
        'wheelbase': wheelbase_corrected,
        'track_width': track_width_corrected,
        'wheel_radius': wheel_radius_corrected
    }

if __name__ == "__main__":
    # Run analysis
    geometry_data = analyze_robot_geometry()
    corrected_params = calculate_corrected_parameters()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"The controller parameters are actually CORRECT!")
    print(f"Wheelbase: 0.20m (matches actual geometry)")
    print(f"Track Width: 0.174m (matches actual geometry)")
    print(f"Wheel Radius: 0.0325m (matches actual geometry)")
    print(f"\nThe 'weird movements' were likely due to:")
    print(f"1. Individual steering actuators instead of single servo")
    print(f"2. Incorrect Ackermann implementation")
    print(f"3. Controller-actuator name mismatches")
    print(f"\nNow using proper bicycle model with single steering servo! ✅")
