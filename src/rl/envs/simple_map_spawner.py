"""
Simplified Map Spawner for RL - Loads robot model with optional map background
"""
import mujoco
import numpy as np
from pathlib import Path
import sys
import random


class SimpleMapSpawner:
    """Simplified map spawner for RL that loads robot model directly"""
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.robot_model_path = self.project_root / "models" / "ackermann_robot_v2.xml"
    
    def load_random_environment(self, robot_pos=None, robot_quat=None):
        """
        Load robot model (simple version - no map merging)
        
        Args:
            robot_pos: [x, y, z] position for robot
            robot_quat: [w, x, y, z] quaternion for robot orientation
        
        Returns:
            tuple: (model, data, map_name)
        """
        if robot_pos is None:
            robot_pos = [0, 0, 0.1]
        
        # Load robot model directly
        model = mujoco.MjModel.from_xml_path(str(self.robot_model_path))
        data = mujoco.MjData(model)
        
        # Set robot position/orientation for freejoint
        # Freejoint has 7 DOF: [x, y, z, quat_w, quat_x, quat_y, quat_z]
        if robot_pos is not None:
            data.qpos[0:3] = robot_pos
        else:
            data.qpos[0:3] = [0, 0, 0.1]
        
        if robot_quat is not None:
            data.qpos[3:7] = robot_quat
        else:
            data.qpos[3:7] = [1, 0, 0, 0]  # Default: no rotation
        
        mujoco.mj_forward(model, data)
        
        return model, data, "simple_floor"

