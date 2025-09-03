"""
Path helper utilities for MuJoCo playground project.
Handles the directory structure: project_root/models/, project_root/scripts/, etc.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory (where models/ and scripts/ folders are located)."""
    # This file is in utils/, so go up one level to project root
    return Path(__file__).resolve().parent.parent


def get_model_path(model_name: str) -> Path:
    """Get the full path to a robot model file.
    
    Args:
        model_name: Name of the model file (e.g., "ackermann_robot" or "ackermann_robot.xml")
    
    Returns:
        Path to the model file in the models/ directory
    """
    project_root = get_project_root()
    
    # Add .xml extension if not provided
    if not model_name.endswith('.xml'):
        model_name += '.xml'
    
    return project_root / "models" / model_name


def get_script_path(script_name: str) -> Path:
    """Get the full path to a script file.
    
    Args:
        script_name: Name of the script file (e.g., "view_ackermann.py")
    
    Returns:
        Path to the script file in the scripts/ directory
    """
    project_root = get_project_root()
    
    # Add .py extension if not provided
    if not script_name.endswith('.py'):
        script_name += '.py'
    
    return project_root / "scripts" / script_name


def list_available_models() -> list[str]:
    """List all available robot models in the models/ directory.
    
    Returns:
        List of model names (without .xml extension)
    """
    project_root = get_project_root()
    models_dir = project_root / "models"
    
    if not models_dir.exists():
        return []
    
    return [f.stem for f in models_dir.glob("*.xml")]


def list_available_scripts() -> dict[str, list[str]]:
    """List all available scripts organized by category.
    
    Returns:
        Dictionary mapping category names to lists of script names
    """
    project_root = get_project_root()
    scripts_dir = project_root / "scripts"
    
    if not scripts_dir.exists():
        return {}
    
    categories = {}
    for category_dir in scripts_dir.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            scripts = [f.stem for f in category_dir.glob("*.py")]
            categories[category_name] = scripts
    
    return categories

