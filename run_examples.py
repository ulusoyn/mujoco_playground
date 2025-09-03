#!/usr/bin/env python3
"""
MuJoCo Playground - Main Entry Point
Lists all available examples and provides easy access to run them.
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).resolve().parent / "utils"))
from path_helpers import list_available_models, list_available_scripts


def print_banner():
    """Print the project banner."""
    print("=" * 60)
    print("🤖 MuJoCo Playground - Robot Simulation Examples")
    print("=" * 60)
    print()


def print_available_models():
    """Print all available robot models."""
    models = list_available_models()
    if models:
        print("📁 Available Robot Models:")
        for model in models:
            print(f"  • {model}")
        print()
    else:
        print("❌ No robot models found in models/ directory")
        print()


def print_available_scripts():
    """Print all available scripts organized by category."""
    scripts = list_available_scripts()
    if scripts:
        print("📜 Available Scripts:")
        for category, script_list in scripts.items():
            print(f"  📂 {category.title()}:")
            for script in script_list:
                print(f"    • {script}")
        print()
    else:
        print("❌ No scripts found in scripts/ directory")
        print()


def print_usage_instructions():
    """Print instructions on how to run examples."""
    print("🚀 How to Run Examples:")
    print("  1. Navigate to the appropriate scripts directory")
    print("  2. Run the Python script directly")
    print()
    print("Examples:")
    print("  cd scripts/basic")
    print("  python view_ackermann.py")
    print()
    print("  cd scripts/interactive")
    print("  python interactive_ackermann.py")
    print()
    print("  cd scripts/examples")
    print("  python view_ackermann_controlled.py")
    print()


def main():
    """Main function to display available examples."""
    print_banner()
    print_available_models()
    print_available_scripts()
    print_usage_instructions()
    
    print("💡 Tip: Use the interactive scripts for full keyboard control!")
    print("   ESC to quit, A/D to steer, W/S to drive, R to reset")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()




