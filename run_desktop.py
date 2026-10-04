"""
VisionPlate Desktop GUI Launcher
Runs the PySide6 desktop application directly.
"""
import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("==================================================")
    print("   Starting VisionPlate Desktop Application...    ")
    print("==================================================")
    
    # Path to main.py
    main_script = os.path.join(PROJECT_ROOT, "app", "main.py")
    
    # Use current python executable
    cmd = [sys.executable, main_script]
    subprocess.run(cmd)
