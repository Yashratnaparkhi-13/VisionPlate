"""
VisionPlate Streamlit Web Dashboard Launcher
Runs the Streamlit web dashboard on local browser.
"""
import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("==================================================")
    print("   Starting VisionPlate Streamlit Web App...      ")
    print("==================================================")
    
    app_script = os.path.join(PROJECT_ROOT, "streamlit_app.py")
    
    # Run streamlit via python -m streamlit run streamlit_app.py
    cmd = [sys.executable, "-m", "streamlit", "run", app_script]
    subprocess.run(cmd)
