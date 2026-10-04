"""
VisionPlate Application Entry Point
"""
import sys
import os

# Add project root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from app.detector import PlateDetector
from app.ui.main_window import MainWindow


def main():
    """Application main function."""
    # Enable High DPI scaling for modern displays
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("VisionPlate")
    app.setOrganizationName("CapStoneDL")

    # Resolve model path safely relative to project root
    model_path = os.path.join(PROJECT_ROOT, "best.pt")

    # Initialize YOLO detector
    detector = PlateDetector(model_path=model_path)
    detector.load_model()

    # Create and display main window
    window = MainWindow(detector=detector)
    window.show()

    # Execute Qt event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
