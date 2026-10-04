"""
VisionPlate Main Window UI Module
Professional PySide6 Desktop GUI for Automatic Vehicle Number Plate Detection.
"""
import os
import cv2
import numpy as np
from typing import Optional, List, Dict, Any

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QFrame,
    QStatusBar, QStyle, QSizePolicy, QScrollArea
)
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPalette, QAction
from PySide6.QtCore import Qt, Slot, QSize

from app.detector import PlateDetector
from app.camera import CameraWorker
from app.utils.image_utils import cv2_to_qpixmap, save_crop_to_disk


class ScaledImageLabel(QLabel):
    """
    Custom QLabel that maintains image aspect ratio dynamically on resize.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: Optional[QPixmap] = None

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.update_display()

    def clear_image(self):
        self._pixmap = None
        self.clear()
        self.setText("No Image or Camera Stream Loaded\n\nClick 'Upload Image' or 'Start Camera' to begin.")

    def update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()


class MainWindow(QMainWindow):
    """
    Main Window for VisionPlate Application.
    """
    def __init__(self, detector: PlateDetector):
        super().__init__()
        self.detector = detector
        self.camera_worker: Optional[CameraWorker] = None
        self.current_image_path: Optional[str] = None
        self.current_detections: List[Dict[str, Any]] = []
        self.current_annotated_cv: Optional[np.ndarray] = None
        self.current_original_cv: Optional[np.ndarray] = None

        self.init_ui()
        self.verify_model_status()

    def init_ui(self):
        """Build window layout and styling."""
        self.setWindowTitle("VisionPlate — Automatic Vehicle Number Plate Detection")
        self.resize(1200, 800)
        self.setMinimumSize(950, 650)

        # Central Widget & Main Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # ----------------------------------------------------
        # 1. HEADER SECTION
        # ----------------------------------------------------
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 10, 12, 10)

        title_vbox = QVBoxLayout()
        title_label = QLabel("VisionPlate")
        title_font = QFont("Arial", 18, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setObjectName("titleLabel")

        subtitle_label = QLabel("Automatic Vehicle Number Plate Detection")
        sub_font = QFont("Arial", 10, QFont.Weight.Normal)
        subtitle_label.setFont(sub_font)
        subtitle_label.setObjectName("subtitleLabel")

        title_vbox.addWidget(title_label)
        title_vbox.addWidget(subtitle_label)

        # Badge Info
        self.badge_label = QLabel("Model: YOLO11n (best.pt) | Class: number_plate")
        self.badge_label.setObjectName("badgeLabel")
        self.badge_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        header_layout.addLayout(title_vbox)
        header_layout.addStretch()
        header_layout.addWidget(self.badge_label)

        main_layout.addWidget(header_frame)

        # ----------------------------------------------------
        # 2. CONTROL TOOLBAR
        # ----------------------------------------------------
        control_frame = QFrame()
        control_frame.setObjectName("controlFrame")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(12, 8, 12, 8)
        control_layout.setSpacing(10)

        # Action Buttons
        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.btn_upload.clicked.connect(self.on_upload_image)
        self.btn_upload.setObjectName("btnUpload")

        self.btn_start_cam = QPushButton("Start Camera")
        self.btn_start_cam.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_start_cam.clicked.connect(self.on_start_camera)
        self.btn_start_cam.setObjectName("btnStartCam")

        self.btn_stop_cam = QPushButton("Stop Camera")
        self.btn_stop_cam.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.btn_stop_cam.clicked.connect(self.on_stop_camera)
        self.btn_stop_cam.setEnabled(False)
        self.btn_stop_cam.setObjectName("btnStopCam")

        self.btn_clear = QPushButton("Clear / Reset")
        self.btn_clear.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogDiscardButton))
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_clear.setObjectName("btnClear")

        control_layout.addWidget(self.btn_upload)
        control_layout.addWidget(self.btn_start_cam)
        control_layout.addWidget(self.btn_stop_cam)
        control_layout.addWidget(self.btn_clear)

        # Separator Line
        v_sep = QFrame()
        v_sep.setFrameShape(QFrame.Shape.VLine)
        v_sep.setFrameShadow(QFrame.Shadow.Sunken)
        control_layout.addWidget(v_sep)

        # Confidence Threshold Control
        conf_lbl = QLabel("Confidence:")
        conf_lbl.setObjectName("confLabel")
        control_layout.addWidget(conf_lbl)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 100)
        self.conf_slider.setValue(25)  # Default 0.25 from notebook
        self.conf_slider.setFixedWidth(120)
        self.conf_slider.setToolTip("Set confidence threshold (0.05 to 1.00)")

        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.05, 1.00)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(0.25)
        self.conf_spinbox.setDecimals(2)
        self.conf_spinbox.setFixedWidth(65)

        # Sync Slider and SpinBox
        self.conf_slider.valueChanged.connect(self.on_slider_changed)
        self.conf_spinbox.valueChanged.connect(self.on_spinbox_changed)

        control_layout.addWidget(self.conf_slider)
        control_layout.addWidget(self.conf_spinbox)
        control_layout.addStretch()

        main_layout.addWidget(control_frame)

        # ----------------------------------------------------
        # 3. MAIN WORKSPACE (SPLITTER: VIEWER vs RESULTS)
        # ----------------------------------------------------
        workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        workspace_splitter.setChildrenCollapsible(False)

        # LEFT PANEL: IMAGE / CAMERA VIEW
        left_group = QGroupBox("IMAGE / CAMERA VIEW")
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(10, 14, 10, 10)

        self.view_status_lbl = QLabel("Status: Standby")
        self.view_status_lbl.setObjectName("viewSubLabel")
        left_layout.addWidget(self.view_status_lbl)

        self.image_display = ScaledImageLabel()
        self.image_display.clear_image()
        left_layout.addWidget(self.image_display)

        workspace_splitter.addWidget(left_group)

        # RIGHT PANEL: RESULTS & CROPPED PLATE
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Top Right: Detection Results List
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(10, 14, 10, 10)

        self.results_summary_lbl = QLabel("No detections yet.")
        self.results_summary_lbl.setObjectName("summaryLabel")
        results_layout.addWidget(self.results_summary_lbl)

        self.results_list = QListWidget()
        self.results_list.setObjectName("resultsList")
        self.results_list.itemSelectionChanged.connect(self.on_detection_selected)
        results_layout.addWidget(self.results_list)

        right_layout.addWidget(results_group, stretch=2)

        # Bottom Right: Cropped Plate Preview
        crop_group = QGroupBox("Detected Plate")
        crop_layout = QVBoxLayout(crop_group)
        crop_layout.setContentsMargins(10, 14, 10, 10)

        self.crop_display = QLabel()
        self.crop_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crop_display.setMinimumHeight(120)
        self.crop_display.setText("Cropped plate will be displayed here.")
        self.crop_display.setObjectName("cropDisplay")
        crop_layout.addWidget(self.crop_display)

        self.crop_info_lbl = QLabel("")
        self.crop_info_lbl.setObjectName("cropInfoLabel")
        self.crop_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_layout.addWidget(self.crop_info_lbl)

        right_layout.addWidget(crop_group, stretch=1)

        workspace_splitter.addWidget(right_panel)
        workspace_splitter.setSizes([700, 350])

        main_layout.addWidget(workspace_splitter, stretch=1)

        # ----------------------------------------------------
        # 4. STATUS BAR
        # ----------------------------------------------------
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Status: Ready")

        # Apply Modern Professional Stylesheet
        self.apply_stylesheet()

    def apply_stylesheet(self):
        """Apply clean modern academic/CV GUI styling."""
        style = """
        QMainWindow {
            background-color: #f4f6f9;
        }
        #headerFrame {
            background-color: #1e293b;
            border-radius: 6px;
        }
        #titleLabel {
            color: #ffffff;
        }
        #subtitleLabel {
            color: #94a3b8;
        }
        #badgeLabel {
            color: #38bdf8;
            font-size: 11px;
            font-weight: bold;
        }
        #controlFrame {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
        }
        QPushButton {
            padding: 6px 14px;
            font-weight: bold;
            border-radius: 4px;
            border: 1px solid #cbd5e1;
            background-color: #f8fafc;
            color: #334155;
        }
        QPushButton:hover {
            background-color: #e2e8f0;
        }
        #btnUpload {
            background-color: #0284c7;
            color: #ffffff;
            border: none;
        }
        #btnUpload:hover {
            background-color: #0369a1;
        }
        #btnStartCam {
            background-color: #16a34a;
            color: #ffffff;
            border: none;
        }
        #btnStartCam:hover {
            background-color: #15803d;
        }
        #btnStopCam {
            background-color: #dc2626;
            color: #ffffff;
            border: none;
        }
        #btnStopCam:hover {
            background-color: #b91c1c;
        }
        #btnStopCam:disabled {
            background-color: #f1f5f9;
            color: #94a3b8;
            border: 1px solid #e2e8f0;
        }
        #btnClear {
            background-color: #64748b;
            color: #ffffff;
            border: none;
        }
        #btnClear:hover {
            background-color: #475569;
        }
        #confLabel {
            font-weight: bold;
            color: #334155;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 12px;
            color: #1e293b;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            margin-top: 6px;
            background-color: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        #viewSubLabel {
            color: #64748b;
            font-size: 11px;
            margin-bottom: 4px;
        }
        ScaledImageLabel {
            background-color: #0f172a;
            border-radius: 4px;
            color: #94a3b8;
        }
        #resultsList {
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            font-size: 12px;
            background-color: #f8fafc;
        }
        #resultsList::item {
            padding: 8px;
            border-bottom: 1px solid #e2e8f0;
        }
        #resultsList::item:selected {
            background-color: #0284c7;
            color: #ffffff;
        }
        #cropDisplay {
            background-color: #0f172a;
            border-radius: 4px;
            color: #94a3b8;
            padding: 6px;
        }
        #cropInfoLabel {
            color: #16a34a;
            font-weight: bold;
            font-size: 11px;
            margin-top: 4px;
        }
        QStatusBar {
            background-color: #1e293b;
            color: #f8fafc;
            font-size: 11px;
        }
        """
        self.setStyleSheet(style)

    def verify_model_status(self):
        """Verify detector model status on launch."""
        if not self.detector.is_loaded:
            success, msg = self.detector.load_model()
            if not success:
                QMessageBox.warning(
                    self,
                    "Model Initialization Warning",
                    f"Warning: Model could not be loaded automatically.\n\nDetails: {msg}\n\nPlease verify 'best.pt' exists in project root."
                )
                self.statusBar.showMessage(f"Error: {msg}")
                return

        self.statusBar.showMessage("Status: Model 'best.pt' loaded successfully. Ready for inference.")

    # ----------------------------------------------------
    # CONFIDENCE CONTROLS
    # ----------------------------------------------------
    def get_confidence(self) -> float:
        return self.conf_spinbox.value()

    @Slot(int)
    def on_slider_changed(self, value: int):
        val_float = value / 100.0
        self.conf_spinbox.blockSignals(True)
        self.conf_spinbox.setValue(val_float)
        self.conf_spinbox.blockSignals(False)
        self.handle_confidence_update(val_float)

    @Slot(float)
    def on_spinbox_changed(self, value: float):
        slider_val = int(value * 100)
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(slider_val)
        self.conf_slider.blockSignals(False)
        self.handle_confidence_update(value)

    def handle_confidence_update(self, conf: float):
        # If camera running, update camera worker
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.set_confidence(conf)
            self.statusBar.showMessage(f"Camera confidence updated to: {conf:.2f}")
        # If static image loaded, re-run detection
        elif self.current_image_path and os.path.exists(self.current_image_path):
            self.run_image_detection(self.current_image_path)

    # ----------------------------------------------------
    # IMAGE UPLOAD
    # ----------------------------------------------------
    @Slot()
    def on_upload_image(self):
        if self.camera_worker and self.camera_worker.isRunning():
            self.on_stop_camera()

        file_filter = "Vehicle Images (*.jpg *.jpeg *.png *.webp);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vehicle Image for Detection",
            "",
            file_filter
        )

        if not file_path:
            return

        self.current_image_path = file_path
        self.run_image_detection(file_path)

    def run_image_detection(self, file_path: str):
        self.statusBar.showMessage("Processing image with YOLO best.pt...")
        conf = self.get_confidence()

        result = self.detector.detect(file_path, conf_threshold=conf)

        if not result["success"]:
            QMessageBox.critical(self, "Detection Error", f"Failed to process image:\n{result['error']}")
            self.statusBar.showMessage(f"Error: {result['error']}")
            return

        self.current_original_cv = result["original_image"]
        self.current_annotated_cv = result["annotated_image"]
        self.current_detections = result["detections"]

        # Display annotated image
        pixmap = cv2_to_qpixmap(self.current_annotated_cv)
        self.image_display.set_pixmap(pixmap)
        filename = os.path.basename(file_path)
        self.view_status_lbl.setText(f"File: {filename} | Detections: {len(self.current_detections)}")

        # Update Results List
        self.update_results_list()

        # Save crops to outputs/crops/
        saved_paths = []
        for det in self.current_detections:
            saved_p = save_crop_to_disk(det["crop"])
            saved_paths.append(saved_p)

        det_count = len(self.current_detections)
        if det_count > 0:
            msg = f"Detection Complete: {det_count} number plate(s) found. Saved to outputs/crops/"
        else:
            msg = "Detection Complete: No number plate detected at current confidence threshold."

        self.statusBar.showMessage(msg)

    # ----------------------------------------------------
    # WEBCAM HANDLING
    # ----------------------------------------------------
    @Slot()
    def on_start_camera(self):
        if not self.detector.is_loaded:
            success, msg = self.detector.load_model()
            if not success:
                QMessageBox.critical(self, "Model Error", f"Cannot start camera: {msg}")
                return

        self.statusBar.showMessage("Initializing camera feed...")
        self.btn_start_cam.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)

        conf = self.get_confidence()
        self.camera_worker = CameraWorker(self.detector, camera_index=0, conf_threshold=conf)
        self.camera_worker.frame_processed.connect(self.on_camera_frame)
        self.camera_worker.error_occurred.connect(self.on_camera_error)
        self.camera_worker.camera_stopped.connect(self.on_camera_stopped_event)
        self.camera_worker.start()

        self.view_status_lbl.setText("Mode: Live Camera Feed (YOLO Active)")

    @Slot()
    def on_stop_camera(self):
        if self.camera_worker and self.camera_worker.isRunning():
            self.statusBar.showMessage("Stopping camera feed...")
            self.camera_worker.stop()

    @Slot(object, list)
    def on_camera_frame(self, frame_cv: np.ndarray, detections: list):
        # Update viewer
        pixmap = cv2_to_qpixmap(frame_cv)
        self.image_display.set_pixmap(pixmap)

        self.current_annotated_cv = frame_cv
        self.current_detections = detections

        self.view_status_lbl.setText(f"Mode: Live Camera Feed | Detections: {len(detections)}")
        self.update_results_list()
        self.statusBar.showMessage(f"Camera Stream Active — {len(detections)} plate(s) detected")

    @Slot(str)
    def on_camera_error(self, err_msg: str):
        QMessageBox.warning(self, "Webcam Alert", err_msg)
        self.statusBar.showMessage(f"Webcam Error: {err_msg}")
        self.on_stop_camera()

    @Slot()
    def on_camera_stopped_event(self):
        self.btn_start_cam.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        self.statusBar.showMessage("Camera feed stopped.")

    # ----------------------------------------------------
    # RESULTS LIST & CROP DISPLAY
    # ----------------------------------------------------
    def update_results_list(self):
        self.results_list.clear()
        count = len(self.current_detections)
        self.results_summary_lbl.setText(f"Total Detected Plates: {count}")

        if count == 0:
            self.crop_display.clear()
            self.crop_display.setText("No cropped plate available.")
            self.crop_info_lbl.setText("")
            return

        for det in self.current_detections:
            idx = det["index"]
            cls_name = det["class_name"]
            conf_pct = det["confidence"] * 100.0
            x1, y1, x2, y2 = det["bbox"]

            item_text = (
                f"Plate #{idx} — {cls_name}\n"
                f"Confidence: {conf_pct:.1f}%\n"
                f"Box: ({x1}, {y1}, {x2}, {y2})"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, det)
            self.results_list.addItem(item)

        # Select first item by default
        self.results_list.setCurrentRow(0)

    @Slot()
    def on_detection_selected(self):
        selected_items = self.results_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        det = item.data(Qt.ItemDataRole.UserRole)
        if not det:
            return

        crop_cv = det["crop"]
        if crop_cv is not None and crop_cv.size > 0:
            pixmap = cv2_to_qpixmap(crop_cv)
            # Scale preview
            scaled_crop = pixmap.scaled(
                QSize(280, 140),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.crop_display.setPixmap(scaled_crop)
            conf_pct = det["confidence"] * 100.0
            self.crop_info_lbl.setText(f"Detected: {det['class_name']} ({conf_pct:.1f}%)")

    # ----------------------------------------------------
    # CLEAR / RESET
    # ----------------------------------------------------
    @Slot()
    def on_clear(self):
        if self.camera_worker and self.camera_worker.isRunning():
            self.on_stop_camera()

        self.current_image_path = None
        self.current_detections = []
        self.current_annotated_cv = None
        self.current_original_cv = None

        self.image_display.clear_image()
        self.results_list.clear()
        self.results_summary_lbl.setText("No detections yet.")
        self.crop_display.clear()
        self.crop_display.setText("Cropped plate will be displayed here.")
        self.crop_info_lbl.setText("")
        self.view_status_lbl.setText("Status: Standby")

        self.statusBar.showMessage("Application reset to initial state.")

    def closeEvent(self, event):
        """Ensure clean thread cleanup on app exit."""
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.camera_worker.wait(1000)
        event.accept()
