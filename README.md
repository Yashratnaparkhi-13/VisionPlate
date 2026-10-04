# VisionPlate — Automatic Vehicle Number Plate Detection

**VisionPlate** is a professional desktop application designed for real-time automatic vehicle number plate detection and cropping using a custom-trained **YOLO11** object detection checkpoint.

---

## 1. Project Overview
VisionPlate serves as Phase 1 of a Deep Learning capstone project focused on intelligent vehicle identification. The desktop GUI provides an intuitive dashboard for loading vehicle images or running live webcam video feeds to detect number plates, draw bounding boxes, report model confidences, and automatically crop and save detected license plates.

---

## 2. Current Functionality
- **Static Image Analysis**: Upload images (`.jpg`, `.jpeg`, `.png`, `.webp`), detect vehicle number plates, overlay bounding boxes with confidence scores, and display results.
- **Cropped Plate Extraction**: Automatically crop detected plate regions and display them in a dedicated inspection panel.
- **Automatic Crop Storage**: Save cropped license plates to disk at `outputs/crops/plate_{timestamp}.jpg` without overwriting prior outputs.
- **Live Webcam Detection**: Real-time video stream detection using system webcams with smooth multi-threading (`QThread`) to prevent GUI freezing.
- **Interactive Confidence Adjustment**: Dynamic slider and spinbox control to tune the detection threshold (`0.05` to `1.00`, default `0.25`).
- **Clear & Reset**: Reset interface state, clear active feeds, and return the dashboard to standby mode.

---

## 3. Environment Requirements
- **Operating System**: Windows 10/11 or macOS / Linux
- **Python Version**: Python 3.9+ (Python 3.10/3.11/3.13 tested)
- **Key Dependencies**:
  - `ultralytics` (YOLO engine)
  - `torch` & `torchvision` (PyTorch backend)
  - `PySide6` (Desktop Qt GUI)
  - `opencv-python` (Image/Video handling)
  - `numpy` & `pillow`

---

## 4. Installation Steps

### Option A: Standard Setup (Windows / macOS)
1. Open a terminal / command prompt in the project root folder:
   ```bash
   cd "DL Project"
   ```
2. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - **Windows**:
     ```cmd
     .venv\Scripts\activate
     ```
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```
4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. How to Run

### Windows Quick Run:
Simply double-click `run.bat` or run in command prompt:
```cmd
run.bat
```

### Manual Execution:
With the virtual environment active, launch the application:
```bash
python app/main.py
```

---

## 6. How to Use Image Upload
1. Launch VisionPlate GUI.
2. Click the **Upload Image** button in the top toolbar.
3. Select any supported image file (`.jpg`, `.jpeg`, `.png`, `.webp`).
4. The system will run `best.pt` YOLO detection.
5. The processed image will render with bounding boxes and confidence scores.
6. Check the **Detection Results** panel for detailed metrics (`x1, y1, x2, y2` coordinates and confidence percentage).
7. Select any detection in the list to view its high-resolution cropped plate preview in the **Detected Plate** box.
8. Cropped plates are automatically exported to `outputs/crops/`.

---

## 7. How to Use Webcam
1. Click the **Start Camera** button.
2. The live camera stream will initialize without blocking or freezing the application UI.
3. The active detector continuously scans frames for number plates and overlays bounding boxes in real time.
4. Adjust the **Confidence Threshold** slider anytime during stream playback to adjust sensitivity.
5. Click **Stop Camera** to release camera resources and return to standby mode.

---

## 8. Project Structure
```
DL Project/
│
├── best.pt                   # Pre-trained YOLO11 number plate detector checkpoint
├── VisionPlate (1).ipynb     # Notebook documenting CCPD training & validation
│
├── app/                      # Application source code
│   ├── __init__.py           # Package marker
│   ├── main.py               # Main entry point & Qt app launcher
│   ├── detector.py           # YOLO inference module (PlateDetector)
│   ├── camera.py             # Multi-threaded webcam worker (CameraWorker QThread)
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main_window.py    # PySide6 desktop GUI window & dashboard layout
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py    # BGR/QPixmap converters & crop saving utilities
│
├── outputs/
│   └── crops/                # Output folder for cropped plate images
│
├── requirements.txt          # Python dependencies
├── README.md                 # Complete project documentation
└── run.bat                   # Windows one-click batch launcher
```

---

## 9. Model Information
- **Model Architecture**: Ultralytics YOLO11 Nano (`yolo11n`)
- **Task Type**: Object Detection (`detect`)
- **Target Class**: `0: number_plate`
- **Training Dataset**: Fine-tuned on CCPD (Chinese City Parking Dataset) bounding box annotations.
- **Training Resolution**: $640 \times 640$ pixels
- **Training Specs**: 10 epochs, batch size 16, default confidence threshold 0.25.
- **Checkpoint File**: `best.pt`

---

## 10. Current Limitations
- **Detection Only**: This phase performs bounding box detection and cropping only; character optical recognition (OCR) is not included in Phase 1.
- **Camera Selection**: Default webcam index is set to `0`. If multiple cameras are attached, configuration is handled via code.

---

## 11. Future OCR Integration Plan
The architecture is designed to seamlessly accept OCR in Phase 2:
```
Vehicle Image / Camera Stream
            ↓
YOLO Number Plate Detection (best.pt)
            ↓
Bounding Box & Cropped Plate Image
            ↓
OpenCV Image Preprocessing (Binarization / Denoising / Deskewing)
            ↓
OCR Engine Integration (EasyOCR / Tesseract / PaddleOCR)
            ↓
Recognized Number Plate Text Display & Verification
```
The modular separation in `app/detector.py` and `app/ui/main_window.py` allows direct injection of an `OCRProcessor` pipeline on `det["crop"]` without restructuring the GUI layout.
