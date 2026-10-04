"""
VisionPlate — Streamlit Web Application
Automatic Vehicle Number Plate Detection Dashboard.
Supports both Client Browser Camera (Cloud Deployment Compatible) and Local Live Stream.
"""
import os
import sys
import time
import cv2
import numpy as np
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.detector import PlateDetector
from app.utils.image_utils import save_crop_to_disk

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="VisionPlate — Automatic Vehicle Number Plate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching PySide6 desktop dark slate / clean academic theme
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* 1. Header Frame */
    .header-frame {
        background-color: #1e293b;
        padding: 16px 24px;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-title-box {
        display: flex;
        flex-direction: column;
    }
    .header-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 2px;
    }
    .badge-label {
        background-color: #0f172a;
        color: #38bdf8;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #334155;
    }

    /* Group Box Styling */
    .group-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .group-title {
        font-size: 14px;
        font-weight: 700;
        color: #1e293b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
    }

    /* Sub Label */
    .view-sublabel {
        color: #64748b;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Result Card Item */
    .result-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .result-card-title {
        color: #0f172a;
        font-weight: 700;
        font-size: 13px;
    }
    .result-card-text {
        color: #334155;
        font-size: 12px;
        margin-top: 2px;
    }

    /* Bottom Status Bar */
    .status-bar {
        background-color: #1e293b;
        color: #f8fafc;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 12px;
        margin-top: 16px;
    }

    /* Streamlit Default Cleanup */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# 2. MODEL INITIALIZATION
# ----------------------------------------------------
@st.cache_resource
def load_detector():
    model_path = os.path.join(PROJECT_ROOT, "best.pt")
    detector = PlateDetector(model_path=model_path)
    success, msg = detector.load_model()
    return detector, success, msg

detector, loaded_success, load_msg = load_detector()


# ----------------------------------------------------
# 3. HEADER BANNER
# ----------------------------------------------------
st.markdown("""
<div class="header-frame">
    <div class="header-title-box">
        <div class="header-title">VisionPlate</div>
        <div class="header-subtitle">Automatic Vehicle Number Plate Detection</div>
    </div>
    <div class="badge-label">Model: YOLO11n (best.pt) | Class: number_plate</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# ----------------------------------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "standby"  # "standby", "upload_dialog", "image", "camera"
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False


# ----------------------------------------------------
# 5. CONTROL TOOLBAR
# ----------------------------------------------------
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns([1.5, 1.5, 1.5, 1.5, 0.2, 3])

with ctrl_col1:
    btn_upload_clicked = st.button("📁 Upload Image", use_container_width=True, type="primary")

with ctrl_col2:
    btn_start_cam = st.button("▶️ Start Camera", use_container_width=True, disabled=st.session_state.camera_active)

with ctrl_col3:
    btn_stop_cam = st.button("⏹️ Stop Camera", use_container_width=True, disabled=not st.session_state.camera_active)

with ctrl_col4:
    btn_clear = st.button("🔄 Clear / Reset", use_container_width=True)

with ctrl_col6:
    conf_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Adjust detection confidence threshold (Default 0.25)"
    )

# Button Handlers
if btn_clear:
    st.session_state.mode = "standby"
    st.session_state.uploaded_image_bytes = None
    st.session_state.camera_active = False
    st.rerun()

if btn_start_cam:
    st.session_state.camera_active = True
    st.session_state.mode = "camera"
    st.rerun()

if btn_stop_cam:
    st.session_state.camera_active = False
    st.session_state.mode = "standby"
    st.rerun()

if btn_upload_clicked:
    st.session_state.mode = "upload_dialog"
    st.session_state.camera_active = False


# ----------------------------------------------------
# 6. MAIN WORKSPACE (SPLITTER LAYOUT)
# ----------------------------------------------------
main_col_left, main_col_right = st.columns([2.2, 1.2])

# Left Column Elements
with main_col_left:
    st.markdown('<div class="group-box">', unsafe_allow_html=True)
    st.markdown('<div class="group-title">IMAGE / CAMERA VIEW</div>', unsafe_allow_html=True)
    view_sublabel_container = st.empty()
    video_container = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Right Column Elements
with main_col_right:
    # 1. Detection Results Box
    st.markdown('<div class="group-box">', unsafe_allow_html=True)
    st.markdown('<div class="group-title">Detection Results</div>', unsafe_allow_html=True)
    results_container = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Detected Plate Crop Box
    st.markdown('<div class="group-box">', unsafe_allow_html=True)
    st.markdown('<div class="group-title">Detected Plate</div>', unsafe_allow_html=True)
    crop_container = st.empty()
    crop_download_container = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom Status Bar Container
status_bar_container = st.empty()


# ----------------------------------------------------
# 7. EXECUTION & RENDER LOGIC
# ----------------------------------------------------

# CASE A: UPLOAD DIALOG MODE
if st.session_state.mode == "upload_dialog":
    view_sublabel_container.markdown('<div class="view-sublabel">Upload Vehicle Image</div>', unsafe_allow_html=True)
    uploaded_file = video_container.file_uploader(
        "Select Vehicle Image (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="file_uploader_input"
    )
    if uploaded_file is not None:
        st.session_state.uploaded_image_bytes = uploaded_file.read()
        st.session_state.mode = "image"
        st.rerun()
    results_container.markdown("**Total Detected Plates:** `0`", unsafe_allow_html=True)
    crop_container.caption("Cropped plate will be displayed here.")
    status_bar_container.markdown('<div class="status-bar">Status: Waiting for image upload...</div>', unsafe_allow_html=True)


# CASE B: CAMERA MODE (SUPPORTING CLIENT BROWSER WEBCAM & LOCAL LIVE STREAM)
elif st.session_state.camera_active:
    view_sublabel_container.markdown('<div class="view-sublabel">Mode: Camera Feed (Client Device & Live Stream)</div>', unsafe_allow_html=True)
    status_bar_container.markdown('<div class="status-bar">Status: Camera Active — Take photo or stream to run YOLO11n detection</div>', unsafe_allow_html=True)

    # Try Client Browser Camera First (Works on Cloud & Mobile Browsers!)
    camera_photo = video_container.camera_input("📷 Take Photo with Device Camera")

    if camera_photo is not None:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            res = detector.detect(img_bgr, conf_threshold=conf_threshold)

            if res["success"]:
                detections = res["detections"]
                det_count = len(detections)

                # Show annotated detection view
                annotated_rgb = cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB)
                video_container.image(annotated_rgb, use_container_width=True)

                if det_count > 0:
                    results_html = f"**Total Detected Plates:** `{det_count}`<br><br>"
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        conf_pct = det["confidence"] * 100.0
                        results_html += f"""
                        <div class="result-card">
                            <div class="result-card-title">Plate #{det['index']} — {det['class_name']}</div>
                            <div class="result-card-text"><b>Confidence:</b> {conf_pct:.1f}%</div>
                            <div class="result-card-text"><b>Bounding Box:</b> <code>({x1}, {y1}, {x2}, {y2})</code></div>
                        </div>
                        """
                    results_container.markdown(results_html, unsafe_allow_html=True)

                    first_det = detections[0]
                    crop_bgr = first_det["crop"]
                    if crop_bgr is not None and crop_bgr.size > 0:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        crop_container.image(crop_rgb, use_container_width=True)
                        save_crop_to_disk(crop_bgr)

                        is_success, buffer = cv2.imencode(".jpg", crop_bgr)
                        if is_success:
                            crop_download_container.download_button(
                                label="⬇️ Download Cropped Plate",
                                data=buffer.tobytes(),
                                file_name=f"plate_crop_{first_det['index']}.jpg",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                else:
                    results_container.markdown("**Total Detected Plates:** `0`<br><span style='color: #64748b; font-size: 12px;'>No plate detected in camera frame.</span>", unsafe_allow_html=True)
                    crop_container.caption("No cropped plate available.")
    else:
        results_container.markdown("**Total Detected Plates:** `0`<br><span style='color: #64748b; font-size: 12px;'>Waiting for camera photo capture...</span>", unsafe_allow_html=True)
        crop_container.caption("Cropped plate will be displayed here.")


# CASE C: STATIC IMAGE DETECTION MODE
elif st.session_state.mode == "image" and st.session_state.uploaded_image_bytes is not None:
    file_bytes = np.asarray(bytearray(st.session_state.uploaded_image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is not None:
        res = detector.detect(img_bgr, conf_threshold=conf_threshold)

        if res["success"]:
            detections = res["detections"]
            det_count = len(detections)

            annotated_rgb = cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB)
            view_sublabel_container.markdown(f'<div class="view-sublabel">Image Processed | Detections: {det_count}</div>', unsafe_allow_html=True)
            video_container.image(annotated_rgb, use_container_width=True)

            if det_count > 0:
                results_html = f"**Total Detected Plates:** `{det_count}`<br><br>"
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    conf_pct = det["confidence"] * 100.0
                    results_html += f"""
                    <div class="result-card">
                        <div class="result-card-title">Plate #{det['index']} — {det['class_name']}</div>
                        <div class="result-card-text"><b>Confidence:</b> {conf_pct:.1f}%</div>
                        <div class="result-card-text"><b>Bounding Box:</b> <code>({x1}, {y1}, {x2}, {y2})</code></div>
                    </div>
                    """
                results_container.markdown(results_html, unsafe_allow_html=True)

                first_det = detections[0]
                crop_bgr = first_det["crop"]
                if crop_bgr is not None and crop_bgr.size > 0:
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crop_container.image(crop_rgb, use_container_width=True)
                    save_crop_to_disk(crop_bgr)

                    is_success, buffer = cv2.imencode(".jpg", crop_bgr)
                    if is_success:
                        crop_download_container.download_button(
                            label="⬇️ Download Cropped Plate",
                            data=buffer.tobytes(),
                            file_name=f"plate_crop_{first_det['index']}.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
            else:
                results_container.markdown("**Total Detected Plates:** `0`<br><span style='color: #64748b; font-size: 12px;'>No plate detected at current confidence threshold.</span>", unsafe_allow_html=True)
                crop_container.caption("No cropped plate available.")

            status_bar_container.markdown(f'<div class="status-bar">Status: Image Detection Complete ({det_count} plate(s) found). Saved to outputs/crops/</div>', unsafe_allow_html=True)
        else:
            video_container.error(f"Detection failed: {res['error']}")
            status_bar_container.markdown(f'<div class="status-bar">Status: Detection Error — {res["error"]}</div>', unsafe_allow_html=True)


# CASE D: STANDBY MODE
else:
    view_sublabel_container.markdown('<div class="view-sublabel">Status: Standby</div>', unsafe_allow_html=True)
    video_container.info("No Image or Camera Stream Loaded.\n\nClick **'Upload Image'** or **'Start Camera'** above to begin detection.")
    results_container.markdown("**Total Detected Plates:** `0`", unsafe_allow_html=True)
    crop_container.caption("Cropped plate will be displayed here.")
    status_bar_container.markdown('<div class="status-bar">Status: Model \'best.pt\' loaded successfully. Ready for inference.</div>', unsafe_allow_html=True)
