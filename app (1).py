import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
import tempfile

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="YOLOv8 Detection App", layout="wide")

# ------------------------
# Custom CSS (Premium Dark + Neon UI)
# ------------------------
st.markdown("""
<style>

body {
    background-color: #0d0d0d;
    color: #ffffff;
}

/* Main Title Glowing Effect */
.main-title {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: #00ffcc;
    text-shadow: 0px 0px 15px #00ffcc, 0px 0px 25px #00ffcc;
    animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 10px #00ffcc; }
    to   { text-shadow: 0 0 25px #00ffaa, 0 0 45px #00ffaa; }
}

/* Card Style */
.block-card {
    padding: 25px;
    margin-top: 20px;
    background: #111111;
    border-radius: 15px;
    border: 1px solid #00ffcc55;
    box-shadow: 0 0 20px #00ffcc33;
}

/* Stylish Buttons */
.stButton button {
    background-color: #00ffaa !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    box-shadow: 0 0 15px #00ffaa;
    transition: 0.3s;
}
.stButton button:hover {
    background-color: #00ffcc !important;
    box-shadow: 0px 0px 25px #00ffcc;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}
.stTabs [data-baseweb="tab"] {
    padding: 14px 25px;
    border-radius: 10px;
    background-color: #111111;
    border: 1px solid #00ffaa55;
    color: #00ffaa;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #00ffaa22;
}

</style>
""", unsafe_allow_html=True)


# ------------------------
# Title
# ------------------------
st.markdown("<h1 class='main-title'>⚡ YOLOv8 Real-Time + Video Detection</h1>", unsafe_allow_html=True)
st.write("")

# ------------------------
# Load YOLO Model
# ------------------------
model = YOLO("yolov8n.pt")   # Replace with your trained model


# ============================================================
#  TABS (Webcam + Video Upload)
# ============================================================
tabs = st.tabs(["📷 Real-Time Camera", "🎬 Video Upload Detection"])

# ============================================================
# 1️⃣ REAL-TIME CAMERA DETECTION
# ============================================================
with tabs[0]:
    st.markdown("<div class='block-card'>", unsafe_allow_html=True)
    st.subheader("📡 Live YOLOv8 Object Detection")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.empty()

    if run:
        camera = cv2.VideoCapture(0)
        st.success("Webcam started!")

        pulse_factor = 0

        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access camera.")
                break

            results = model(frame)
            annotated = results[0].plot()

            pulse_factor = (pulse_factor + 4) % 255

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls_id]} {conf:.2f}"

                color = (pulse_factor, 255 - pulse_factor, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            FRAME_WINDOW.image(annotated, channels="BGR")
            time.sleep(0.03)

        camera.release()
        st.warning("Camera Stopped.")

    else:
        st.info("Start the camera to begin detection.")

    st.markdown("</div>", unsafe_allow_html=True)



# ============================================================
# 2️⃣ VIDEO UPLOAD YOLO DETECTION (FULLY FIXED)
# ============================================================
with tabs[1]:
    st.markdown("<div class='block-card'>", unsafe_allow_html=True)
    st.subheader("🎬 Upload Video for YOLO Detection")

    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        st.success("Video uploaded successfully!")
        st.info("Processing video... Please wait 🔄")

        # Save uploaded file to temp location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name

        # Load video
        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()  # Frame display window

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model(frame)
            annotated_bgr = results[0].plot()

            # BGR → RGB conversion for Streamlit
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            # Display in Streamlit
            stframe.image(annotated_rgb, channels="RGB", use_column_width=True)

        cap.release()
        st.success("✔ Video Processing Completed!")

    else:
        st.info("Upload a video to start detection.")

    st.markdown("</div>", unsafe_allow_html=True)
