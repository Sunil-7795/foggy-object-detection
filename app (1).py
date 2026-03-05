import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

# ----------------------------
# Custom UI Styling
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #0f1116;
    color: white;
}

.main-title {
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#00ffaa;
}

.block-card {
    padding:20px;
    border-radius:12px;
    background:#1a1c23;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.markdown("<h1 class='main-title'>⚡ YOLOv8 Object Detection System</h1>", unsafe_allow_html=True)

st.write("Use **Real-Time Camera or Video Upload** to detect objects.")

# ----------------------------
# Load YOLO Model
# ----------------------------
model = YOLO("yolov8n.pt")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📷 Real-Time Camera", "🎬 Video Detection"])

# ====================================================
# REAL-TIME CAMERA DETECTION
# ====================================================
with tab1:

    st.markdown("<div class='block-card'>", unsafe_allow_html=True)

    st.subheader("Real-Time Camera Detection")

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:

        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        results = model(image)

        annotated = results[0].plot()

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated, caption="Detection Result")

    else:
        st.info("Use your webcam to capture an image.")

    st.markdown("</div>", unsafe_allow_html=True)


# ====================================================
# VIDEO DETECTION
# ====================================================
with tab2:

    st.markdown("<div class='block-card'>", unsafe_allow_html=True)

    st.subheader("Upload Video for YOLO Detection")

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.success("Video uploaded successfully!")
        st.info("Processing video... Please wait.")

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame)

            annotated = results[0].plot()

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            stframe.image(annotated, channels="RGB")

        cap.release()

        st.success("✔ Video Processing Completed!")

    else:
        st.info("Upload a video to start detection.")

    st.markdown("</div>", unsafe_allow_html=True)
