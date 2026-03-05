import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
import tempfile

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="YOLOv8 Real-Time Detection", layout="wide")

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>
body {
    background-color: #111111;
    color: white;
}

h1 {
    text-align: center;
    color: white;
    font-size: 48px;
}

.stButton button {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Title
# ------------------------
st.markdown("<h1>🎥 YOLOv8 Real-Time Detection</h1>", unsafe_allow_html=True)

# ------------------------
# Load YOLO model
# ------------------------
model = YOLO("yolov8n.pt")

# ------------------------
# Tabs
# ------------------------
tab1, tab2 = st.tabs(["📷 Camera Detection", "🎬 Video Detection"])

# ======================================================
# CAMERA DETECTION
# ======================================================
with tab1:

    st.write("Capture image from webcam")

    camera_image = st.camera_input("Take Picture")

    if camera_image is not None:

        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame)
        annotated = results[0].plot()

        # keep your detection style
        pulse_factor = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{model.names[cls_id]} {conf:.2f}"

            color = (pulse_factor, 255 - pulse_factor, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        st.image(annotated, channels="BGR")

# ======================================================
# VIDEO DETECTION
# ======================================================
with tab2:

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.success("Video uploaded successfully!")

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)

        stframe = st.empty()

        pulse_factor = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            pulse_factor = (pulse_factor + 5) % 255

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label = f"{model.names[cls_id]} {conf:.2f}"

                color = (pulse_factor, 255 - pulse_factor, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

            stframe.image(annotated, channels="BGR")

            time.sleep(0.03)

        cap.release()

        st.success("Video processing completed")
    st.markdown("</div>", unsafe_allow_html=True)
