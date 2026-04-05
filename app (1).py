import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="YOLOv8 Detection App", layout="wide")

# ------------------------
# Load Model (cached)
# ------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------------
# Title
# ------------------------
st.title("⚡ YOLOv8 Real-Time + Video Detection")

# ============================================================
# TABS
# ============================================================
tabs = st.tabs(["📷 Webcam Detection", "🎬 Video Upload Detection"])

# ============================================================
# 1️⃣ WEBCAM (WORKS IN STREAMLIT CLOUD)
# ============================================================
with tabs[0]:
    st.subheader("📡 Live Webcam Detection (Browser Camera)")

    class YOLOTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            results = model(img)
            annotated = results[0].plot()

            return annotated

    webrtc_streamer(
        key="webcam",
        video_transformer_factory=YOLOTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

# ============================================================
# 2️⃣ VIDEO UPLOAD
# ============================================================
with tabs[1]:
    st.subheader("🎬 Upload Video for Detection")

    uploaded_video = st.file_uploader(
        "Upload Video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        st.success("Video uploaded successfully!")

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
            stframe.image(annotated)

        cap.release()
        st.success("✅ Video processing completed!")

    else:
        st.info("Upload a video to start detection.")
