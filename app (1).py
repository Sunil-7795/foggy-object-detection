import streamlit as st
import cv2
import av
import numpy as np
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="YOLOv8 Detection", layout="wide")

st.title("🎥 YOLOv8 Real-Time Object Detection")

# ---------------------------
# Load YOLO Model
# ---------------------------
model = YOLO("yolov8n.pt")

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["📷 Live Camera Detection", "🎬 Video Upload Detection"])


# =========================================================
# REAL-TIME CAMERA DETECTION
# =========================================================

class YOLOProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img)

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

with tab1:

    st.subheader("Live Webcam Detection")

    webrtc_streamer(
        key="yolo",
        video_processor_factory=YOLOProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )


# =========================================================
# VIDEO DETECTION (3× FASTER)
# =========================================================

with tab2:

    st.subheader("Upload Video for Detection")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.success("Video uploaded successfully!")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        frame_count = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Process every 3rd frame
            if frame_count % 3 != 0:
                continue

            results = model(frame)

            annotated_frame = results[0].plot()

            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            stframe.image(annotated_frame)

        cap.release()

        st.success("✔ Video processing completed")
