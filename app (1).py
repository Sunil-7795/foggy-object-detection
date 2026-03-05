import streamlit as st
import cv2
import av
import numpy as np
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Page config
st.set_page_config(page_title="YOLOv8 Real-Time Detection", layout="wide")

st.title("🎥 YOLOv8 Real-Time Object Detection")

# Load model
model = YOLO("yolov8n.pt")

# Tabs
tab1, tab2 = st.tabs(["📷 Real-Time Camera", "🎬 Video Upload Detection"])


# ===============================
# REAL-TIME CAMERA DETECTION
# ===============================
class YOLOProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img)

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


with tab1:

    st.subheader("Live Webcam Detection")

    webrtc_streamer(
        key="yolo",
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )


# ===============================
# VIDEO DETECTION
# ===============================
with tab2:

    st.subheader("Video Upload Detection")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4","avi","mov","mkv"]
    )

    if uploaded_video is not None:

        st.success("Video uploaded successfully!")

        # Save uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO detection
            results = model(frame)

            annotated_frame = results[0].plot()

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            stframe.image(annotated_frame)

        cap.release()

        st.success("Video processing completed!")
