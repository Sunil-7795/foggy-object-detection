import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import tempfile

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="YOLOv8 Detection App", layout="wide")

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>

body {
    background-color: #111111;
    color: white;
}

.main-title {
    text-align:center;
    font-size:50px;
    color:#00ffaa;
}

.block-card{
    padding:20px;
    background:#1a1a1a;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------
# Title
# ------------------------
st.markdown("<h1 class='main-title'>⚡ YOLOv8 Real-Time Detection</h1>", unsafe_allow_html=True)

# ------------------------
# Load YOLO Model
# ------------------------
model = YOLO("yolov8n.pt")


# =====================================================
# YOLO Webcam Processor
# =====================================================

class YOLOProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img)

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =====================================================
# Tabs
# =====================================================

tab1, tab2 = st.tabs(["📷 Real-Time Camera", "🎬 Video Upload Detection"])


# =====================================================
# Real-Time Webcam Detection
# =====================================================

with tab1:

    st.markdown("<div class='block-card'>", unsafe_allow_html=True)

    st.subheader("📡 Live YOLOv8 Detection")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="yolo",
        video_processor_factory=YOLOProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# Video Upload Detection
# =====================================================

with tab2:

    st.markdown("<div class='block-card'>", unsafe_allow_html=True)

    st.subheader("🎬 Upload Video for YOLO Detection")

    uploaded_video = st.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv"]
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

            stframe.image(annotated, channels="RGB")

        cap.release()

        st.success("✔ Video Processing Completed!")

    else:
        st.info("Upload a video to start detection.")

    st.markdown("</div>", unsafe_allow_html=True)
