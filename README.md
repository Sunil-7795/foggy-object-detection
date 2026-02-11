#📌 Overview

Fog significantly degrades visibility and poses a serious challenge to real-time vision-based intelligent transportation systems. This project presents a fog-aware real-time object detection framework that integrates Gray-level Density and Illumination-based Preprocessing (GDIP) with YOLOv8 to improve object detection accuracy in foggy road scenarios while maintaining real-time performance.

The proposed GDIP-YOLOv8 framework adaptively enhances image visibility based on estimated fog density before object detection, enabling robust detection without modifying the underlying YOLOv8 architecture.

🎯 Objectives

Detect objects accurately in foggy and low-visibility road conditions

Estimate fog density using global image statistics

Enhance image contrast and illumination adaptively

Preserve real-time inference speed

Enable deployment on edge and GPU-based systems

🧠 Methodology

The system consists of the following stages:

Input Acquisition
Foggy road images or video frames are captured.

Fog Density Estimation
Fog density is estimated using global grayscale image statistics (mean and standard deviation).

GDIP-Based Image Enhancement
Adaptive contrast and illumination enhancement using CLAHE based on fog density.

YOLOv8 Object Detection
Enhanced images are passed to an anchor-free YOLOv8 detector to identify objects such as vehicles and pedestrians.

Real-Time Visualization
Detected objects are visualized with bounding boxes and confidence scores.

🏗️ System Architecture
Input Image
     ↓
Fog Density Estimation
     ↓
GDIP Image Enhancement
     ↓
YOLOv8 Object Detector
     ↓
Real-Time Object Detection Output

📂 Datasets Used

Foggy Cityscapes Dataset

RTTS (Real-world Task-driven Testing Set)

Real-world foggy road images

Dataset split:

70% Training

15% Validation

15% Testing

⚙️ Implementation Details

Model: YOLOv8-s

Optimizer: Adam

Epochs: 100

Batch Size: 16

Learning Rate: 0.001

Data Augmentation: Brightness variation, horizontal flipping

Hardware:

NVIDIA RTX 3060 (GPU)

NVIDIA Jetson Xavier NX (Edge device)

📊 Performance Results

Precision: 1.00

Recall: 1.00

mAP@0.5: 0.99

mAP@0.5:0.95: 0.93

Inference Speed: ~32 FPS (GPU)

Edge Deployment: ~18 FPS (Jetson Xavier NX)

The results demonstrate strong detection accuracy and real-time feasibility under varying fog densities.

✅ Key Advantages

Fog-aware adaptive preprocessing

Lightweight and computationally efficient

No architectural modification to YOLOv8

Real-time performance preserved

Suitable for intelligent transportation systems (ITS)

🚀 Applications

Intelligent Transportation Systems (ITS)

Advanced Driver Assistance Systems (ADAS)

Autonomous Vehicles

Traffic Surveillance in Adverse Weather

Road Safety Monitoring

⚠️ Limitations

Evaluated primarily on daytime fog conditions

Performance in night-time fog and extreme weather not fully explored

🔮 Future Work

Extend to night-time and low-light fog scenarios

Support additional adverse weather conditions (rain, snow)

Improve generalization with larger datasets

Integrate multi-sensor data (LiDAR, Radar)


👩‍💻 Authors

Nivedita Manohar Mathkunti

Sunil

Mamatha Kumari Singh

Varun A R

Vinod S Toravi


📜 License

This project is licensed under the MIT License.
