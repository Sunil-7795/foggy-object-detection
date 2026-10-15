
# 🌫️ Object Detection in Foggy Weather Conditions

A fog-aware real-time object detection framework combining Gray-level Density and Illumination-based Preprocessing (GDIP) with YOLOv8 to improve object detection accuracy in foggy road environments while maintaining real-time performance.

This project enhances image visibility based on estimated fog density before detecting road objects such as vehicles and pedestrians using YOLOv8.

Key Technologies: Python · YOLOv8 · OpenCV · CLAHE · Deep Learning · Computer Vision


## 🎯 Objectives

* Detect objects accurately in foggy and low-visibility road conditions.
* Estimate fog density using global grayscale image statistics.
* Enhance image contrast and illumination adaptively.
* Maintain real-time inference performance.
* Support deployment on GPU-based systems and edge devices.
## 🧠 Methodology

The proposed GDIP-YOLOv8 framework consists of the following stages:

1. **Input Acquisition**

   Foggy road images or video frames are captured as input to the system.

2. **Fog Density Estimation**

   Fog density is estimated using global grayscale image statistics, including mean intensity and standard deviation.

3. **GDIP-Based Image Enhancement**

   Adaptive image enhancement is performed using Contrast Limited Adaptive Histogram Equalization (CLAHE) based on the estimated fog density to improve visibility and contrast.

4. **YOLOv8 Object Detection**

   The enhanced images are passed to the YOLOv8-s object detection model to identify road objects such as vehicles and pedestrians.

5. **Real-Time Visualization**

   The detected objects are displayed with bounding boxes, class labels, and confidence scores for real-time monitoring.
## 🏗️ System Architecture

The proposed system follows a sequential pipeline in which fog density estimation and image enhancement are performed before object detection.

```text
Input Image / Video Frame
          |
          v
Fog Density Estimation
          |
          v
GDIP Image Enhancement
          |
          v
YOLOv8 Object Detector
          |
          v
Object Detection Results
          |
          v
Bounding Boxes, Labels,
and Confidence Scores
```

### Architecture Components

- **Input Module:** Receives foggy road images or video frames.
- **Fog Estimation Module:** Calculates grayscale image statistics to estimate fog density.
- **Enhancement Module:** Applies adaptive contrast enhancement using CLAHE.
- **Detection Module:** Uses YOLOv8-s to detect road objects.
- **Output Module:** Displays detected objects with bounding boxes and confidence scores.
## 📂 Datasets Used

The proposed framework uses foggy road images and real-world datasets to evaluate object detection performance under low-visibility conditions.

### 1. Foggy Cityscapes Dataset

- Contains urban road-scene images captured under foggy conditions.
- Supports the evaluation of object detection in adverse visibility.
- Provides road-scene information relevant to intelligent transportation systems.

### 2. RTTS (Real-world Task-driven Testing Set)

- Contains real-world hazy images.
- Helps evaluate object detection performance in challenging visibility conditions.
- Supports testing the generalization of detection models on real-world images.

### 3. Real-World Foggy Road Images

- Used to test the framework under practical road conditions.
- Help assess image enhancement and object detection performance in fog.

### 📊 Dataset Split

| Dataset Category | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |



## ⚙️ Implementation Details

The GDIP-YOLOv8 framework combines adaptive image preprocessing with deep learning-based object detection to improve visibility and detection performance in foggy road environments.

### 🧠 Model Configuration

| Parameter | Value |
|---|---|
| Object Detection Model | YOLOv8-s |
| Preprocessing Technique | GDIP |
| Contrast Enhancement | CLAHE |
| Optimizer | Adam |
| Number of Epochs | 100 |
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Data Augmentation | Brightness Variation, Horizontal Flipping |

### 💻 Hardware Requirements

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 3060 |
| Edge Device | NVIDIA Jetson Xavier NX |

### 🛠️ Technologies Used

* **Programming Language:** Python
* **Object Detection:** YOLOv8
* **Image Processing:** OpenCV
* **Image Enhancement:** CLAHE
* **Numerical Computing:** NumPy
* **Deep Learning Framework:** PyTorch
* **Core Concepts:** Computer Vision and Deep Learning

### 🚀 Implementation Workflow

1. Load foggy road images or video frames.
2. Estimate fog density using grayscale image statistics.
3. Apply adaptive image enhancement using GDIP and CLAHE.
4. Pass the enhanced images to the YOLOv8-s detector.
5. Generate object detection results with bounding boxes, class labels, and confidence scores.
6. Evaluate detection performance and inference speed.
## 📊 Performance Results

The proposed GDIP-YOLOv8 framework aims to improve object detection in foggy road environments while maintaining real-time inference performance.

### 📈 Evaluation Metrics

| Metric | Score | Performance |
|---|---:|---|
| Precision | 1.00 (100%) | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 |
| Recall | 1.00 (100%) | 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩 |
| mAP@0.5 | 0.99 (99%) | 🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧 |
| mAP@0.5:0.95 | 0.93 (93%) | 🟪🟪🟪🟪🟪🟪🟪🟪🟪⬜ |

### ⚡ Inference Speed

| Hardware Platform | Speed | Performance |
|---|---:|---|
| NVIDIA RTX 3060 GPU | 32 FPS | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 |
| NVIDIA Jetson Xavier NX | 18 FPS | 🟩🟩🟩🟩🟩🟦⬜⬜⬜⬜ |

### 🔍 Results Summary

- **High Detection Accuracy:** Reported mAP@0.5 of 0.99.
- **Strong Detection Performance:** Reported precision and recall of 1.00.
- **Real-Time GPU Processing:** Approximately 32 FPS on NVIDIA RTX 3060.
- **Edge Deployment:** Approximately 18 FPS on NVIDIA Jetson Xavier NX.
- **Fog-Aware Enhancement:** Adaptive preprocessing improves image visibility before object detection.
## ✅ Key Advantages

- **Fog-Aware Preprocessing:** Estimates fog density and adapts image enhancement accordingly.
- **Improved Visibility:** Enhances image contrast using CLAHE before object detection.
- **YOLOv8 Compatibility:** Works with the existing YOLOv8 architecture without architectural modifications.
- **Real-Time Performance:** Supports fast object detection on GPU-based systems.
- **Edge Deployment:** Can be deployed on edge devices such as NVIDIA Jetson Xavier NX.
- **Computational Efficiency:** Combines image enhancement with a lightweight YOLOv8-s detection model.

## 🚀 Applications

- **Intelligent Transportation Systems (ITS):** Improves road-scene monitoring in foggy weather.
- **Advanced Driver Assistance Systems (ADAS):** Supports environmental perception in low-visibility conditions.
- **Autonomous Vehicles:** Helps detect vehicles and pedestrians in foggy environments.
- **Traffic Surveillance:** Supports monitoring of roads and highways during adverse weather.
- **Road Safety Monitoring:** Assists in identifying road users under reduced visibility.
- **Smart Traffic Management:** Supports intelligent monitoring systems for traffic analysis.
## ⚠️ Limitations

- **Daytime Fog Conditions:** The framework has primarily been evaluated under daytime fog conditions.
- **Night-Time Performance:** Detection performance in night-time fog and low-light environments requires further evaluation.
- **Extreme Weather Conditions:** Performance under heavy rain, snow, and other extreme weather conditions has not been fully explored.
- **Dataset Generalization:** Detection performance may vary across different road environments, camera types, and fog densities.
- **Edge Computing Constraints:** Processing speed may vary depending on hardware capabilities and image resolution.

## 🔮 Future Work

- **Night-Time Fog Detection:** Extend the framework to support low-light and night-time fog scenarios.
- **Multi-Weather Support:** Adapt the system to handle rain, snow, and other adverse weather conditions.
- **Larger Datasets:** Improve generalization using more diverse real-world road images.
- **Multi-Sensor Integration:** Combine camera data with LiDAR and radar for more robust environmental perception.
- **Edge Optimization:** Optimize preprocessing and detection for faster inference on edge devices.
- **Advanced Evaluation:** Evaluate performance across different fog densities and real-world driving scenarios.
## 👩‍💻 Authors

This project was developed by the following team members:

- **Sunil**
- **Mamatha Kumari Singh**
- **Varun A R**
- **Vinod S Toravi**

## 📜 License

This project is licensed under the [MIT](LICENSE) License.

