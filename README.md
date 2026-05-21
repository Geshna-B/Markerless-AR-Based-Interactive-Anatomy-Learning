# Markerless AR-Based Interactive Anatomy Learning

### Using Learning-Based Pose Estimation and Sparse 3D Reconstruction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ObjectDetection-red)
![AR](https://img.shields.io/badge/Augmented-Reality-purple)
![Status](https://img.shields.io/badge/Status-ResearchProject-success)

---

## Overview

This project presents a **Markerless Augmented Reality (AR)-Based Interactive Anatomy Learning System** designed to enhance medical education through immersive visualization of human organs.

Unlike traditional AR systems that rely on fiducial markers such as QR codes or ArUco markers, the proposed system uses:

* Deep learning-based object detection
* Learning-based feature matching
* Sparse 3D reconstruction
* 6D pose estimation
* Real-time AR rendering

to accurately align virtual anatomical organs with real-world anatomy reference cards.

The system enables users to interactively visualize and manipulate 3D anatomical models in real time using a completely markerless pipeline.

---

## Key Features

* Markerless AR anatomy visualization
* Real-time 6D pose estimation
* Sparse 3D reconstruction using multi-view geometry
* YOLOv8-based anatomy card detection
* LoFTR-based feature matching
* Interactive 3D organ manipulation
* Multi-object AR visualization
* Real-time rendering with optimized mesh simplification
* CPU-based lightweight rendering pipeline

---

## System Architecture

The proposed pipeline consists of the following stages:

1. Camera Calibration
2. Markerless Object Detection
3. Feature Extraction and Matching
4. Sparse 3D Reconstruction
5. 6D Pose Estimation
6. AR Rendering and Visualization

---

## Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* OpenCV
* NumPy
* Open3D
* PyTorch
* Ultralytics YOLOv8

### Computer Vision Techniques

* Feature Matching
* Structure from Motion (SfM)
* Perspective-n-Point (PnP)
* Camera Calibration
* 6D Pose Estimation

### Deep Learning Models

* YOLOv8
* LoFTR (Local Feature Transformer)

---

## Dataset

A custom dataset was created for this project containing anatomy reference cards for:

* Heart
* Brain
* Lungs

### Dataset Statistics

| Split      | Heart | Brain | Lungs | Total |
| ---------- | ----- | ----- | ----- | ----- |
| Train      | 20    | 13    | 18    | 51    |
| Validation | 5     | 6     | 6     | 17    |
| Test       | 3     | 6     | 6     | 15    |
| Total      | 28    | 25    | 30    | 83    |

### Data Collection Conditions

* Different viewpoints
* Scale variations
* Partial occlusions
* Indoor lighting conditions
* Multi-object scenes

### Annotation Format

YOLO annotation format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

---

## Model Performance

### Object Detection Results

| Class   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ------- | --------- | ------ | ------- | ------------ |
| Heart   | 1.00      | 1.00   | 0.995   | 0.862        |
| Brain   | 1.00      | 0.67   | 0.995   | 0.893        |
| Lungs   | 1.00      | 1.00   | 0.995   | 0.919        |
| Overall | 1.00      | 0.89   | 0.995   | 0.891        |

---

## Ablation Study

### Feature Matching Comparison

| Method | Avg Matches | Inlier Matches | Reconstruction Quality | Runtime   |
| ------ | ----------- | -------------- | ---------------------- | --------- |
| SIFT   | 320         | 185            | Moderate               | 1259.8 ms |
| LoFTR  | 1420        | 1035           | High                   | 210 ms    |

LoFTR significantly improved:

* Correspondence quality
* Pose stability
* Reconstruction robustness
* Runtime efficiency

---

## AR Rendering Optimization

To achieve real-time performance, all 3D meshes were simplified to approximately **1500 triangles**.

| Organ | Original Triangles | Simplified Triangles |
| ----- | ------------------ | -------------------- |
| Heart | 217,600            | 1,500                |
| Brain | 130,910            | 1,500                |
| Lungs | 1,023,012          | 1,500                |

### Rendering Techniques

* Lambertian shading
* Painter’s algorithm
* Vectorized NumPy projection
* Depth buffering

### Performance

* ~30 FPS on standard CPU hardware

---

## Project Structure

```text
Markerless-AR-Based-Interactive-Anatomy-Learning/
│
├── dataset/
├── models/
├── outputs/
├── src/
├── notebooks/
├── assets/
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Geshna-B/Markerless-AR-Based-Interactive-Anatomy-Learning.git
```

Move into the project directory:

```bash
cd Markerless-AR-Based-Interactive-Anatomy-Learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the AR system:

```bash
python main.py
```

---

## Applications

* Medical Education
* Anatomy Learning
* Simulation-Based Training
* Interactive Healthcare Visualization
* AR-Assisted Learning Platforms

---

## Future Improvements

* Full GPU acceleration
* Dense 3D reconstruction
* Mobile AR deployment
* Multi-user collaborative AR
* Integration with hand tracking and gesture control
* Real-time anatomical labeling

---

## Authors

### Research Team

* Geshna B
* Malavika S Prasad
* Vibhu Sanchana
* Asmi K

### Guide

Dr. Mithun Kumar Kar
Assistant Professor (Sr. Gr.)
Amrita School of Artificial Intelligence
[Amrita Vishwa Vidyapeetham](https://www.amrita.edu?utm_source=chatgpt.com)

---

## Institution

Amrita Vishwa Vidyapeetham
Amrita School of Artificial Intelligence
Coimbatore, Tamil Nadu, India

---

## Research Contribution

This work demonstrates how deep learning, computer vision, and augmented reality can be combined to create scalable, markerless, and immersive anatomy learning systems that improve spatial understanding and educational accessibility.

---

## License

This project is intended for academic and research purposes under MIT license.
