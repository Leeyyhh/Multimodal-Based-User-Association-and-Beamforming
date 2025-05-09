# Multimodal-Based User Association and Beamforming

This repository supports our work on **vision-aided joint user association and beamforming optimization** in wireless networks. The system leverages both visual (camera) data and RF pilot signals to make optimized user association and beamforming decisions using Graph Neural Networks (GNNs).

---

## 📁 Dataset Access

### 1. YOLOv5 Training for Object Detection

Train YOLOv5 with supervision to detect users (e.g., vehicles) from multi-camera views.

- 🔗 [YOLOv5 Detection Training Dataset](https://drive.google.com/file/d/1zTJ5cJ2EAZE74LPTIOUnJHBjfP08XAC1/view?usp=drive_link)

> **Note**: YOLOv5 is used only for detection. During GNN training, YOLOv5 parameters are fixed.
> 

---

### 2. GNN-Based Optimization: User Association & Beamforming

We use the detection results and RF pilot data to train a GNN for joint optimization. Due to device limitations, we precompute detection results and use them directly.

#### 📸 Camera Images
- First camera images: *https://drive.google.com/file/d/17BzrcXxvhINju0FPcEbm7HiWJqRp2LDT/view?usp=drive_link*
- Second camera images: *[Camera 2](https://drive.google.com/file/d/1TH-mo6iVHRKkDO66S448kZDXgcT-77HW/view?usp=sharing)*
- Third camera images: *[Camera 3](https://drive.google.com/file/d/1bPdIjcDFpHx-KfeUhmOP16_kXDkNC9d0/view?usp=sharing)*
- Fourth camera images: *[Camera 4](https://drive.google.com/file/d/1ODPMXXoUPI7X0HBBMqWeDpV1q7vymxNj/view?usp=sharing)*

#### 📦 YOLOv5 Detection Results
- Four-camera detection results: *[Detection result in tensor](https://drive.google.com/file/d/1GGs_ZP3ueztmBzYmawAz679lKjuoSFaH/view?usp=sharing)*

#### 📡 RF Pilot Data
- Pilot data: *[Communication data](https://drive.google.com/file/d/1RLLXLdPLCgVopW6LRlQrg_ZPHoLoy65N/view?usp=sharing)*

---

## 🔧 Pipeline Overview

1. **Train YOLOv5** on labeled images for object detection.
2. **Run detection** on unseen multi-view camera images.
3. **Extract features** from detection results and pilot signals.
4. **Train GNN model** to perform user association and beamforming optimization based on extracted multimodal features.

---

## 📌 Notes

- The detection stage and optimization stage are decoupled.
- This framework ensures scalability and can generalize to unseen scenarios using vision and RF data.
- GNN architecture is designed to exploit spatial relationships and multimodal data dependencies.

---

## 📄 License

 

---

## ✏️ Citation

If you use this code or dataset in your research, please cite our work:

