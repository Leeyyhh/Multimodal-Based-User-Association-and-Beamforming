# Multimodal-Based User Association and Beamforming

This repository supports our work on **vision-aided joint user association and beamforming optimization** in wireless networks. The system leverages both visual (camera) data and RF pilot signals to make optimized user association and beamforming decisions using Graph Neural Networks (GNNs).

---

## 📁 Dataset Access

### DATASET 1. YOLOv5 Training for Object Detection

Train YOLOv5 with supervision to detect users (e.g., vehicles) from multi-camera views.

- 🔗 [YOLOv5 Detection Training Dataset](https://drive.google.com/file/d/1zTJ5cJ2EAZE74LPTIOUnJHBjfP08XAC1/view?usp=drive_link)

> **Note**: YOLOv5 is used only for detection. During GNN training, YOLOv5 parameters are fixed.
> 

---

### DATASET 2. GNN-Based Optimization: User Association & Beamforming

We use the detection results and RF pilot data to train a GNN for joint optimization. Due to device limitations, we precompute detection results and use them directly.

#### 📸 Camera Images
- First camera images: *https://drive.google.com/file/d/17BzrcXxvhINju0FPcEbm7HiWJqRp2LDT/view?usp=drive_link*
- Second camera images: *[Camera 2](https://drive.google.com/file/d/1TH-mo6iVHRKkDO66S448kZDXgcT-77HW/view?usp=sharing)*
- Third camera images: *[Camera 3](https://drive.google.com/file/d/1bPdIjcDFpHx-KfeUhmOP16_kXDkNC9d0/view?usp=sharing)*
- Fourth camera images: *[Camera 4](https://drive.google.com/file/d/1ODPMXXoUPI7X0HBBMqWeDpV1q7vymxNj/view?usp=sharing)*

#### 📦 YOLOv5 Detection Results
- Four-camera detection results: *[Detection result in tensor](https://drive.google.com/file/d/1GGs_ZP3ueztmBzYmawAz679lKjuoSFaH/view?usp=sharing)*
> **Note:**  
> This detection dataset was generated using a well-trained YOLOv5 model.  
> The model used is **YOLOv5s**, trained on **DATASET 1** for **100 epochs**.  
> You can directly use this dataset as input to the GNN model, or alternatively, you may train your own YOLOv5 model on **DATASET 1** and apply it to the multi-view camera images to obtain your own detection results.

#### 📡 RF Pilot Data
- Pilot data: *[Communication data](https://drive.google.com/file/d/1RLLXLdPLCgVopW6LRlQrg_ZPHoLoy65N/view?usp=sharing)*

---

## 🔧 Pipeline Overview

1. **Train YOLOv5** on labeled images for object detection (**DATASET 1**).

2. **Run the well-trained YOLOv5** on multi-view camera images from **DATASET 2** to obtain detection results. These results are then used to generate the training and validation datasets for the GNN-based user association and beamforming network.  
   *(If your device has sufficient resources, you can transmit the detection results from YOLOv5 directly to the GNN during training.  
   However, in our implementation, we precompute all detection results and split them into training and validation sets.  
   This avoids repeated execution of YOLOv5, which would be inefficient since the same samples are reused many times during training.  
   Note that the YOLOv5 parameters are fixed, so precomputing detections does not affect the final performance.)*

3. **Train the GNN model** using both the detection results and pilot signals to optimize user association and beamforming decisions.
---

 
---

## 📄 License

 

---

## ✏️ Citation

If you use this code or dataset in your research, please cite our work:

