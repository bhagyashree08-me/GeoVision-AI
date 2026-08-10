
# 🌍 GeoVision AI

> **AI-powered Land Cover Segmentation using Deep Learning**

GeoVision AI is an end-to-end deep learning project for semantic segmentation of high-resolution satellite imagery. The objective is to classify every pixel of an aerial image into different land-cover categories such as urban areas, forests, water bodies, agriculture, barren land, and more.

This project is being built from scratch using **PyTorch**, following a modular and production-oriented architecture instead of relying on high-level training frameworks.

---

# 🎯 Project Goal

The primary objective of GeoVision AI is to build an intelligent land-cover segmentation system that can:

* Detect different land-cover regions from satellite imagery.
* Produce pixel-level segmentation masks.
* Assist in environmental monitoring and urban planning.
* Serve as a portfolio-quality Computer Vision project for AI/ML roles.

---

# 🗂 Dataset

**Dataset:** DeepGlobe Land Cover Classification Dataset

The dataset contains RGB satellite images and corresponding segmentation masks.

### Land Cover Classes

| Class ID | Class            |
| -------: | ---------------- |
|        0 | Urban Land       |
|        1 | Agriculture Land |
|        2 | Rangeland        |
|        3 | Forest Land      |
|        4 | Water            |
|        5 | Barren Land      |
|        6 | Unknown          |

---

# 🛠 Tech Stack

### Programming

* Python

### Deep Learning

* PyTorch
* TorchVision

### Image Processing

* OpenCV
* Pillow
* NumPy

### Data Augmentation

* Albumentations

### Data Handling

* Scikit-Learn

### Visualization

* Matplotlib

### Future Deployment

* Streamlit

---

# 📁 Project Structure

```
GeoVision-AI/
│
├── app/
├── assets/
├── configs/
├── data/
├── docs/
├── logs/
├── models/
├── notebooks/
├── outputs/
├── reports/
│
├── src/
│   ├── dataset/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── utils/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ✅ Progress

## Completed

### Dataset Exploration

* Dataset inspection utility
* Dataset statistics
* Class verification
* CSV metadata exploration

### Data Pipeline

* Custom `DeepGlobeDataset`
* RGB image loading
* RGB mask loading
* RGB mask → Class ID conversion
* Albumentations integration
* Image normalization
* Train/Validation transforms

### DataLoader

* Automatic train/validation split
* PyTorch DataLoader
* Batch loading
* Data shuffling
* Batch verification

### Validation

Verified:

* Image tensor shape
* Mask tensor shape
* Dataset loading
* Class IDs
* Data augmentation pipeline

---

# 🚧 Current Status

**Current Phase**

✅ Dataset Pipeline Completed

Next milestone:

* Build U-Net architecture
* Train segmentation model
* Evaluate using IoU and Dice Score
* Build inference pipeline
* Deploy with Streamlit

---

# 🧠 Planned Architecture

```
Satellite Image
        │
        ▼
Data Loader
        │
        ▼
Data Augmentation
        │
        ▼
U-Net
        │
        ▼
Predicted Segmentation Mask
        │
        ▼
Visualization
        │
        ▼
Streamlit Application
```

---

# 📌 Roadmap

* [x] Dataset preparation
* [x] Dataset exploration
* [x] Dataset class
* [x] Data augmentation
* [x] DataLoader
* [x] Mask preprocessing
* [ ] U-Net implementation
* [ ] Training pipeline
* [ ] Validation pipeline
* [ ] IoU metric
* [ ] Dice Score
* [ ] Model checkpointing
* [ ] Inference pipeline
* [ ] Prediction visualization
* [ ] Streamlit web application
* [ ] Project documentation

---

# 📊 Current Milestone

**Phase 1 — Dataset Pipeline**

✔ Dataset verified

✔ Dataset exploration

✔ RGB mask preprocessing

✔ Data augmentation

✔ DataLoader implementation

✔ Batch validation

---

# 🚀 Future Improvements

* Attention U-Net
* DeepLabV3+
* SegFormer
* Model comparison
* Mixed precision training
* ONNX export
* Docker deployment
* Hugging Face Spaces deployment

---

# 📷 Expected Output

Input satellite image

⬇

Semantic segmentation mask

⬇

Colorized land-cover prediction

---

# 👩‍💻 Author

**Bhagyashree**

MCA Student | AI & Machine Learning Enthusiast

GitHub: https://github.com/bhagyashree08-me

---

# ⭐ Repository Status

This project is currently under active development. New features and improvements are being added incrementally as the complete semantic segmentation pipeline is built.
