# Topology-Aware Deep Learning Framework for Retinal Vessel Segmentation in OCT Angiography

## Overview

This project implements a **Topology-Aware Skeleton U-Net** for retinal blood vessel segmentation in **Optical Coherence Tomography Angiography (OCTA)** images.

The framework is designed to preserve vessel connectivity by incorporating skeleton information into the training process. It performs automatic vessel segmentation using a deep learning pipeline built with **PyTorch**.

---

## Features

- Skeleton-Aware U-Net architecture
- Automatic skeleton generation from vessel labels
- Custom dataset loader
- Combined segmentation loss function
- Model checkpoint saving
- Prediction on unseen OCTA images
- Automatic segmentation mask generation
- CPU and GPU compatible

---

## Dataset

**Dataset Used:** OCTA-500

Dataset Structure

```
Dataset/
└── OCTA-500/
    ├── Training/
    │   ├── OCTA500_3M_train_images/
    │   ├── OCTA500_3M_train_labels/
    │   └── Skeletons/
    │
    ├── Validation/
    │   ├── OCTA500_3M_val_images/
    │   ├── OCTA500_3M_val_labels/
    │   └── Skeletons/
    │
    ├── Testing/
    │   ├── OCTA500_3M_test_images/
    │   ├── OCTA500_3M_test_labels/
    │   └── Skeletons/
    │
    └── Output/
```

Current Dataset

- Training Images : 140
- Validation Images : 10
- Testing Images : 50

---

## Project Structure

```
Topology_OCTA/
│
├── config.py
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── losses.py
├── skeletonize.py
├── utils.py
├── requirements.txt
├── README.md
│
├── Checkpoints/
│   └── best_model.pth
│
└── Dataset/
```

---

## Model Architecture

```
Input Image
      │
      ▼
Double Convolution
      │
      ▼
Encoder
      │
      ▼
Bottleneck
      │
      ▼
Decoder
      │
      ▼
1×1 Convolution
      │
      ▼
Sigmoid
      │
      ▼
Segmented Vessel Mask
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Pritam-Bharadwaz/Topology-Aware-Deep-Learning-Framework-for-Retinal-Vessel-Segmentation-in-OCT-Angiography.git
```

Move into the project folder

```bash
cd Topology-Aware-Deep-Learning-Framework-for-Retinal-Vessel-Segmentation-in-OCT-Angiography
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Generate Skeleton Images

```bash
python skeletonize.py
```

---

## Train the Model

```bash
python train.py
```

The best model will be saved in

```
Checkpoints/
```

---

## Prediction

Generate vessel segmentation masks

```bash
python predict.py
```

Predicted images are automatically saved in

```
Dataset/OCTA-500/Output
```

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- scikit-image
- Albumentations
- TensorBoard
- tqdm

---

## Results

The model successfully performs retinal vessel segmentation on OCTA images.

Output images are generated automatically after prediction and stored in the **Output** directory.

---

## Future Improvements

- Graph Neural Networks (GNN)
- Topology Loss Functions
- clDice Loss
- Transformer-based Encoder
- Multi-scale Feature Fusion
- Attention Gates
- Quantitative Evaluation (Dice Score, IoU)

---

## Author

**Pritam Bharadwaz**

B.Tech in Electronics & Communication Engineering (ECE)

Barak Valley Engineering College

GitHub: https://github.com/Pritam-Bharadwaz

---

## License

This project is developed for academic and research purposes.