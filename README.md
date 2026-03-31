# 🍎 Fruit Classification System

> **Deep Learning · Transfer Learning · MobileNetV2**
> A self-contained Jupyter Notebook pipeline that trains a fruit image classifier and provides an interactive prediction interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Classes Supported](#classes-supported)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [How to Use](#how-to-use)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Configuration](#configuration)
- [File Reference](#file-reference)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

This project builds a **7-class fruit image classifier** using **Transfer Learning** on top of **MobileNetV2** (pre-trained on ImageNet). The entire workflow — from raw images to live predictions — is packaged in a single Jupyter Notebook (`experiment.ipynb`) for maximum reproducibility.

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **98.14%** |
| Best Validation Loss | **0.0592** |
| Epochs Trained (of 10) | **4** (early stopping) |
| Total Parameters | 2,422,855 |
| Trainable Parameters | 164,871 (head only) |

---

## Demo

When you run **Cell 5** of the notebook, a native OS file-picker dialog opens. Select any fruit image and the model predicts:

```
============================================================
🔍 FRUIT CLASSIFICATION — PREDICTION MODULE
============================================================

📦 Step 1 — Loading trained model from '../models/fruit_model.h5'...
   ✅ Model loaded successfully! (10.9 MB)

🖼️  Step 2 — Selecting Image...
   Opening file chooser window — please select a fruit image.
   ✅ File selected: .../Apple/r0_0_100.jpg

🤖 Step 4 — Running Model Inference...

   📊 Raw Probabilities:
      Apple           ██████████████████████████████    97.0% ◀ PREDICTED
      Cherry                                             3.0%
      ...

============================================================
  🍎 Predicted Fruit : Apple
  📊 Confidence      : 97.04%
  🟢 Confidence Level: HIGH — very likely correct
============================================================
```

---

## Project Structure

```
FruitClassification/
├── notebooks/
│   └── experiment.ipynb        ← ⭐ Main entry point (run this)
├── src/
│   ├── train.py                ← Training pipeline helper
│   ├── predict.py              ← Prediction pipeline helper
│   └── utils.py                ← Visualisation helpers
├── data/
│   └── train/
│       ├── Apple/
│       ├── Banana 1/
│       ├── Blackberry 1/
│       ├── Cherry/
│       ├── Mango 1/
│       ├── Raspberry 1/
│       └── Strawberry 1/
├── models/
│   └── fruit_model.h5          ← Saved model (auto-generated after training)
├── README.md
└── Fruit_Classification_Report.pdf
```

> **Note:** The notebook (`experiment.ipynb`) is fully self-contained — all logic is written inline. The `.py` files in `src/` are standalone helpers kept for modularity but are **not imported** by the notebook.

---

## Classes Supported

| # | Class | Example |
|---|-------|---------|
| 1 | Apple | 🍎 |
| 2 | Banana 1 | 🍌 |
| 3 | Blackberry 1 | 🫐 |
| 4 | Cherry | 🍒 |
| 5 | Mango 1 | 🥭 |
| 6 | Raspberry 1 | 🫐 |
| 7 | Strawberry 1 | 🍓 |

> ⚠️ Folder names must match these class names **exactly** (case-sensitive).

---

## Technology Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.10 | Runtime |
| TensorFlow / Keras | 2.21.0 | Model training & inference |
| MobileNetV2 | Keras Apps | Pre-trained base model |
| OpenCV (`cv2`) | 4.13.0 | Image loading & colour conversion |
| NumPy | latest | Array operations |
| Matplotlib | latest | Plots & visualisations |
| tkinter | stdlib | Native file-picker dialog |
| Jupyter Notebook | any | Interactive environment |

---

## Getting Started

### Prerequisites

- Python **3.9+** (3.12 recommended)
- Jupyter Notebook: `pip install notebook`

### 1 — Clone the Repository

```bash
git clone https://github.com/your-username/FruitClassification.git
cd FruitClassification
```

### 2 — Install Dependencies

The notebook auto-installs missing packages in **Cell 1**. Alternatively, install manually:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 3 — Prepare Your Dataset

Place images inside `data/train/` following the **one-folder-per-class** structure:

```
data/train/
  Apple/         ← put all apple images here
  Banana 1/
  Blackberry 1/
  Cherry/
  Mango 1/
  Raspberry 1/
  Strawberry 1/
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### 4 — Launch the Notebook

```bash
cd notebooks
jupyter notebook experiment.ipynb
```

---

## How to Use

Run the notebook cells **top to bottom**:

| Cell | Action |
|------|--------|
| **Cell 1** | Checks and installs dependencies |
| **Cell 2** | Imports libraries, sets paths and constants |
| **Cell 3** | Trains the model *(auto-skipped if model already exists)* |
| **Cell 4** | Plots training accuracy & loss curves |
| **Cell 5** | Opens file picker → predict any fruit image |

### Re-training vs. Loading a Saved Model

- **First run** — Cell 3 trains from scratch and saves `models/fruit_model.h5`.
- **Subsequent runs** — Cell 3 detects the saved model and **skips training automatically**.
- **Force re-training** — delete `models/fruit_model.h5` before running Cell 3.

---

## Model Architecture

```
Input (224 × 224 × 3)
        │
        ▼
MobileNetV2 [FROZEN — 154 layers, ImageNet weights]
        │
        ▼
GlobalAveragePooling2D   →  (None, 1280)
        │
        ▼
Dense(128, activation='relu')   →  (None, 128)
        │
        ▼
Dense(7, activation='softmax')  →  (None, 7)
        │
        ▼
Predicted Class + Confidence
```

**Transfer learning strategy:** Feature extraction — only the classification head is trained. MobileNetV2 weights remain frozen.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Batch size | 32 |
| Max epochs | 10 |
| Early stopping | patience=3, restore_best_weights=True |
| Input size | 224 × 224 |
| Augmentation | rotation ±20°, zoom ±20%, horizontal flip |

---

## Results

### Training History

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| **1** ⭐ | 98.41% | **98.14%** | 0.0539 | **0.0592** |
| 2 | 100.00% | 89.93% | 0.0020 | 0.2352 |
| 3 | 99.99% | 97.73% | 0.0017 | 0.0642 |
| 4 | 99.94% | 90.34% | 0.0018 | 0.3229 |

> ⭐ Best epoch restored by EarlyStopping. Training stopped at epoch 4.

### Confidence Thresholds

| Confidence | Level | Interpretation |
|------------|-------|----------------|
| ≥ 80% | 🟢 HIGH | Very likely correct |
| 50–79% | 🟡 MODERATE | Probably correct |
| < 50% | 🔴 LOW | Model is uncertain |

---

## Configuration

All configurable values live in **Cell 2** of the notebook:

```python
# ── Paths ─────────────────────────────────────────────────────
DATA_PATH  = '../data/train'        # Training images root folder
MODEL_PATH = '../models/fruit_model.h5'  # Where model is saved/loaded

# ── Settings ──────────────────────────────────────────────────
IMG_SIZE   = (224, 224)   # Input image dimensions
BATCH_SIZE = 32           # Training batch size

# ── Class names (must match folder names exactly) ─────────────
CLASS_NAMES = [
    'Apple',
    'Banana 1',
    'Blackberry 1',
    'Cherry',
    'Mango 1',
    'Raspberry 1',
    'Strawberry 1'
]
```

To add a new fruit class:
1. Create a new sub-folder under `data/train/` with the class name.
2. Add the class name to `CLASS_NAMES` in Cell 2.
3. Delete the old `fruit_model.h5` and re-run Cell 3 to retrain.

---

## File Reference

### `src/train.py`

```python
train_model(data_path="../data/train", model_path="../models/fruit_model.h5")
# Returns: (model, history, class_names)
```

Trains MobileNetV2 with augmentation and early stopping, saves the model to disk.

---

### `src/predict.py`

```python
run_prediction(model_path="../models/fruit_model.h5", class_names=CLASS_NAMES)
# Returns: (img_rgb, predicted_class, confidence, all_probs, class_names)
```

Loads the model, opens a GUI file picker, preprocesses the image, runs inference.

---

### `src/utils.py`

```python
plot_history(history)
# Plots accuracy and loss curves side-by-side.

plot_prediction(img_rgb, predicted_class, confidence, all_probs, class_names)
# Renders image + horizontal probability bar chart.
```