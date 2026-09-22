# CIFAR-10 Image Classifier 🖼️

A Convolutional Neural Network (CNN) built with PyTorch that classifies images from the CIFAR-10 dataset into 10 categories, achieving ~**87.75% validation accuracy** in 25 epochs.


## Result

The submitted Kaggle prediction achieved **0.90050 accuracy**.

## 📌 Description

This project implements a deep CNN trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset — a standard benchmark dataset containing 60,000 32×32 color images across 10 classes. The model uses a VGG-style architecture with three convolutional blocks, batch normalization, dropout regularization, and fully connected layers.

## Demo

![Project demo](assets/demo.gif)

---

## 🗂️ Classes

| Label | Class |
|-------|-------|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

---

## 🏗️ Model Architecture

```
Input (3 × 32 × 32)
│
├── Conv Block 1
│   ├── Conv2d(3 → 64) + BatchNorm + ReLU
│   ├── Conv2d(64 → 64) + BatchNorm + ReLU
│   ├── MaxPool2d(2×2)
│   └── Dropout2d(0.2)
│
├── Conv Block 2
│   ├── Conv2d(64 → 128) + BatchNorm + ReLU
│   ├── Conv2d(128 → 128) + BatchNorm + ReLU
│   ├── MaxPool2d(2×2)
│   └── Dropout2d(0.3)
│
├── Conv Block 3
│   ├── Conv2d(128 → 256) + BatchNorm + ReLU
│   ├── Conv2d(256 → 256) + BatchNorm + ReLU
│   ├── MaxPool2d(2×2)
│   └── Dropout2d(0.4)
│
└── Fully Connected
    ├── Flatten → Linear(4096 → 512) + BatchNorm + ReLU + Dropout(0.3)
    └── Linear(512 → 10)
```

---

## ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (default lr) |
| Loss Function | CrossEntropyLoss |
| Epochs | 25 |
| Batch Size | 64 |
| num_workers | 8 |
| Device | CUDA (GPU) |
| Best Val Accuracy | **87.75%** |

---

## 🔄 Data Augmentation

```python
transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

---

## 🚀 Getting Started

### Requirements

```bash
pip install torch torchvision matplotlib numpy pandas
```

### Run the Notebook

```bash
jupyter notebook CIFAR-10_Image_Classifier.ipynb
```

The dataset will be automatically downloaded on first run.

---

## 🔍 Single Image Prediction

After training, you can test any image from the test set by index (0–9999):

```python
predict_single(320)
```

**Example output:**
```
True Label  : cat
Predicted   : cat
Confidence  : 99.87%
```

---

## 📈 Training Results

| Epoch | Loss | Val Acc |
|-------|------|---------|
| 1 | 1.0231 | 85.13% |
| 5 | 0.2790 | 86.58% |
| 10 | 0.2147 | 87.34% |
| 15 | 0.1756 | 86.76% |
| 20 | 0.1432 | 86.82% |
| 25 | 0.1257 | **87.75%** |

---

## 📁 Project Structure

```
├── CIFAR-10_Image_Classifier.ipynb   # Main notebook
├── best_model.pth                    # Saved best model weights
├── data/                             # CIFAR-10 dataset (auto-downloaded)
└── README.md
```

---

## 📝 Notes

- The best model weights are automatically saved to `best_model.pth` during training whenever validation accuracy improves.
- Augmentations are applied on-the-fly each epoch, giving the model varied views of each image without increasing dataset size.
