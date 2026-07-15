# CIFAR-10 Image Classification with Convolutional Neural Networks

A convolutional neural network that classifies natural images across 10
object categories from the CIFAR-10 dataset, with dropout regularization and
confusion-matrix-based performance analysis.

## Overview

- **Dataset:** [CIFAR-10]
  60,000 32×32 color images across 10 classes (50,000 train / 10,000 test)
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog,
  horse, ship, truck
- **Task:** Multi-class image classification

## Pipeline

1. Load CIFAR-10 via `tf.keras.datasets.cifar10`
2. Visualize a 3×7 grid of sample images with class labels
3. Normalize pixel values to `[0, 1]`
4. One-hot encode labels (`keras.utils.to_categorical`)
5. Build the 3-block Conv2D/MaxPool/Dropout CNN described above
6. Compile and train for 20 epochs with the test set used as
   `validation_data`
7. Plot training vs. validation accuracy and loss curves
8. Generate predictions on the test set
9. Compute and visualize a confusion matrix (Seaborn heatmap)

## Setup

```bash
pip install tensorflow keras numpy pandas opencv-python scikit-learn seaborn matplotlib
```

## Usage

```bash
jupyter notebook cifar10_cnn.ipynb
```

