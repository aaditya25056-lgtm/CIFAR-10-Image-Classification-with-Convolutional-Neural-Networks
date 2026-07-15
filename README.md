# CIFAR-10 Image Classification with Convolutional Neural Networks

A convolutional neural network that classifies natural images across 10
object categories from the CIFAR-10 dataset, with dropout regularization and
confusion-matrix-based performance analysis.

## Overview

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) —
  60,000 32×32 color images across 10 classes (50,000 train / 10,000 test)
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog,
  horse, ship, truck
- **Task:** Multi-class image classification

## Architecture

```
Input (32x32x3)
   -> Conv2D(128, 3x3, padding='same', relu) -> MaxPool(2x2) -> Dropout(0.3)
   -> Conv2D(64,  3x3, padding='same', relu) -> MaxPool(2x2) -> Dropout(0.3)
   -> Conv2D(32,  3x3, padding='same', relu) -> MaxPool(2x2) -> Dropout(0.3)
   -> Flatten
   -> Dense(64, activation='relu')
   -> Dense(10, activation='softmax')
```

- **Loss:** Categorical cross-entropy
- **Optimizer:** RMSprop
- **Metric:** Accuracy
- **Epochs:** 20, batch size 128

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

CIFAR-10 is downloaded automatically via `tf.keras.datasets.cifar10.load_data()`.

## Results

Reports test-set accuracy/loss curves over 20 epochs and a confusion matrix
showing per-class classification performance.

## Known limitations

- Uses the test set as validation data during training (`validation_data=(X_test, y_test)`)
  rather than a held-out validation split — test performance during training
  isn't a fully unbiased estimate
- No data augmentation (random crop/flip), which is standard practice for
  CIFAR-10 to reduce overfitting
- No learning-rate scheduling or early stopping

## Tech stack

Python, TensorFlow, Keras, OpenCV, Scikit-learn, Seaborn, Matplotlib, NumPy
