import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os

print("TensorFlow version:", tf.__version__)

#configuration
VALIDATION_SPLIT = 0.2 # Use 20% of training data for validation
RANDOM_SEED = 42 # For reproducible splits
OUTPUT_DIR = "mnist_cnn_model" # Directory to save the model

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading MNIST dataset...")
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
print(f"Initial training data shape: {x_train_full.shape}")
print(f"Initial training labels shape: {y_train_full.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

print("Preprocessing data...")
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

IMG_ROWS, IMG_COLS = 28, 28
x_train_full = x_train_full.reshape(x_train_full.shape[0], IMG_ROWS, IMG_COLS, 1)
x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
input_shape = (IMG_ROWS, IMG_COLS, 1)

print(f"Reshaped training data shape: {x_train_full.shape}")
print(f"Reshaped test data shape: {x_test.shape}")

print(f"Splitting training data (Validation split: {VALIDATION_SPLIT})...")
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    stratify=y_train_full 
)
print(f"Final training data shape: {x_train.shape}")
print(f"Final training labels shape: {y_train.shape}")
print(f"Validation data shape: {x_val.shape}")
print(f"Validation labels shape: {y_val.shape}")

print("Data loading and preprocessing complete.")
