import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# --- Determine Project Root and Define Output Path ---
# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assume the project root is one level up from the script directory
project_root = os.path.dirname(script_dir)
# Define the output directory relative to the project root
OUTPUT_DIR_NAME = "mnist_cnn_model"
OUTPUT_DIR = os.path.join(project_root, OUTPUT_DIR_NAME)

print("TensorFlow version:", tf.__version__)

#configuration
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
OUTPUT_DIR = "mnist_cnn_model" 
MODEL_FILENAME = "best_model.keras" #kv3
PLOT_FILENAME = "training_history.png"
EPOCHS = 50 
BATCH_SIZE = 128 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

print("Loading MNIST dataset...")
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
print(f"Initial training data shape: {x_train_full.shape}")
print(f"Initial training labels shape: {y_train_full.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

print("Preprocessing data...")

x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#grayscale so channel dimension == 1
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
    stratify=y_train_full #proportional representation
)
print(f"Final training data shape: {x_train.shape}")
print(f"Final training labels shape: {y_train.shape}")
print(f"Validation data shape: {x_val.shape}")
print(f"Validation labels shape: {y_val.shape}")

print("Data loading and preprocessing complete.")

#CNN Architecture
print("Defining CNN model architecture...")
model = Sequential([
    Input(shape=input_shape, name="input_layer"), # (28, 28, 1)

    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name="conv1"),
    BatchNormalization(name="bn1"),
    MaxPooling2D(pool_size=(2, 2), name="pool1"),
    # Dropout(0.25)
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name="conv2"),
    BatchNormalization(name="bn2"),
    MaxPooling2D(pool_size=(2, 2), name="pool2"),
    # Dropout(0.25),
    Flatten(name="flatten"),
    Dropout(0.5, name="dropout"), #Dropout before final dense layer
    Dense(10, activation='softmax', name="output_layer") #10 classes for digits 0,1,2,3,4,5,6,7,8,9
], name="mnist_cnn")
print("Model Summary:")
model.summary()

print("Compiling model...")
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Setting up data aug and callbacks...")

#Augmentation -  very subtle
datagen = ImageDataGenerator(
    rotation_range=10,       #degrees
    width_shift_range=0.1,   #fraction of total width
    height_shift_range=0.1,  #raction of total height
    zoom_range=0.1           #zoom range [1-0.1, 1+0.1]
)

#stop training when loss hasn't improved for "patience" epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, #number with no improvement EPOCHS
    verbose=1,
    restore_best_weights=True
)
model_checkpoint_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
model_checkpoint = ModelCheckpoint(
    filepath=model_checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max', 
    verbose=1
)
callbacks_list = [early_stopping, model_checkpoint]
#traininginit
print(f"Starting model training (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})...")

train_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

#epochsteps
steps_per_epoch = x_train.shape[0] // BATCH_SIZE
if x_train.shape[0] % BATCH_SIZE != 0:
    steps_per_epoch += 1 #all data is seen

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=(x_val, y_val), 
    callbacks=callbacks_list,
    verbose=1 #progress bar
)

print("Model training finished.")

print("Plotting training history...")

#check expected keys
acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
epochs_range = range(len(acc)) #actual number of epoochs run

plt.figure(figsize=(12, 5))

#PlotAcc
plt.subplot(1, 2, 1)
if acc and val_acc:
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
else:
    plt.text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center')
    plt.title('Accuracy Plot Unavailable')


#PlotLoss
plt.subplot(1, 2, 2)
if loss and val_loss:
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
else:
     plt.text(0.5, 0.5, 'Loss data not available', ha='center', va='center')
     plt.title('Loss Plot Unavailable')

plt.suptitle('Model Training History', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #title overlap fix

#Save
plot_path = os.path.join(OUTPUT_DIR, PLOT_FILENAME)
try:
    plt.savefig(plot_path)
    print(f"Training history plot saved to: {plot_path}")
except Exception as e:
    print(f"Error saving plot: {e}")

# plt.show() #display the plot. I dont think this helps right now

print("Training script finished.")