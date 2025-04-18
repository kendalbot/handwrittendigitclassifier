# File: app/model_loader.py

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging 

#basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#absolute path of the directory containing this script (app/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
#Define  project root
MODEL_DIR_NAME = "mnist_cnn_model"
MODEL_FILENAME = "best_model.keras"
MODEL_PATH = os.path.join(project_root, MODEL_DIR_NAME, MODEL_FILENAME)

#Initialize global to None --> will be loaded by the load_prediction_model function
model = None

def load_prediction_model():
    """
    Loads the trained Keras model from the specified path.

    Sets the global 'model' variable if loading is successful.
    Logs errors if loading fails.
    """
    global model #modifying the global variable

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at: {MODEL_PATH}")
        logger.error("Cannot load the model. Please ensure the model has been trained and saved correctly.")
        model = None #model is 'None' if loading failed
        return

    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    try:
        #Load the saved model --> dummy prediction for low hanging fruit
        model = load_model(MODEL_PATH)
        logger.info("TensorFlow model loaded successfully.")
        logger.info("Performing dummy prediction to verify model integrity...")
        dummy_input = tf.zeros((1, 28, 28, 1)) 
        _ = model.predict(dummy_input) #don't care about the output just that it runs
        logger.info("Dummy prediction successful.")

    except Exception as e:
        logger.error(f"Error loading model from {MODEL_PATH}: {e}", exc_info=True)
        logger.error("Model loading failed. Predictions will not be available.")
        model = None #model is 'None' if loading failed

def get_model():
    #Currently just returns the global variable, but it provides a clear interface and could be expanded later
    # (ex. to handle reload if needed).
    return model

#direct execution (nice testing this module independently)
if __name__ == "__main__":
    print("Running model loader directly for testing...")
    load_prediction_model()
    loaded_model_instance = get_model()
    if loaded_model_instance:
        print("Model loaded successfully via direct execution.")
        loaded_model_instance.summary()
    else:
        print("Model loading failed during direct execution test.")

