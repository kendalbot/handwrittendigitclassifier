from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import logging
from .model_loader import load_prediction_model, get_model
from .preprocessing import preprocess_image
import numpy as np
import base64 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


app = FastAPI(title="Handwritten Digit Recognizer API")



@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Loading ML model...")
    load_prediction_model()

    model_instance = get_model()
    if model_instance is None:
        logger.error("Model could not be loaded on startup. API prediction endpoint might not function correctly.")
    else:
        logger.info("Model check after startup: Model appears loaded.")



@app.get("/")
async def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "API Online"}


@app.post("/predict")
async def predict_digit(
    image: UploadFile = File(None), 
    image_data: str = Form(None)    
):

    logger.info("Received request to /predict endpoint.")
    image_bytes = None

    if image:
        logger.info(f"Processing uploaded file: {image.filename}, content type: {image.content_type}")
        image_bytes = await image.read()
        logger.info(f"Read {len(image_bytes)} bytes from uploaded file.")
    elif image_data:
        logger.info("Processing base64 image data.")
        try:
            if "," in image_data:
                header, encoded = image_data.split(",", 1)
                logger.info(f"Removed data URL header: {header}")
            else:
                encoded = image_data 

            image_bytes = base64.b64decode(encoded)
            logger.info(f"Decoded {len(image_bytes)} bytes from base64 data.")
        except Exception as e:
            logger.error(f"Error decoding base64 image data: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid base64 image data provided.")
    else:

        logger.warning("No image file or base64 data provided in the request.")
        raise HTTPException(status_code=400, detail="No image file or image_data provided.")


    if image_bytes is None:
         logger.error("Image bytes are None before preprocessing.")
         raise HTTPException(status_code=500, detail="Internal server error: Failed to read image data.")

    logger.info("Calling preprocessing function...")
    processed_image_array = preprocess_image(image_bytes, MAX_FILE_SIZE_BYTES)

    if processed_image_array is None:
        logger.warning("Image preprocessing returned None. Cannot predict.")
        raise HTTPException(status_code=400, detail="Image preprocessing failed. Check image format, size, and content.")

    model = get_model()
    if model is None:
        logger.error("Model is not loaded. Cannot perform prediction.")
        raise HTTPException(status_code=503, detail="Model not available. Please try again later.")


    try:
        logger.info(f"Preprocessed array details before prediction:")
        logger.info(f"  Shape: {processed_image_array.shape}")
        logger.info(f"  dtype: {processed_image_array.dtype}")
        logger.info(f"  Min value: {np.min(processed_image_array)}")
        logger.info(f"  Max value: {np.max(processed_image_array)}")
        center_y, center_x = processed_image_array.shape[1] // 2, processed_image_array.shape[2] // 2
        logger.info(f"  Sample center pixel value: {processed_image_array[0, center_y, center_x, 0]}")
    except Exception as log_e:
        logger.error(f"Error during debug logging of processed array: {log_e}")
    #     END DEBUGGING 

    logger.info("Performing prediction with the loaded model...")
    try:
        prediction = model.predict(processed_image_array)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        logger.info(f"Prediction successful: Digit={predicted_digit}, Confidence={confidence:.4f}")

    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")

    return {
        "prediction": predicted_digit,
        "confidence": confidence
    }

