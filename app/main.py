# File: app/main.py

from fastapi import FastAPI
import logging
from .model_loader import load_prediction_model, get_model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Handwritten Digit Recognizer API")


# --- Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Loading ML model...")
    load_prediction_model() # Call the function from model_loader
    model_instance = get_model()
    if model_instance is None:
        logger.error("Model could not be loaded on startup. API prediction endpoint might not function correctly.")
    else:
        logger.info("Model check after startup: Model appears loaded.")

#root endpoint --> GET requests
@app.get("/")
async def read_root():
    """
    Root Endpoint to check API Status
    """

    return {"message": "We online..."}





# @app.post("/predict") - Will be added later
# @app.on_event("startup") - Will be added later 