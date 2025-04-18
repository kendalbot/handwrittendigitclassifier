#File: app/main.py
from fastapi import FastAPI


app = FastAPI(title="Handwritten Digits Classifier API")


#root endpoint --> GET requests
@app.get("/")
async def read_root()
    """
    Root Endpoint to check API Status
    """

    return {"message": "We online..."}





# @app.post("/predict") - Will be added later
# @app.on_event("startup") - Will be added later 