from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

def run_ai(selected_crop):
    return {
        "ai_crop": selected_crop,
        "disease": "healthy",
        "disease_confidence": 99.0,
        "risk": "LOW",
        "recommend": "test mode",
        "crop_match": True
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown")
):
    await file.read()
    result = run_ai(selected_crop)
    return JSONResponse(content=result)