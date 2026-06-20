import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO

print("🚀 AI SERVER START")

app = FastAPI()

# =========================
# MODEL LOAD (핵심)
# =========================
model = YOLO("best.pt")

print("✅ YOLO MODEL LOADED")


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "chanchobi-ai"}


# =========================
# FALLBACK
# =========================
def fallback():
    return {
        "ai_crop": "unknown",
        "disease": "safe_mode",
        "confidence": 0,
        "risk": "LOW",
        "crop_match": False
    }


# =========================
# AI INFERENCE
# =========================
def run_ai(image_bytes):
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return fallback()

        results = model(img)

        r = results[0]

        # =========================
        # CLASSIFICATION ONLY SAFE
        # =========================
        if hasattr(r, "probs") and r.probs is not None:
            top1 = int(r.probs.top1)
            conf = float(r.probs.top1conf)

            disease_name = model.names[top1]

        else:
            # ❗ detection fallback (안 쓰는 경우 안전 처리)
            return fallback()

        crop = disease_name.split("_")[0] if "_" in disease_name else "unknown"

        if conf > 0.85:
            risk = "HIGH"
        elif conf > 0.6:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "ai_crop": crop,
            "disease": disease_name,
            "confidence": round(conf * 100, 2),
            "risk": risk,
            "crop_match": True
        }

    except Exception as e:
        print("AI ERROR:", e)
        return fallback()


# =========================
# PREDICT API
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown")
):

    try:
        contents = await file.read()
        result = run_ai(contents)

        return JSONResponse({
            "status": "success",
            "farm": farm,
            "result": result
        })

    except Exception as e:
        return JSONResponse({
            "status": "failed",
            "error": str(e),
            "result": fallback()
        })