from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2

# =========================
# APP 먼저 생성 (핵심)
# =========================
app = FastAPI()

print("🚀 AI SERVER START")

# =========================
# YOLO LOAD
# =========================
model = YOLO("best.pt")

print("✅ YOLO MODEL LOADED")


# =========================
# ROOT TEST
# =========================
@app.get("/")
def root():
    return {"status": "ok"}


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

        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "failed", "error": "image decode failed"}

        results = model(img)
        r = results[0]

        if r.boxes and len(r.boxes) > 0:
            cls = int(r.boxes.cls[0])
            conf = float(r.boxes.conf[0])

            return {
                "status": "success",
                "farm": farm,
                "result": {
                    "disease": model.names.get(cls, "unknown"),
                    "confidence": round(conf * 100, 2),
                    "risk": "HIGH" if conf > 0.7 else "MEDIUM"
                }
            }

        return {
            "status": "success",
            "farm": farm,
            "result": {
                "disease": "no_detection",
                "confidence": 0,
                "risk": "LOW"
            }
        }

    except Exception as e:
        return JSONResponse({
            "status": "failed",
            "error": str(e)
        })