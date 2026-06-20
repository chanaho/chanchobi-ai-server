import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO

model = YOLO("best.pt")

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
        import numpy as np
        import cv2

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return fallback()

        # 🔥 YOLO 직접 사용 (핵심)
        results = model(img)

        if results and len(results) > 0:
            r = results[0]

            if r.boxes and len(r.boxes) > 0:
                cls = int(r.boxes.cls[0])
                conf = float(r.boxes.conf[0])

                return {
                    "ai_crop": "test_crop",
                    "disease": model.names.get(cls, "unknown"),
                    "confidence": round(conf * 100, 2),
                    "risk": "HIGH" if conf > 0.7 else "MEDIUM",
                    "crop_match": True
                }

        return fallback()

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

        # 🚨 LOG 추가 (Render에서 보임)
        print("🔥 FILE NAME:", file.filename)
        print("🔥 FILE SIZE:", len(contents))

        result = run_ai(contents)

        return JSONResponse({
            "status": "success",
            "farm": farm,
            "result": result
        })

    except Exception as e:
        print("❌ ERROR:", e)

        return JSONResponse({
            "status": "failed",
            "error": str(e),
            "result": fallback()
        })