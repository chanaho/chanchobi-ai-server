import os
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

# =========================
# TEST BOOT STRAP (여기 위치 중요)
# =========================
print("🔥 MAIN START")
from fastapi import FastAPI
print("🔥 BEFORE APP")

app = FastAPI()

print("🔥 AFTER APP")

# =========================
# HEALTH CHECK (필수)
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "stable-ai"}


# =========================
# SAFE FALLBACK
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
# LAZY AI LOAD (핵심)
# =========================
def run_ai(image_bytes):
    """
    ⚠️ 여기서만 AI 로딩 (요청 시 실행)
    """

    try:
        import numpy as np
        import cv2

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return fallback()

        h, w = img.shape[:2]

        return {
            "ai_crop": "test_crop",
            "disease": "detected",
            "confidence": 70.0,
            "risk": "MEDIUM" if w > 500 else "LOW",
            "crop_match": True
        }

    except Exception as e:
        print("AI ERROR:", e)
        return fallback()


# =========================
# PREDICT API (안정 핵심)
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