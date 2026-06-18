from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2

app = FastAPI()

# =========================
# HEALTH CHECK (Render 필수)
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "chanchobi-ai-server"
    }


@app.get("/health")
def health():
    return {"status": "running"}


# =========================
# AI CORE (안정 Mock)
# =========================
def run_ai(selected_crop: str):
    return {
        "ai_crop": selected_crop,
        "disease": "healthy",
        "disease_confidence": 99.0,
        "risk": "LOW",
        "recommend": "stable mode",
        "crop_match": True
    }


# =========================
# IMAGE PREPROCESS (안정형)
# =========================
def preprocess(img):
    if img is None:
        return None
    try:
        return cv2.resize(img, (640, 640))
    except:
        return img


# =========================
# PREDICT API (Render 핵심)
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown")
):
    try:
        # 이미지 읽기
        contents = await file.read()

        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "image decode failed"}
            )

        img = preprocess(img)

        # AI 실행 (현재 안정 mock)
        result = run_ai(selected_crop)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "status": "failed"
            }
        )