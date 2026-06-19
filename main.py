import os

print("🔥 MAIN START")

from fastapi import FastAPI

print("🔥 BEFORE APP")

app = FastAPI()

print("🔥 AFTER APP")


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "server running"
    }

# force redeploy fix

# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {
        "status": "running"
    }


# =========================
# SIMPLE MOCK AI
# =========================
def run_ai(selected_crop):
    return {
        "ai_crop": selected_crop,
        "disease": "정상",
        "disease_confidence": 99.0,
        "risk": "LOW",
        "recommend": "Render 안정 테스트 단계",
        "crop_match": True
    }


# =========================
# PREPROCESS
# =========================
def preprocess(img):
    if img is None:
        return None
    return cv2.resize(img, (640, 640))


# =========================
# PREDICT
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
            return JSONResponse(
                status_code=400,
                content={"error": "image decode failed"}
            )

        img = preprocess(img)

        result = run_ai(selected_crop)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
