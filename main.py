from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os

from services.ai_service import Predictor

# =========================
# Firebase SAFE IMPORT
# =========================
try:
    from services.firebase_logger import log_result
    FIREBASE_ENABLED = True
except:
    print("⚠ Firebase Disabled")
    FIREBASE_ENABLED = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")

print("MODEL:", MODEL_PATH)

predictor = Predictor(MODEL_PATH)

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "ok"}

# =========================
# PREDICT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        result = predictor.predict(image_bytes)

        # Firebase 안전 처리
        if FIREBASE_ENABLED:
            try:
                log_result(result)
            except:
                pass

        return result

    except Exception as e:
        print("API ERROR:", e)
        return {
            "status": "error",
            "message": str(e)
        }