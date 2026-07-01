from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os

from services.ai_service import Predictor
from services.firebase_logger import log_result

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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best.pt"
)

print("===================================")
print("MODEL PATH =", MODEL_PATH)
print("MODEL EXISTS =", os.path.exists(MODEL_PATH))
print("===================================")

predictor = Predictor(MODEL_PATH)

# =========================
# ROOT
# =========================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "farm-ai"
    }

# =========================
# PREDICT
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        result = predictor.predict(image_bytes)

        try:
            log_result(result)
        except Exception as e:
            print("Firebase Log Error:", e)

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        print(e)

        return {
            "status": "error",
            "message": str(e)
        }