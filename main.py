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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

predictor = Predictor(MODEL_PATH)

@app.get("/")
def root():
    return {"status": "ok", "service": "farm-ai"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        result = predictor.predict(image_bytes)

        # =========================
        # DB 저장 (런칭 핵심)
        # =========================
        log_result(result)

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }