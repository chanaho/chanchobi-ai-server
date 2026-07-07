from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
from PIL import Image
import random

app = FastAPI()

MODEL = YOLO("model/best.pt")
print("✅ MODEL LOADED:", MODEL.names)

# =========================
# 🔥 CORS (앱 연결 필수)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 업로드 폴더
# =========================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# 🔥 임시 클래스 (AI 대체)
# =========================
DISEASES = [
    "감귤_궤양병",
    "고추_탄저병",
    "키위_점무늬병",
    "unknown"
]

RISK_LEVEL = ["LOW", "MEDIUM", "HIGH"]

# =========================
# 상태 확인
# =========================
@app.get("/")
def root():
    return {"status": "AI SERVER RUNNING"}

# =========================
# 🔥 AI 분석 API
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form(None)
):

    try:
        if not crop:
            crop = "unknown"

        file_path = f"{UPLOAD_DIR}/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = MODEL.predict(file_path, conf=0.25, verbose=False)
        result = results[0]    

        if len(result.boxes) == 0:
           return {
              "crop": crop,
              "disease": "알 수 없음",
              "confidence": 0.0,
              "risk": "UNKNOWN"
           }

        cls = int(result.boxes.cls[0])
        conf = float(result.boxes.conf[0])
        disease = MODEL.names[cls]

        return {
           "crop": crop,
           "disease": disease,
           "confidence": round(conf, 2),
           "risk": "LOW"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed_safe"
        }