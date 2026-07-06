from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from PIL import Image
import random

app = FastAPI()

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
async def predict(file: UploadFile = File(...)):

    # 파일 저장
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # =========================
    # 🔥 임시 AI 결과 (추후 YOLO 교체)
    # =========================
    result = {
        "disease": random.choice(DISEASES),
        "risk": random.choice(RISK_LEVEL),
        "confidence": round(random.uniform(0.6, 0.98), 2)
    }

    return result