from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import time

app = FastAPI()

# =========================
# AI 모델 로드
# =========================
MODEL = YOLO("model/best.pt")
print("✅ MODEL LOADED:", MODEL.names)

# =========================
# CORS
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
# 상태 확인
# =========================
@app.get("/")
def root():
    return {
        "status": "AI SERVER RUNNING",
        "classes": MODEL.names
    }

# =========================
# AI 분석 API
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form(None)
):

    try:

        if not crop:
            crop = "unknown"

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("===================================")
        print("START PREDICT")
        print("FILE :", file_path)
        print("CROP :", crop)

        start = time.time()

        results = MODEL.predict(
            source=file_path,
            conf=0.25,
            verbose=False
        )

        elapsed = time.time() - start

        print("END PREDICT")
        print("TIME :", round(elapsed, 2), "sec")
        print("RESULT COUNT :", len(results))

        result = results[0]

        print("========== DEBUG ==========")
        print("MODEL NAMES :", MODEL.names)
        print("BOXES :", result.boxes)
        print("BOX COUNT :", len(result.boxes))
        print("===========================")

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

        print("CLASS :", cls)
        print("DISEASE :", disease)
        print("CONF :", conf)

        return {
            "crop": crop,
            "disease": disease,
            "confidence": round(conf, 2),
            "risk": "LOW"
        }

    except Exception as e:
        import traceback

        print("######## ERROR ########")
        traceback.print_exc()

        return {
            "status": "failed_safe",
            "error": str(e)
        }