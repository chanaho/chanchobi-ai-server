from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import time
import traceback

app = FastAPI()

# =========================
# AI 모델 로드
# =========================
print("🔥 LOADING MODEL...")

MODEL = YOLO("model/best.pt")

print("✅ MODEL LOADED")
print("MODEL TASK :", MODEL.task)
print("MODEL NAMES :", MODEL.names)

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
# ROOT
# =========================
@app.get("/")
def root():
    return {
        "status": "AI SERVER RUNNING",
        "classes": MODEL.names
    }

# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok"
    }

# =========================
# AI 분석
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form(None)
):

    try:

        print("\n====================================")
        print("🔥 NEW REQUEST")

        if crop is None or crop == "":
            crop = "unknown"

        print("CROP :", crop)

        filename = file.filename or f"{int(time.time())}.jpg"

        file_path = os.path.join(
            UPLOAD_DIR,
            filename
        )

        print("SAVE :", file_path)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("SAVE OK")

        # ---------------------
        # YOLO 시작
        # ---------------------
        print("START PREDICT")

        t0 = time.time()

        results = MODEL(
            file_path,
            imgsz=256,
            conf=0.25,
            max_det=1,
            verbose=False
            half=False,
            augment=False
        )

        sec = round(time.time() - t0, 2)

        print("END PREDICT")
        print("TIME :", sec, "sec")
        print("RESULT LEN :", len(results))

        if len(results) == 0:
            print("NO RESULT")

            return {
                "crop": crop,
                "disease": "알 수 없음",
                "confidence": 0.0,
                "risk": "UNKNOWN"
            }

        result = results[0]

        print("BOX COUNT :", len(result.boxes))

        if len(result.boxes) == 0:

            print("NO BOX")

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

    except Exception:

        print("######## EXCEPTION ########")
        traceback.print_exc()

        return {
            "status": "failed_safe",
            "error": traceback.format_exc()
        }