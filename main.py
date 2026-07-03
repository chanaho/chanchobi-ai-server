from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os

from services.disease_db import get_disease_info

# =========================
# Firebase SAFE IMPORT
# =========================
try:
    from services.firebase_logger import log_result
    FIREBASE_ENABLED = True
except Exception:
    print("Firebase completely disabled")
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
# MODEL (LAZY LOAD - 중요)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")

print("MODEL:", MODEL_PATH)

predictor = None


@app.on_event("startup")
def load_model():
    global predictor

    try:
        from services.ai_service import Predictor

        print("🔥 MODEL LOADING:", MODEL_PATH)

        predictor = Predictor(MODEL_PATH)

        print("✅ MODEL LOADED SUCCESS")

    except Exception as e:
        print("❌ MODEL LOAD FAILED:", e)
        predictor = None


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
# AI PREDICT
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form("")
):
    try:

        # 0. 모델 체크
        if predictor is None:
            return {
                "status": "error",
                "message": "Model not loaded"
            }

        # 1. 이미지 읽기
        image_bytes = await file.read()

        # 2. AI 예측 (안전 처리)
        try:
            ai_result = predictor.predict(image_bytes)
        except Exception as e:
            print("MODEL ERROR:", e)
            return {
                "status": "error",
                "message": "inference failed"
            }

        if not ai_result:
            return {
                "status": "error",
                "message": "empty result"
            }

        # 3. disease 안정화
        raw_disease = ai_result.get("disease") or "unknown"

        # 4. DB 매핑
        info = get_disease_info(raw_disease)

        # 5. 안전 응답 생성
        response = {
            "status": "success",

            "crop": crop if crop else info.get("crop", "unknown"),

            "disease": info.get("name", "알 수 없음"),

            "confidence": float(ai_result.get("confidence", 0) or 0),

            "risk": info.get("risk", "UNKNOWN"),

            "chemical": info.get("chemical", []),

            "method": info.get("method", "추가 분석 필요"),

            "note": info.get("note", ""),

            "warning": info.get("warning", ""),
        }

        # 6. Firebase (완전 안전)
        if FIREBASE_ENABLED:
            try:
                log_result(response)
            except Exception as e:
                print("Firebase Skip:", e)

        return response

    except Exception as e:

        print("API CRASH:", e)

        return {
            "status": "error",
            "message": str(e)
        }