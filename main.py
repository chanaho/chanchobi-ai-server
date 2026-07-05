from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os

# =========================
# Firebase SAFE IMPORT
# =========================
try:
    from services.firebase_logger import log_result
    FIREBASE_ENABLED = True
except Exception:
    print("Firebase disabled")
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

predictor = None


# =========================
# LOAD MODEL
# =========================
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
# ROOT TEST
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "farm-ai"
    }


# =========================
# 🔥 SAFE PREDICT API (핵심)
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form("")
):
    try:
        # -------------------------
        # 1. 모델 체크
        # -------------------------
        if predictor is None:
            return {
                "status": "error",
                "message": "Model not loaded",
                "data": None
            }

        # -------------------------
        # 2. 이미지 읽기
        # -------------------------
        image_bytes = await file.read()

        # -------------------------
        # 3. AI 추론
        # -------------------------
        try:
            ai_result = predictor.predict(image_bytes)
        except Exception as e:
            print("AI ERROR:", e)
            return {
                "status": "error",
                "message": "inference failed",
                "data": None
            }

        # -------------------------
        # 4. 결과 fallback 보호
        # -------------------------
        if not ai_result:
            ai_result = {
                "crop": "unknown",
                "disease": "unknown",
                "confidence": 0,
                "risk": "UNKNOWN",
                "chemical": [],
                "method": "-",
                "warning": ""
            }

        # -------------------------
        # 5. 🔥 JSON 구조 고정 (앱 깨짐 방지 핵심)
        # -------------------------
        response = {
            "status": "success",
            "data": {
                "crop": ai_result.get("crop", crop or "unknown"),
                "disease": ai_result.get("disease", "unknown"),
                "confidence": float(ai_result.get("confidence", 0) or 0),
                "risk": ai_result.get("risk", "UNKNOWN"),
                "chemical": ai_result.get("chemical", []),
                "method": ai_result.get("method", "-"),
                "warning": ai_result.get("warning", ""),

                # 🔥 확장 필드 (앱 UI 안전)
                "interval": ai_result.get("interval", "-"),
                "riskPrediction": ai_result.get("riskPrediction", "UNKNOWN"),
                "riskScore": ai_result.get("riskScore", 0),
                "riskMessage": ai_result.get("riskMessage", "-")
            }
        }

        # -------------------------
        # 6. 로그 (옵션)
        # -------------------------
        if FIREBASE_ENABLED:
            try:
                log_result(response)
            except Exception as e:
                print("Firebase skip:", e)

        # -------------------------
        # 7. DEBUG 출력
        # -------------------------
        print("FINAL RESPONSE =", response)

        return response

    except Exception as e:
        print("API CRASH:", e)

        return {
            "status": "error",
            "message": str(e),
            "data": None
        }