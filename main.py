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

        if predictor is None:
            return {
                "status": "error",
                "message": "Model not loaded"
            }

        image_bytes = await file.read()

        ai_result = predictor.predict(image_bytes)

        print("FINAL RESPONSE =", ai_result)

        if FIREBASE_ENABLED:
            try:
                log_result(ai_result)
            except Exception as e:
                print("Firebase Skip:", e)

        return ai_result

    except Exception as e:
        print("API CRASH:", e)

        return {
            "status": "error",
            "message": str(e)
        }