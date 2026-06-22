from fastapi import FastAPI
import torch

app = FastAPI()

# ==========================
# 🤖 GLOBAL MODEL
# ==========================
model = None


# ==========================
# 🚀 STARTUP (중요)
# ==========================
@app.on_event("startup")
def load_model():
    global model

    try:
        print("🔥 MODEL LOADING START...")

        # ==========================
        # YOLOv5 / YOLOv8 로딩
        # ==========================
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="best.pt",   # 👉 너 모델 파일
            force_reload=False
        )

        # confidence threshold
        model.conf = 0.4

        print("✅ MODEL LOADED SUCCESS")

    except Exception as e:
        print("❌ MODEL LOAD FAILED:", e)
        model = None


# ==========================
# 🧪 TEST API (필수 확인용)
# ==========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }