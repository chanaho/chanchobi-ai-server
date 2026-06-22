from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import base64
import os
import requests

app = FastAPI()

# ==========================
# 🌐 CORS
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# ☁️ UPSTASH REDIS (REST)
# ==========================
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")


# ==========================
# 🤖 MODEL SAFE LOAD (핵심 추가)
# ==========================
model = None
model_loaded = False

try:
    # YOLO or AI 모델 로드
    from ultralytics import YOLO
    model = YOLO("best.pt")
    model_loaded = True
    print("🔥 MODEL LOADED SUCCESS")
except Exception as e:
    model = None
    model_loaded = False
    print("❌ MODEL LOAD FAILED:", e)


# ==========================
# 🔥 UPSTASH PUSH
# ==========================
def push_queue(task):
    try:
        url = f"{UPSTASH_URL}/lpush/ai_queue/{json.dumps(task)}"

        headers = {
            "Authorization": f"Bearer {UPSTASH_TOKEN}"
        }

        res = requests.post(url, headers=headers)

        print("🔥 UPSTASH PUSH:", res.status_code)

        return res.json()

    except Exception as e:
        print("🔥 PUSH ERROR:", e)
        return None


# ==========================
# 🚀 HEALTH CHECK
# ==========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "ai-api-server",
        "model_loaded": model_loaded
    }


# ==========================
# 🚀 PREDICT (SAFE QUEUE ONLY)
# ==========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown")
):

    try:
        task_id = str(uuid.uuid4())

        image_bytes = await file.read()

        if not image_bytes:
            return {
                "status": "error",
                "message": "empty image"
            }

        image_b64 = base64.b64encode(image_bytes).decode()

        task = {
            "task_id": task_id,
            "image": image_b64,
            "farm": farm,
            "crop": selected_crop
        }

        push_queue(task)

        print("🔥 QUEUED:", task_id)

        return {
            "status": "queued",
            "task_id": task_id,
            "model_loaded": model_loaded   # 🔥 중요: 상태 확인용
        }

    except Exception as e:
        print("🔥 PREDICT ERROR:", str(e))

        return {
            "status": "error",
            "message": str(e)
        }


# ==========================
# 📦 RESULT CHECK
# ==========================
@app.get("/result/{task_id}")
def get_result(task_id: str):

    try:
        url = f"{UPSTASH_URL}/get/result:{task_id}"

        headers = {
            "Authorization": f"Bearer {UPSTASH_TOKEN}"
        }

        res = requests.get(url, headers=headers)

        data = res.json()

        result = data.get("result")

        if not result:
            return {
                "status": "processing",
                "task_id": task_id
            }

        return json.loads(result)

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }