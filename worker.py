import os
import time
import json
import base64
import requests
import io
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

# ==========================
# 🌱 LOAD ENV (가장 먼저!)
# ==========================
load_dotenv()

# ==========================
# ☁️ UPSTASH CONFIG
# ==========================
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {UPSTASH_TOKEN}"
}

# ==========================
# 🤖 YOLO MODEL LOAD
# ==========================
model = YOLO("best.pt")

print("🔥 WORKER STARTED - AI ENGINE READY")


# ==========================
# 🖼️ IMAGE DECODE
# ==========================
def decode_image(image_b64):
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        print("❌ IMAGE DECODE ERROR:", e)
        return None


# ==========================
# 🧠 YOLO INFERENCE
# ==========================
def run_yolo(image):
    results = model(image)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return {
            "disease": "no_detection",
            "confidence": 0,
            "risk": "LOW"
        }

    box = boxes[0]

    conf = float(box.conf[0]) * 100
    cls = int(box.cls[0])

    label = model.names[cls]

    if conf >= 80:
        risk = "HIGH"
    elif conf >= 50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "disease": label,
        "confidence": round(conf, 2),
        "risk": risk
    }


# ==========================
# 📥 POP TASK (QUEUE)
# ==========================
def pop_task():
    try:
        url = f"{UPSTASH_URL}/brpop/ai_queue/0"
        res = requests.get(url, headers=HEADERS, timeout=30)

        data = res.json()
        result = data.get("result")

        if not result:
            return None

        task_json = result[1]
        return json.loads(task_json)

    except Exception as e:
        print("❌ POP ERROR:", e)
        return None


# ==========================
# 💾 SAVE RESULT
# ==========================
def save_result(task_id, result):
    try:
        url = f"{UPSTASH_URL}/set/result:{task_id}/{json.dumps(result)}"
        requests.post(url, headers=HEADERS, timeout=30)

    except Exception as e:
        print("❌ SAVE ERROR:", e)


# ==========================
# 🚀 MAIN LOOP
# ==========================
while True:
    try:
        task = pop_task()

        if not task:
            time.sleep(1)
            continue

        task_id = task.get("task_id")
        image_b64 = task.get("image")

        print("🔥 PROCESS TASK:", task_id)

        image = decode_image(image_b64)

        if image is None:
            save_result(task_id, {
                "status": "error",
                "message": "image decode failed"
            })
            continue

        result = run_yolo(image)

        final_result = {
            "status": "done",
            "task_id": task_id,
            "result": result,
            "farm": task.get("farm"),
            "crop": task.get("crop"),
            "timestamp": time.time()
        }

        save_result(task_id, final_result)

        print("✅ DONE:", task_id)

    except Exception as e:
        print("🔥 WORKER CRASH:", str(e))
        time.sleep(2)