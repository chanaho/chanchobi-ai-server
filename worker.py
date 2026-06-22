import base64
import requests
from PIL import Image
import io
from ultralytics import YOLO

# ==========================
# ☁️ UPSTASH REDIS (REST)
# ==========================
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

# ==========================
# 🤖 YOLO MODEL LOAD
# ==========================
model = YOLO("best.pt")

print("🔥 WORKER STARTED - AI READY")


# ==========================
# 🖼️ IMAGE DECODE
# ==========================
def decode_image(image_b64):
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


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
# 🔥 GET TASK (QUEUE POP)
# ==========================
def pop_task():
    url = f"{UPSTASH_URL}/brpop/ai_queue/0"

    headers = {
        "Authorization": f"Bearer {UPSTASH_TOKEN}"
    }

    res = requests.get(url, headers=headers)
    data = res.json()

    result = data.get("result")

    if not result:
        return None

    # Upstash format: [queue_name, value]
    task_json = result[1]
    return json.loads(task_json)


# ==========================
# 📦 SAVE RESULT
# ==========================
def save_result(task_id, result):
    url = f"{UPSTASH_URL}/set/result:{task_id}/{json.dumps(result)}"

    headers = {
        "Authorization": f"Bearer {UPSTASH_TOKEN}"
    }

    requests.post(url, headers=headers)


# ==========================
# 🚀 MAIN LOOP
# ==========================
while True:
    try:
        task = pop_task()

        if not task:
            time.sleep(1)
            continue

        task_id = task["task_id"]
        print("🔥 PROCESS:", task_id)

        image = decode_image(task["image"])
        result = run_yolo(image)

        final_result = {
            "status": "done",
            "result": result,
            "farm": task.get("farm"),
            "crop": task.get("crop")
        }

        save_result(task_id, final_result)

        print("✅ DONE:", task_id)

    except Exception as e:
        print("🔥 WORKER ERROR:", str(e))
        time.sleep(2)