from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os
import torch
import asyncio

app = FastAPI()

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
# LOCK (동시 요청 방지)
# =========================
lock = asyncio.Lock()

# =========================
# MODEL LOAD
# =========================
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

print("CURRENT DIR =", os.getcwd())
print("FILES =", os.listdir())

MODEL_PATH = "best.pt"

print("=" * 60)
print("MODEL PATH =", os.path.abspath(MODEL_PATH))

try:
    model = YOLO(MODEL_PATH)

    print("🔥 MODEL LOADED SUCCESS")
    print("MODEL TYPE =", type(model))
    print("MODEL NAMES =", model.names)
    print("NUMBER OF CLASSES =", len(model.names))

    if hasattr(model, "task"):
        print("MODEL TASK =", model.task)

except Exception as e:
    model = None
    print("❌ MODEL LOAD FAILED:", str(e))

print("=" * 60)
# =========================
# DISEASE DB
# =========================
DISEASE_DB = {
    "anthracnose": {
        "name": "탄저병",
        "risk": "HIGH",
        "chemical": ["아족시스트로빈", "디페노코나졸"],
        "rotation": "교호살포",
        "note": "고온다습 주의",
        "warning": "확산 빠름",
    },
    "scab": {
        "name": "갈색무늬병",
        "risk": "MEDIUM",
        "chemical": ["카벤다짐"],
        "rotation": "교호살포",
        "note": "강우 후 증가",
        "warning": "관리 필요",
    },
    "aphid": {
        "name": "진딧물",
        "risk": "LOW",
        "chemical": ["이미다클로프리드"],
        "rotation": "연속사용 금지",
        "note": "초기 방제 중요",
        "warning": "꿀벌 주의",
    },
}

# =========================
# INTERPRETER
# =========================
def interpret_result(label: str):
    data = DISEASE_DB.get(label.lower())

    if not data:
        return {
            "disease": label,
            "risk": "UNKNOWN",
            "chemical": [],
            "rotation": "확인 필요",
            "note": "추가 학습 필요",
            "warning": "정밀 진단 권장",
        }

    return {
        "disease": data["name"],
        "risk": data["risk"],
        "chemical": data["chemical"],
        "rotation": data["rotation"],
        "note": data["note"],
        "warning": data["warning"],
    }

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}


# =========================
# PREDICT
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown"),
):

    async with lock:

        print("===== PREDICT REQUEST =====")
        print("farm =", farm)
        print("crop =", selected_crop)

        if model is None:
            return {"status": "error", "message": "model not loaded"}

        image_bytes = await file.read()
        print("image size =", len(image_bytes))

        if not image_bytes:
            return {"status": "error", "message": "empty image"}

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"status": "error", "message": f"decode error: {e}"}

        image.thumbnail((512, 512))

        print("🔥 PREDICT START")

        try:
            with torch.inference_mode():
                results = model.predict(
                    source=image,
                    imgsz=640,
                    conf=0.05,
                    iou=0.50,
                    max_det=10,
                    device="cpu",
                    verbose=False,
                )

            print("YOLO finished")
            print("results count =", len(results))

            print("image shape:", img.shape)
            print("model classes:", model.names)

            if len(results) > 0:
                r = results[0]

                print("boxes:", r.boxes)
                print("box count:", len(r.boxes) if r.boxes else 0)

                if r.boxes is not None and len(r.boxes) > 0:
                    print("conf:", r.boxes.conf)
                    print("cls:", r.boxes.cls)                              
                else:
                    print("NO DETECTIONS")

            else:
                 print("NO RESULTS")

        except Exception as e:
            print("YOLO ERROR =", str(e))
            return {"status": "error", "message": str(e)}

        # =========================
        # EMPTY RESULT
        # =========================
        if not results or len(results) == 0:
            return {
                "status": "success",
                "farm": farm,
                "ai_crop": selected_crop,
                "disease": "정상",
                "confidence": 0,
                "risk": "LOW",
                "chemical": [],
                "rotation": "",
                "note": "검출 없음",
                "warning": "",
            }

        r = results[0]

        # =========================
        # NO BOXES
        # =========================
        if r.boxes is None or len(r.boxes) == 0:
            return {
                "status": "success",
                "farm": farm,
                "ai_crop": selected_crop,
                "disease": "정상",
                "confidence": 0,
                "risk": "LOW",
                "chemical": [],
                "rotation": "",
                "note": "병해충 미검출",
                "warning": "",
            }

        # =========================
        # TOP RESULT
        # =========================
        box = r.boxes[0]
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = round(float(box.conf[0]) * 100, 1)

        print("label =", label)
        print("confidence =", confidence)

        decision = interpret_result(label)

        return {
            "status": "success",
            "farm": farm,
            "ai_crop": selected_crop,
            "disease": decision["disease"],
            "confidence": confidence,
            "risk": decision["risk"],
            "chemical": decision["chemical"],
            "rotation": decision["rotation"],
            "note": decision["note"],
            "warning": decision["warning"],
            "ai_label": label,
        }