from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import traceback

app = FastAPI()

# ==========================
# CORS
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# MODEL LOAD
# ==========================
try:
    model = YOLO("best.pt")
    print("🔥 MODEL LOADED SUCCESS")
except Exception as e:
    model = None
    print("❌ MODEL LOAD FAILED:", e)

# ==========================
# DISEASE DB
# ==========================
DISEASE_DB = {
    "anthracnose": {
        "name": "탄저병",
        "risk": "HIGH",
        "chemical": ["아족시스트로빈", "디페노코나졸", "프로피코나졸"],
        "rotation": "교호살포 필수",
        "note": "저항성 높음",
        "warning": "고온기 주의",
    },
    "scab": {
        "name": "갈색무늬병",
        "risk": "MEDIUM",
        "chemical": ["카벤다짐", "테부코나졸"],
        "rotation": "교호살포",
        "note": "강우 후 발생",
        "warning": "과다살포 주의",
    },
    "aphid": {
        "name": "진딧물",
        "risk": "LOW",
        "chemical": ["이미다클로프리드", "아세타미프리드"],
        "rotation": "연속 사용 금지",
        "note": "초기 방제 중요",
        "warning": "꿀벌 주의",
    },
}

# ==========================
# INTERPRETER
# ==========================
def interpret_result(label: str):
    data = DISEASE_DB.get(label.lower())

    if not data:
        return {
            "disease": label,
            "risk": "UNKNOWN",
            "chemical": ["정보 없음"],
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

# ==========================
# ROOT
# ==========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

# ==========================
# PREDICT (STEP1 STABLE)
# ==========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown"),
):

    if model is None:
        return {"status": "error", "message": "AI 모델 미로드"}

    image_bytes = await file.read()
    if not image_bytes:
        return {"status": "error", "message": "empty image"}

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"status": "error", "message": f"image decode error: {str(e)}"}

    image.thumbnail((512, 512))  # 🔥 최종 안정 크기

    print("🔥 PREDICT START")

    # ==========================
    # YOLO SAFE MODE (FINAL)
    # ==========================
    try:
        results = model.predict(
            source=image,
            imgsz=416,        # 🔥 최종 안정값
            conf=0.30,
            iou=0.50,
            max_det=5,        # 🔥 과부하 방지
            device="cpu",
            verbose=False,
        )

    except Exception as e:
        print("🔥 YOLO CRASH:", str(e))
        return {
            "status": "error",
            "message": "YOLO inference failed safely"
        }

    # ==========================
    # RESULT SAFE CHECK
    # ==========================
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

    # ==========================
    # TOP RESULT ONLY
    # ==========================
    box = r.boxes[0]

    cls = int(box.cls[0])
    label = model.names[cls]
    confidence = round(float(box.conf[0]) * 100, 1)

    decision = interpret_result(label)

    print("🔥 FINAL RESULT:", label, confidence)

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