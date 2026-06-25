from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

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
model = None

try:
    model = YOLO("best.pt")
    print("🔥 MODEL LOADED SUCCESS")
except Exception as e:
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
# PREDICT (FINAL SAFE VERSION)
# ==========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown"),
):
    try:
        # --------------------------
        # MODEL CHECK
        # --------------------------
        if model is None:
            return {
                "status": "error",
                "message": "AI 모델 미로드",
            }

        # --------------------------
        # IMAGE LOAD
        # --------------------------
        image_bytes = await file.read()

        if not image_bytes:
            return {
                "status": "error",
                "message": "empty image",
            }

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print("🔥 PREDICT START")

        # ======================================================
        # 🔥 YOLO SAFE MODE (현재는 안정화용)
        # ======================================================
        USE_YOLO = False  # 👉 여기만 True로 바꾸면 YOLO 실행

        print("🔥 YOLO MODE:", "ON" if USE_YOLO else "OFF")

        # ==========================
        # SKIP MODE
        # ==========================
        if not USE_YOLO:
            print("🔥 SKIP MODE RESPONSE")

            return {
                "status": "success",
                "farm": farm,
                "ai_crop": selected_crop,
                "disease": "테스트모드",
                "confidence": 0,
                "risk": "LOW",
                "chemical": [],
                "rotation": "",
                "note": "YOLO SKIP MODE",
                "warning": "",
            }

        # ==========================
        # YOLO REAL MODE
        # ==========================
        print("🔥 YOLO REAL START")

        results = model(image)

        if results is None or len(results) == 0:
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

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
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

        box = boxes[0]
        cls = int(box.cls[0])
        label = model.names[cls]

        confidence = round(float(box.conf[0]) * 100, 1)

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

    except Exception as e:
        import traceback
        print("🔥 PREDICT ERROR:", str(e))
        traceback.print_exc()

        return {
            "status": "error",
            "message": str(e),
        }