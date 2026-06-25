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
# MODEL LOAD (안정화)
# ==========================
model = None

try:
    model = YOLO("best.pt")
    print("🔥 MODEL LOADED SUCCESS")
except Exception as e:
    print("❌ MODEL LOAD FAILED:", e)

# ==========================
# 농약 + 방제 DB
# ==========================
DISEASE_DB = {
    "anthracnose": {
        "name": "탄저병",
        "risk": "HIGH",
        "chemical": [
            "아족시스트로빈 계열",
            "디페노코나졸 계열",
            "프로피코나졸 계열",
        ],
        "rotation": "교호살포 필수 (같은 계열 연속 금지)",
        "note": "저항성 발생 매우 높음",
        "warning": "고온기 약해 가능",
    },

    "scab": {
        "name": "갈색무늬병/반점병",
        "risk": "MEDIUM",
        "chemical": [
            "카벤다짐",
            "테부코나졸",
            "트리플록시스트로빈",
        ],
        "rotation": "계통 교호살포 권장",
        "note": "강우 후 즉시 방제 필요",
        "warning": "과다 살포 시 잎 황화",
    },

    "aphid": {
        "name": "진딧물",
        "risk": "LOW",
        "chemical": [
            "이미다클로프리드",
            "아세타미프리드",
        ],
        "rotation": "2회 이상 연속 사용 금지",
        "note": "초기 방제가 핵심",
        "warning": "꿀벌 피해 주의",
    },
}

# ==========================
# AI RESULT → 농업 의사결정
# ==========================
def interpret_result(label: str):
    data = DISEASE_DB.get(label.lower())

    if not data:
        return {
            "disease": label,
            "risk": "UNKNOWN",
            "chemical": ["등록 약제 확인 필요"],
            "rotation": "정보 없음",
            "note": "AI 추가 학습 필요",
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
# HEALTH CHECK
# ==========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "pest-ai-complete",
        "model_loaded": model is not None,
    }

# ==========================
# MAIN ANALYSIS
# ==========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown"),
):
    try:
        if model is None:
            return {
                "status": "error",
                "message": "AI 모델 미로드",
            }

        image_bytes = await file.read()

        if not image_bytes:
            return {
                "status": "error",
                "message": "empty image",
            }

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        print("🔥 PREDICT START")

        try:
            print("🔥 START YOLO")
            print(f"🔥 IMAGE TYPE = {type(image)}")

            print("🔥 BEFORE MODEL")

            results = model(image)

            print("🔥 YOLO SKIP TEST START")
            results = None  # YOLO 완전 우회


            print("🔥 YOLO SKIPPED")

        except Exception as e:
            import traceback

            print("🔥 YOLO ERROR:", str(e))
            traceback.print_exc()

            raise e

            return {
                "status": "error",
                "message": f"YOLO inference failed: {str(e)}"
            }

        boxes = results[0].boxes

        print(
            "🔥 BOX COUNT:",
            0 if boxes is None else len(boxes)
        )

        # 병해 미검출
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

        # 최고 신뢰도 결과
        box = boxes[0]

        cls = int(box.cls[0])
        label = model.names[cls]

        confidence = round(
            float(box.conf[0]) * 100,
            1,
        )

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
        print("🔥 PREDICT ERROR:", str(e))

        return {
            "status": "error",
            "message": str(e),
        }