import os
import psutil

# Render 메모리 및 설정 최적화
os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from fastapi import FastAPI, UploadFile, File, Form
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort

import json
import time
import traceback
import gc
import cv2
import numpy as np

from services.disease_db import get_disease_info

cv2.setNumThreads(1)

app = FastAPI()

# Firebase 초기화

if not firebase_admin._apps:
    firebase_json = os.environ.get("FIREBASE_KEY")
    if firebase_json:
        cred = credentials.Certificate(
            json.loads(firebase_json)
        )
        firebase_admin.initialize_app(
            cred
        )
    elif os.path.exists("firebase-key.json"):
        cred = credentials.Certificate(
            "firebase-key.json"
        )
        firebase_admin.initialize_app(
            cred
        )
    else:
        print("⚠ FIREBASE 인증키 없음")

if firebase_admin._apps:

    db = firestore.client()

else:

    db = None


# =========================
# AI 모델 로드
# =========================

import gc
import psutil
import onnxruntime as ort

MODEL_PATH = "models/chanchobi_cls_best.onnx"

CLASS_NAMES = [
    "고추_정상",
    "고추_탄저병",
    "블랙커런트_병징",
    "블랙커런트_정상",
    "블루베리_병징",
    "블루베리_정상",
    "사과_병징",
    "사과_정상",
    "사과_탄저병",
    "아로니아_병징",
    "아로니아_정상",
    "아로니아_진딧물",
    "자두_병징",
    "자두_잉크병",
    "자두_정상",
    "자두_진딧물",
    "한라봉_병징",
    "한라봉_정상"
]

if not os.path.exists(MODEL_PATH):
    print(
        "❌ MODEL FILE NOT FOUND :",
        MODEL_PATH
    )
    exit()

session = None


def get_model():
    global session

    if session is None:

        print("🔥 LOADING ONNX MODEL")

        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        process = psutil.Process(os.getpid())

        print(
            "MEMORY AFTER ONNX LOAD :",
            round(
                process.memory_info().rss / 1024 / 1024,
                1
            ),
            "MB"
        )

        print("🔥 ONNX MODEL LOADED")

    return session
    
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
# Upload
# =========================

UPLOAD_DIR = "uploads"


os.makedirs(

    UPLOAD_DIR,

    exist_ok=True

)

# =========================
# ROOT
# =========================
@app.get("/")
def root():

    get_model()

    return {

        "status": "AI SERVER RUNNING",

        "classes": CLASS_NAMES

    }

@app.get("/health")
def health():

    return {

        "status": "ok"

    }

# =========================
# PREDICT
# =========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form(None)
):

    print("🔥 PREDICT START")

    try:

        print("==============================")
        print("🔥 NEW REQUEST")
        print("RECEIVED CROP :", crop)

        if crop is None:
            crop = ""

        crop = crop.strip()
        print("FINAL CROP :", crop)

        contents = await file.read()

        img = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:

            print("❌ IMAGE DECODE FAIL")

            return {
                "success": False,
                "crop": crop,
                "disease": "이미지 읽기 실패",
                "confidence": 0,
                "risk": "UNKNOWN"
            }

        if img.flags.writeable:
            img = img.copy()

        print("✅ IMAGE DECODE SUCCESS")

        h, w = img.shape[:2]

        print(
            "IMAGE SIZE :",
            w,
            "x",
            h
        )

        img = cv2.resize(
            img,
            (416, 416),
            interpolation=cv2.INTER_AREA
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        input_tensor = img.astype(np.float32) / 255.0

        input_tensor = np.transpose(
            input_tensor,
            (2, 0, 1)
        )

        input_tensor = np.expand_dims(
            input_tensor,
            axis=0
        )

        print(
            "FINAL IMAGE :",
            img.shape
        )

        # =====================
        # ONNX CLASSIFICATION
        # =====================

        print("🔥 BEFORE ONNX")

        t1 = time.time()

        session = get_model()

        input_name = session.get_inputs()[0].name

        outputs = session.run(
            None,
            {
                input_name: input_tensor
            }
        )

        t2 = time.time()

        elapsed = round(
            t2 - t1,
            2
        )

        print(
            "ONNX TIME :",
            elapsed
        )

        all_probs = outputs[0][0]

        if elapsed > 30:
            return {
                "success": False,
                "crop": crop,
                "disease": "분석시간초과",
                "confidence": 0,
                "risk": "UNKNOWN",
                "time": elapsed
            }

        # =====================
        # RESULT
        # =====================

        print("==============================")
        print("TOP5 PREDICTIONS")

        top5 = np.argsort(all_probs)[::-1][:5]

        for idx in top5:

            print(
                f"{idx:2d}",
                CLASS_NAMES[idx],
                f"{float(all_probs[idx]) * 100:.2f}%"
            )

        print("==============================")

        # crop별 허용 클래스
        allowed_classes = []

        if crop == "고추":
            allowed_classes = [0, 1]

        elif crop == "블랙커런트":
            allowed_classes = [2, 3]

        elif crop == "블루베리":
            allowed_classes = [4, 5]

        elif crop == "사과":
            allowed_classes = [6, 7, 8]

        elif crop == "아로니아":
            allowed_classes = [9, 10, 11]

        elif crop == "자두":
            allowed_classes = [12, 13, 14, 15]

        elif crop == "한라봉":
            allowed_classes = [16, 17]

        else:

            print("⚠ 알 수 없는 crop :", crop)

            allowed_classes = list(
                range(len(CLASS_NAMES))
            )

        print(
            "ALLOWED :",
            allowed_classes
        )

        # 허용 클래스 중 최고 확률 찾기
        best_score = -1.0
        cls_id = -1

        for idx in allowed_classes:

            score = float(all_probs[idx])

            if score > best_score:
                best_score = score
                cls_id = idx

        confidence = best_score

        disease = CLASS_NAMES[cls_id]

        print("==============================")

        print(
            "TOP CLASS :",
            cls_id
        )

        print(
            "TOP NAME :",
            disease
        )

        print(
            "TOP CONF :",
            round(confidence * 100, 2)
        )

        print("==============================")

        # =====================
        # CROP MATCH 검사
        # =====================

        crop_match = True

        if crop:
            if not disease.startswith(crop):
                crop_match = False

        # =====================
        # CONFIDENCE 보정
        # =====================

        if confidence < 0.40:

            disease = "판정 불확실"
            disease_id = None
            risk = "UNKNOWN"

        else:

            if "정상" in disease:

                disease_id = "normal"
                risk = "LOW"

            elif "잉크병" in disease:

                disease_id = "ink_disease"
                risk = "HIGH"

            elif "진딧물" in disease:

                disease_id = "aphid"
                risk = "MEDIUM"

            elif "탄저병" in disease:

                disease_id = "anthracnose"
                risk = "HIGH"

            elif "병징" in disease:

                disease_id = None
                risk = "MEDIUM"

            else:

                disease_id = None
                risk = "UNKNOWN"

        print(
            "DISEASE ID :",
            disease_id
        )

        print(
            "CROP MATCH :",
            crop_match
        )

        print(
            "FINAL DISEASE :",
            disease
        )

        # =====================
        # FIREBASE INFO 조회
        # =====================

        disease_info = None

        if disease_id is not None:

            print(
                "CHECK DISEASE ID:",
                disease_id
            )

            print(
                "CHECK CROP:",
                crop
            )

            disease_info = get_disease_info(
                disease_id,
                crop
            )

            print(
                "FIREBASE INFO :",
                disease_info
            )

        print(
            "FINAL RISK :",
            risk
        )

        print("==============================")

        # =====================
        # FIREBASE SEARCH NAME
        # =====================

        firebase_disease_name = disease

        if disease == "판정 불확실":

            firebase_disease_name = CLASS_NAMES[cls_id].replace(
                crop + "_",
                ""
            )

        # =====================
        # FIREBASE DISEASE INFO
        # =====================

        disease_info = None

        try:
            disease_info = get_disease_info(
                crop,
                disease_id
            )

            print(
                "LOCAL DISEASE INFO:",
                disease_info
            )

        except Exception as e:
            print(
                "LOCAL DB ERROR:",
                e
            )            

        return {
            "success": True,
            "crop": crop,
            "disease": disease,
            "confidence": round(
                confidence * 100,
                2
            ),
            "risk": risk,
            "crop_match": crop_match,
            "info": disease_info,
            "time": elapsed
        }

    except Exception as e:

        print(
            "🔥 ERROR:",
            e
        )

        traceback.print_exc()

        return {
            "success": False,
            "crop": crop,
            "disease": "분석 실패",
            "confidence": 0,
            "risk": "UNKNOWN",
            "error": str(e)
        }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )