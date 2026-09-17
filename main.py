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
from ultralytics import YOLO

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
from ultralytics import YOLO

MODEL_PATH = "models/chanchobi_cls_best.onnx"

# =========================
# YOLO DETECTION MODEL
# =========================

PEPPER_MODEL_PATH = "models/pepper_yolo_best.pt"
CITRUS_MODEL_PATH = "models/citrus_ulcer_best.pt"
PEPPER_DISEASE_MODEL_PATH = "models/pepper_a7_a8_best.pt"

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

# =========================
# ONNX MODEL CHECK
# =========================

if not os.path.exists(MODEL_PATH):
    print(
        "❌ MODEL FILE NOT FOUND :",
        MODEL_PATH
    )
    exit()

# =========================
# YOLO MODEL CHECK
# =========================

if not os.path.exists(PEPPER_MODEL_PATH):
    print(
        "❌ PEPPER MODEL FILE NOT FOUND :",
        PEPPER_MODEL_PATH
    )

if not os.path.exists(CITRUS_MODEL_PATH):
    print(
        "❌ CITRUS MODEL FILE NOT FOUND :",
        CITRUS_MODEL_PATH
    )

if not os.path.exists(PEPPER_DISEASE_MODEL_PATH):
    print(
        "❌ PEPPER DISEASE MODEL FILE NOT FOUND :",
        PEPPER_DISEASE_MODEL_PATH
    )

# =========================
# MODEL SESSION
# =========================

session = None
pepper_model = None
citrus_model = None
pepper_disease_model = None


# =========================
# ONNX CLASSIFICATION MODEL
# =========================

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
# PEPPER YOLO MODEL
# =========================

def get_pepper_model():
    global pepper_model

    if pepper_model is None:

        print("🔥 LOADING PEPPER YOLO MODEL")

        pepper_model = YOLO(
            PEPPER_MODEL_PATH
        )

        print(
            "🔥 PEPPER YOLO MODEL LOADED"
        )

        print(
            "PEPPER TASK :",
            pepper_model.task
        )

        print(
            "PEPPER NAMES :",
            pepper_model.names
        )

    return pepper_model


# =========================
# CITRUS YOLO MODEL
# =========================


# =========================
# PEPPER A7/A8 DISEASE YOLO MODEL
# =========================

def get_pepper_disease_model():
    global pepper_disease_model

    if pepper_disease_model is None:
        print("🔥 LOADING PEPPER A7/A8 DISEASE MODEL")

        pepper_disease_model = YOLO(
            PEPPER_DISEASE_MODEL_PATH
        )

        print(
            "🔥 PEPPER A7/A8 DISEASE MODEL LOADED"
        )

        print(
            "PEPPER DISEASE TASK :",
            pepper_disease_model.task
        )

        print(
            "PEPPER DISEASE NAMES :",
            pepper_disease_model.names
        )

    return pepper_disease_model

def get_citrus_model():
    global citrus_model

    if citrus_model is None:

        print("🔥 LOADING CITRUS YOLO MODEL")

        citrus_model = YOLO(
            CITRUS_MODEL_PATH
        )

        print(
            "🔥 CITRUS YOLO MODEL LOADED"
        )

        print(
            "CITRUS TASK :",
            citrus_model.task
        )

        print(
            "CITRUS NAMES :",
            citrus_model.names
        )

    return citrus_model


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

        original_img = img.copy()

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

        # =====================
        # 전체 모델 최고 결과
        # =====================

        global_cls_id = int(np.argmax(all_probs))
        global_confidence = float(all_probs[global_cls_id])
        global_class = CLASS_NAMES[global_cls_id]

        if "_" in global_class:
            predicted_crop = global_class.split("_")[0]
        else:
            predicted_crop = ""

        print(
            "GLOBAL TOP CLASS :",
            global_cls_id
        )

        print(
            "GLOBAL TOP NAME :",
            global_class
        )

        print(
            "GLOBAL TOP CONF :",
            round(global_confidence * 100, 2)
        )

        print(
            "PREDICTED CROP :",
            predicted_crop
        )

        print(
            "SELECTED CROP :",
            crop
        )

        # =====================
        # 선택 작물과 실제 예측 작물 비교
        # =====================

        crop_match = True

        if crop and predicted_crop:

            if global_confidence >= 0.50 and crop != "고추":

                if predicted_crop != crop:

                    crop_match = False

                    print("==============================")
                    print("⚠️ 작물 불일치")
                    print(
                        "SELECTED :",
                        crop
                    )
                    print(
                        "PREDICTED :",
                        predicted_crop
                    )
                    print(
                        "CONFIDENCE :",
                        round(global_confidence * 100, 2)
                    )
                    print("==============================")

                    return {
                        "success": True,
                        "crop": crop,
                        "disease": "작물 불일치",
                        "confidence": 0,
                        "risk": "UNKNOWN",
                        "crop_match": False,
                        "info": None,
                        "time": elapsed
                    }

        # =====================
        # crop별 허용 클래스
        # =====================

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

        # =====================
        # 허용 클래스 중 최고 확률
        # =====================

        best_score = -1.0
        cls_id = -1

        for idx in allowed_classes:

            score = float(all_probs[idx])

            if score > best_score:
                best_score = score
                cls_id = idx

        confidence = best_score

        disease = CLASS_NAMES[cls_id]

        if crop == "":
            if "_" in disease:
                crop = disease.split("_")[0]

                print(
                    "AUTO CROP:",
                    crop
                )

        print(
            "FINAL CROP:",
            crop
        )

        # =========================
        # PEPPER YOLO DETECTION
        # =========================

        if crop == "고추":

            print("==============================")
            print("🌶️ PEPPER YOLO START")

            pepper_model = get_pepper_model()

            pepper_results = pepper_model.predict(
                source=original_img,
                conf=0.25,
                verbose=False
            )

            pepper_detections = []

            for result in pepper_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    cls_id_yolo = int(
                        box.cls[0].item()
                    )

                    conf_yolo = float(
                        box.conf[0].item()
                    )

                    class_name_yolo = pepper_model.names[
                        cls_id_yolo
                    ]

                    pepper_detections.append({
                        "id": cls_id_yolo,
                        "name": class_name_yolo,
                        "confidence": round(
                            conf_yolo * 100,
                            2
                        )
                    })

            print(
                "PEPPER YOLO RESULT :",
                pepper_detections
            )


            # =========================
            # PEPPER A7/A8 DISEASE YOLO TEST
            # =========================
            pepper_disease_model = get_pepper_disease_model()

            pepper_disease_results = pepper_disease_model.predict(
                source=original_img,
                conf=0.25,
                verbose=False
            )

            pepper_disease_detections = []

            for disease_result in pepper_disease_results:
                if disease_result.boxes is None:
                    continue

                for box in disease_result.boxes:
                    cls_id_disease = int(box.cls[0].item())
                    conf_disease = float(box.conf[0].item())
                    class_name_disease = pepper_disease_model.names[cls_id_disease]

                    pepper_disease_detections.append({
                        "id": cls_id_disease,
                        "name": class_name_disease,
                        "confidence": round(conf_disease * 100, 2)
                    })

            print("==============================")
            print("🌶️ A7/A8 DISEASE RESULT :", pepper_disease_detections)
            print("==============================")

            print("==============================")

            # =========================
        # PEPPER YOLO FINAL RESULT
        # =========================

        pepper_yolo_found = False
        pepper_yolo_name = None
        pepper_yolo_confidence = 0.0
        pepper_db_name = None
        pest_info = None
        pest_risk = "UNKNOWN"

        if pepper_detections:

            best_yolo = max(
                pepper_detections,
                key=lambda x: x["confidence"]
            )

            pepper_yolo_found = True
            pepper_yolo_name = best_yolo["name"]
            pepper_yolo_confidence = (
                best_yolo["confidence"] / 100.0
            )

            # YOLO 해충명 → disease_db 일반명 연결
            if pepper_yolo_name == "복숭아혹진딧물":
                pepper_db_name = "진딧물"

            elif pepper_yolo_name == "꽃노랑총채벌레":
                pepper_db_name = "총채벌레"

            else:
                pepper_db_name = pepper_yolo_name

            print("==============================")
            print("🌶️ PEPPER YOLO FINAL")
            print(
                "YOLO NAME :",
                pepper_yolo_name
            )
            print(
                "YOLO CONF :",
                round(
                    pepper_yolo_confidence * 100,
                    2
                )
            )
            print(
                "DB NAME :",
                pepper_db_name
            )
            print("==============================")

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

        # =========================
        # YOLO 해충 결과 분리
        # =========================
        # 해충 YOLO가 질병 결과를 덮어쓰지 않도록 한다.
        # 질병은 ONNX 결과를 유지하고,
        # 해충은 pest 필드로 별도 반환한다.

        if pepper_yolo_found:

            print("==============================")
            print("🌶️ PEPPER PEST RESULT")
            print(
                "PEST NAME :",
                pepper_yolo_name
            )
            print(
                "PEST CONF :",
                round(
                    pepper_yolo_confidence * 100,
                    2
                )
            )
            print(
                "PEST DB NAME :",
                pepper_db_name
            )
            print("==============================")

        # =====================
        # CROP MATCH 검사
        # =====================

        crop_match = True

        # YOLO가 검출한 결과는 이미 선택 작물의 전용 모델 결과이므로
        # 작물 불일치로 처리하지 않는다.
        if pepper_yolo_found:

            crop_match = True

        else:

            if crop:
                if not disease.startswith(crop):
                    crop_match = False

        # =====================
        # =========================
        # PEPPER A7/A8 DISEASE PRIORITY
        # =========================
        if crop == "고추" and "pepper_disease_detections" in locals():
            if pepper_disease_detections:
                best_disease_yolo = max(
                    pepper_disease_detections,
                    key=lambda x: x["confidence"]
                )

                a7a8_name = best_disease_yolo["name"]
                a7a8_confidence = best_disease_yolo["confidence"] / 100.0

                disease = a7a8_name
                confidence = a7a8_confidence

                print(
                    "🌶️ A7/A8 DISEASE PRIORITY :",
                    disease
                )

                print(
                    "A7/A8 DISEASE CONF : ",
                    round(confidence * 100, 2)
                )


        # CONFIDENCE 보정
        # =====================

        if confidence < 0.40:

            disease = "판정 불확실"
            disease_id = None
            risk = "UNKNOWN"

        else:

            # =====================
            # DISEASE ID 자동 검색
            # 현재 crop의 disease_db를 기준으로
            # 실제 ID를 찾는다.
            # =====================

            disease_id = None
            disease_info = None
            risk = "UNKNOWN"

            try:

                # YOLO 병해충명 → disease_db 검색명
                lookup_disease_name = disease.replace(f"{crop}_", "", 1) if disease.startswith(f"{crop}_") else disease

                # 현재 crop의 disease_db JSON 탐색
                db_dir = os.path.join(
                    os.path.dirname(__file__),
                    "disease_db"
                )

                for filename in os.listdir(db_dir):

                    if not filename.endswith(".json"):
                        continue

                    file_path = os.path.join(
                        db_dir,
                        filename
                    )

                    try:

                        with open(
                            file_path,
                            "r",
                            encoding="utf-8"
                        ) as f:

                            db_data = json.load(f)

                        db_crop = db_data.get(
                            "crop",
                            ""
                        )

                        # 현재 crop과 일치하는 DB만 검색
                        if db_crop != crop:
                            continue

                        for item in db_data.get(
                            "diseases",
                            []
                        ):

                            db_name = item.get(
                                "name",
                                ""
                            )

                            db_id = item.get(
                                "id"
                            )

                            db_type = item.get(
                                "type",
                                ""
                            )

                            if db_name and db_name == lookup_disease_name:

                                disease_id = db_id
                                disease_info = item

                                if db_type == "normal":
                                    risk = "LOW"

                                elif db_type == "pest":
                                    risk = item.get(
                                        "risk",
                                        "MEDIUM"
                                    )

                                elif db_type == "disease":
                                    risk = item.get(
                                        "risk",
                                        "HIGH"
                                    )

                                print(
                                    "DB MATCH :",
                                    db_crop,
                                    "|",
                                    db_name,
                                    "|",
                                    db_id
                                )

                                break

                        if disease_id is not None:
                            break

                    except Exception as e:

                        print(
                            "DB SEARCH ERROR:",
                            filename,
                            e
                        )

            except Exception as e:

                print(
                    "DISEASE DB SEARCH ERROR:",
                    e
                )

            print(
                "DISEASE ID :",
                disease_id
            )

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
        # DISEASE DB 최종 조회
        # =====================

        if disease_id is not None:

            print(
                "CHECK DISEASE ID:",
                disease_id
            )

            print(
                "CHECK CROP:",
                crop
            )

            try:

                disease_info = get_disease_info(
                    crop,
                    disease_id
                )

                print(
                    "DISEASE INFO:",
                    disease_info
                )

            except Exception as e:

                print(
                    "DISEASE DB ERROR:",
                    e
                )

        else:

            disease_info = None

            print(
                "DISEASE ID 없음"
            )


        print(
            "FINAL RISK :",
            risk
        )

        print("==============================")


        # =====================
        # PEST DB 최종 조회
        # =====================

        pest_info = None
        pest_risk = "UNKNOWN"

        if pepper_yolo_found and pepper_db_name:

            try:

                pest_info = get_disease_info(
                    crop,
                    {
                        "복숭아혹진딧물": "aphid",
                        "꽃노랑총채벌레": "thrips",
                        "담배나방": "tobacco_budworm",
                        "검거세미밤나방": "beet_armyworm",
                        "담배가루이": "sweetpotato_whitefly",
                        "담배거세미나방": "common_cutworm",
                        "도둑나방": "cabbage_looper",
                        "먹노린재": "rice_black_stink_bug",
                        "목화바둑명나방": "cotton_leaf_roller",
                        "무잎벌": "turnip_sawfly",
                        "배추좀나방": "diamondback_moth",
                        "배추흰나비": "cabbage_white_butterfly",
                        "벼룩잎벌레": "flea_beetle",
                        "비단노린재": "striped_shield_bug",
                        "썩덩나무노린재": "brown_marmorated_stink_bug",
                        "알락수염노린재": "yellow_spotted_stink_bug",
                        "열대거세미나방": "fall_armyworm",
                        "큰28점박이무당벌레": "large_28_spotted_ladybird",
                        "톱다리개미허리노린재": "bean_bug",
                        "파밤나방": "beet_armyworm_spodoptera_exigua"
                    }.get(
                        pepper_yolo_name,
                        ""
                    )
                ) if pepper_yolo_name else None

                if pest_info:

                    pest_risk = pest_info.get(
                        "risk",
                        "UNKNOWN"
                    )

                    print(
                        "PEST DB MATCH:",
                        pest_info.get("name"),
                        "|",
                        pest_info.get("id"),
                        "|",
                        pest_risk
                    )

                else:

                    print(
                        "PEST DB NOT FOUND:",
                        pepper_db_name
                    )

            except Exception as e:

                print(
                    "PEST DB FINAL ERROR:",
                    e
                )

        # =====================
        # 종합 위험도 계산
        # 병해와 해충 중 더 높은 위험도를 적용
        # =====================

        risk_priority = {
            "UNKNOWN": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3
        }

        disease_risk_value = risk_priority.get(
            risk,
            0
        )

        pest_risk_value = risk_priority.get(
            pest_risk,
            0
        )

        if pest_risk_value > disease_risk_value:
            risk = pest_risk

        print(
            "COMBINED RISK :",
            risk
        )


        # =====================
        # FIREBASE SEARCH NAME
        # =====================

        firebase_disease_name = disease

        if disease == "판정 불확실":

            firebase_disease_name = CLASS_NAMES[cls_id].replace(
                crop + "_",
                ""
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
            "pest": pepper_db_name if pepper_yolo_found else None,
            "pest_confidence": round(
                pepper_yolo_confidence * 100,
                2
            ) if pepper_yolo_found else 0,
            "pest_info": pest_info,
            "pest_risk": pest_risk,
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




