from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort

import os
import time
import traceback
import cv2
import numpy as np

app = FastAPI()


# =========================
# AI 모델 로드
# =========================

print("🔥 LOADING MODEL...")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["ORT_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

session = ort.InferenceSession(
    "model/best-int8.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

CLASS_NAMES = {
    0: "고추_탄저병",
    1: "사과_갈색무늬병",
    2: "자두_세균성구멍병",
    3: "아로니아_잿빛곰팡이병",
    4: "복숭아_세균성구멍병",
    5: "체리_갈색무늬병",
    6: "포도_노균병",
    7: "블루베리_잿빛곰팡이병",
    8: "토마토_잎마름병",
    9: "오이_노균병",
    10: "수박_덩굴마름병"
}

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

    return {
        "status": "AI SERVER RUNNING",
        "classes": CLASS_NAMES
    }



@app.get("/health")
def health():

    return {
        "status":"ok"
    }



# =========================
# PREDICT
# =========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form(None)
):

    try:

        print("==============================")
        print("🔥 NEW REQUEST")


        if not crop:
            crop = "unknown"


        print("CROP :", crop)



        contents = await file.read()

        img = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        print("IMAGE LOAD OK")

        if img is None:
            print("❌ IMAGE DECODE FAIL")

            return {
                "success":False,
                "crop": crop,
                "disease": "이미지 읽기 실패",
                "confidence": 0,
                "risk": "UNKNOWN",
                "error": "IMAGE DECODE FAIL"
            }

        print("✅ IMAGE DECODE SUCCESS")    

        h,w = img.shape[:2]


        print(
            "IMAGE SIZE:",
            w,
            "x",
            h
        )



        img = cv2.resize(
            img,
            (320,320)
        )


        print("FINAL IMAGE:", img.shape)



        # =====================
        # YOLO
        # =====================

        print("BEFORE MODEL")


        start = time.time()

        # OpenCV(BGR) → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # float32 변환
        img_rgb = img_rgb.astype(np.float32) / 255.0

        # (320,320,3) → (1,3,320,320)
        img_input = np.transpose(img_rgb, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)

        # ONNX 추론
        outputs = session.run(
            output_names,
            {input_name: img_input}
        )

        elapsed = round(time.time() - start, 2)


        print("MODEL TIME :", elapsed)
        print("OUTPUT SHAPE :", outputs[0].shape)
        print("OUTPUT TYPE :", type(outputs[0]))        

        if elapsed > 30:

           print("⚠️ AI TIMEOUT") 

           return { 
            "success": False,
            "crop": crop,
            "disease": "분석시간초과",
             "confidence": 0,
             "risk": "UNKNOWN",
             "time": elapsed
           }

        print("AFTER MODEL")



        # =====================
        # ONNX 결과
        # =====================

        pred = outputs[0]

        print("OUTPUT SHAPE :", outputs[0].shape)

        best = None
        best_score = 0.0

        # 출력 형태 확인
        if len(pred.shape) == 3:
            pred = pred[0]

        # (15,2100) 형태이면 전치
        if pred.shape[0] < pred.shape[1]:
           pred = pred.T

        print("PRED SHAPE :", pred.shape)

        best = None
        best_score = 0.0

        for row in pred:
            score = float(np.max(row[4:]))

            if score > best_score:
               best_score = score
               best = row

        print("BEST SCORE :", best_score)

        # =====================
        # 검출 없음
        # =====================

        if best is None or best_score < 0.10:

           return {

               "success": True,

               "crop": crop,

               "disease": "알 수 없음",

               "confidence": 0,

               "risk": "UNKNOWN",

               "time": elapsed

           }

        # =====================
        # 결과
        # =====================

        # x, y, w, h 다음부터 클래스 점수
        # YOLO 출력
        objectness = float(best[4])
        class_scores = best[5:]

        cls = int(np.argmax(class_scores))
        class_conf = float(class_scores[cls])

        conf = objectness * class_conf

        disease = CLASS_NAMES.get(cls, "알 수 없음")

        print("CLASS :", cls)
        print("DISEASE :", disease)
        print("CONF :", round(conf, 2))

        return {
            "success": True,
            "crop": crop,
            "disease": disease,
            "confidence": round(conf, 2),
            "risk": "LOW" if conf >= 0.7 else "MEDIUM",
            "time": elapsed
        }


    except Exception as e:


        print("######## ERROR ########")

        traceback.print_exc()


        return {

            "success":False,

            "crop":crop,

            "disease":"분석 실패",

            "confidence":0,

            "risk":"UNKNOWN",

            "error":str(e)

        }