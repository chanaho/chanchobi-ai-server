from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort

import os
import time
import traceback
import cv2
import numpy as np

cv2.setNumThreads(1)

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

so = ort.SessionOptions()

so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "model/best.onnx",
    sess_options=so,
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



        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)

        print("FINAL IMAGE :", img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255.0

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        # =====================
        # YOLO
        # =====================

        print("BEFORE MODEL")

        t1 = time.time()

        outputs = session.run(
            output_names,
            {input_name: img}
        )

        t2 = time.time()

        model_time = round(t2 - t1, 2)

        print("ONNX TIME :", model_time)

        elapsed = model_time

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

        if pred.ndim == 3:
            pred = pred[0]

        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        print(
            "PRED SHAPE :",
            pred.shape
        )

        scores = np.max(
            pred[:, 5:],
            axis=1
        )

        best_idx = int(
            np.argmax(scores)
        )

        best = pred[best_idx]

        best_score = float(
            scores[best_idx]
        )

        print(
            "BEST INDEX :",
            best_idx
        )

        print(
            "BEST SCORE :",
            round(best_score, 4)
        )

        print(
            "BEST ROW :",
            best
        )

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