from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort

import os
import time
import traceback
import cv2
import numpy as np
import yaml
import onnxruntime as ort
from fastapi import FastAPI

cv2.setNumThreads(1)

app = FastAPI()

# =========================
# AI 모델 로드
# =========================

print("🔥 LOADING MODEL...")

# ONNX 파일 확인
onnx_path = os.path.abspath("model/best.onnx")

print("=" * 60)
print("ONNX FILE :", onnx_path)
print("ONNX EXISTS :", os.path.exists(onnx_path))

if os.path.exists(onnx_path):
    print("ONNX SIZE :", os.path.getsize(onnx_path), "bytes")

print("=" * 60)

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
    onnx_path,
    sess_options=so,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

print("INPUTS")
for i in session.get_inputs():
    print(
        i.name,
        i.shape,
        i.type
    )

print()

print("OUTPUTS")
for o in session.get_outputs():
    print(
        o.name,
        o.shape,
        o.type
    )

print("=" * 60)

# =========================
# CLASS NAME LOAD
# =========================

DATASET_YAML = "crop_dataset.yaml"

with open(DATASET_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

if isinstance(data["names"], dict):
    CLASS_NAMES = {
        int(k): v
        for k, v in data["names"].items()
    }
else:
    CLASS_NAMES = {
        i: name
        for i, name in enumerate(data["names"])
    }

print("=" * 50)
print("DATASET :", DATASET_YAML)
print("CLASS_NAMES :", CLASS_NAMES)
print("=" * 50)
print("### MAIN.PY VERSION : 2026-07-10 STEP1 ###")
print("CLASS_NAMES =", CLASS_NAMES)

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

        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)

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

        print("PRED SHAPE :", pred.shape)

        print("=" * 60)
        print("FIRST DETECTION")
        print(np.round(pred[0], 4))

        # ==========================================
        # YOLOv8 ONNX 후처리
        # output : (8400, 15)
        # [x,y,w,h,class0...class10]
        # ==========================================

        boxes = pred[:, :4]

        class_scores = pred[:, 4:]

        class_scores = 1 / (1 + np.exp(-class_scores))

        cls_scores = np.max(class_scores, axis=1)

        cls_ids = np.argmax(class_scores, axis=1)

        best_idx = int(np.argmax(cls_scores))

        best_score = float(cls_scores[best_idx])

        best_box = boxes[best_idx]

        best_cls = int(cls_ids[best_idx])


        print()
        print("BEST DETECTION")
        print("BOX :", np.round(best_box, 4))
        print("CLASS :", best_cls)
        print("SCORE :", round(best_score, 4))
        print("=" * 60)


        print("BEST INDEX :", best_idx)
        print("MAX OUTPUT :", round(float(np.max(pred)), 4))
        print("MIN OUTPUT :", round(float(np.min(pred)), 4))


        # =====================
        # 검출 없음
        # =====================

        if best_score < 0.10:

            return {
                "success": True,
                "crop": crop,
                "disease": "알 수 없음",
                "confidence": 0,
                "risk": "UNKNOWN",
                "time": elapsed
            }


        # =====================
        # 최종 결과
        # =====================

        confidence = best_score

        disease = CLASS_NAMES.get(
            best_cls,
            "알 수 없음"
        )


        print("CLASS :", best_cls)

        print("DISEASE :", disease)

        print(
            "CONFIDENCE :",
            round(confidence, 4)
        )


        if confidence >= 0.80:

            risk = "HIGH"

        elif confidence >= 0.50:

            risk = "MEDIUM"

        else:

            risk = "LOW"



        return {

            "success": True,

            "crop": crop,

            "disease": disease,

            "confidence": round(confidence, 2),

            "risk": risk,

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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )         

