from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

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

MODEL = YOLO(
    "model/best-int8.onnx",
    task="detect"
)     

MODEL.overrides["verbose"] = False

print("✅ MODEL LOADED")
print("MODEL TASK :", MODEL.task)
print("MODEL NAMES :", MODEL.names)



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
        "classes": MODEL.names
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


        results = MODEL(
            img,
            imgsz=320,
            conf=0.10,
            iou=0.45,
            max_det=1,
            device="cpu",                                  
            verbose=False
        )


        elapsed = round(
            time.time()-start,
            2
        )


        print(
            "MODEL TIME:",
            elapsed,
            "sec"
        )

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



        result = results[0]


        box_count = len(
            result.boxes
        )


        print(
            "BOX COUNT:",
            box_count
        )



        # =====================
        # 검출 없음
        # =====================

        if box_count == 0:

            return {

                "success":True,

                "crop":crop,

                "disease":"알 수 없음",

                "confidence":0,

                "risk":"UNKNOWN",

                "time":elapsed

            }



        # =====================
        # 결과
        # =====================

        cls = int(
            result.boxes.cls[0]
        )


        conf = float(
            result.boxes.conf[0]
        )


        disease = MODEL.names[cls]



        print(
            "CLASS:",
            cls
        )


        print(
            "DISEASE:",
            disease
        )


        print(
            "CONF:",
            conf
        )



        return {

            "success":True,

            "crop":crop,

            "disease":disease,

            "confidence":round(
                conf,
                2
            ),

            "risk":"LOW",

            "time":elapsed

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