from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

import shutil
import os
import time
import traceback
import cv2


app = FastAPI()


# =========================
# AI 모델 로드
# =========================

print("🔥 LOADING MODEL...")

MODEL = YOLO("model/best.onnx")

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



        filename = file.filename or "photo.jpg"


        file_path = os.path.join(
            UPLOAD_DIR,
            filename
        )


        print("SAVE :", file_path)



        with open(file_path,"wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        print("SAVE OK")



        # =====================
        # 이미지 처리
        # =====================

        img = cv2.imread(file_path)


        if img is None:

            return {
                "success":False,
                "error":"IMAGE READ FAIL"
            }



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


        print("RESIZE OK")



        # =====================
        # YOLO
        # =====================

        print("BEFORE MODEL")


        start = time.time()


        results = MODEL.predict(
            source=img,
            imgsz=256,
            conf=0.05,
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