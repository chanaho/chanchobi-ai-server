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

MODEL = YOLO("model/best.pt")

print("✅ MODEL LOADED")
print("MODEL TASK :", MODEL.task)
print("MODEL NAMES :", MODEL.names)

print("MODEL INFO START")
print(MODEL.model)
print("MODEL INFO END")


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

        print("\n==============================")
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



        # -----------------------
        # 이미지 축소
        # -----------------------

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
            (256,256)
        )


        cv2.imwrite(
            file_path,
            img
        )


        print("IMAGE RESIZE OK")



        # -----------------------
        # YOLO
        # -----------------------

        print("BEFORE MODEL")

        start=time.time()


        results = MODEL(
            file_path,
            imgsz=256,
            conf=0.25,
            max_det=1,
            verbose=False,
            half=False,
            augment=False
        )


        elapsed = round(
            time.time()-start,
            2
        )


        print("AFTER MODEL")

        print(
            "TIME:",
            elapsed,
            "sec"
        )



        result = results[0]


        box_count = len(
            result.boxes
        )


        print(
            "BOX COUNT:",
            box_count
        )



        if box_count == 0:


            return {

                "success":True,
                "crop":crop,
                "disease":"알 수 없음",
                "confidence":0,
                "risk":"UNKNOWN",
                "time":elapsed

            }



        cls = int(
            result.boxes.cls[0]
        )


        conf=float(
            result.boxes.conf[0]
        )


        disease = MODEL.names[cls]



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



    except Exception:


        print("######## ERROR ########")

        traceback.print_exc()


        return {


            "success":False,

            "error":
            traceback.format_exc()

        }