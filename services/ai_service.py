from ultralytics import YOLO
from PIL import Image
import torch
import io
import os


class Predictor:
    def __init__(self, model_path):
        print("===================================")
        print("🔥 AI MODEL LOADING...")
        print("MODEL PATH =", model_path)

        if not os.path.exists(model_path):
            raise Exception(f"Model not found : {model_path}")

        self.model = YOLO(model_path)

        print("🔥 MODEL LOADED SUCCESS")
        print("CLASS =", self.model.names)
        print("===================================")

    def predict(self, image_bytes):

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            with torch.inference_mode():

                results = self.model.predict(
                    source=image,
                    imgsz=640,
                    conf=0.25,
                    verbose=False
                )

            if len(results) == 0:
                return {
                    "status": "success",
                    "crop": "unknown",
                    "disease": "정상",
                    "confidence": 0,
                    "risk": "LOW"
                }

            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                return {
                    "status": "success",
                    "crop": "unknown",
                    "disease": "정상",
                    "confidence": 0,
                    "risk": "LOW"
                }

            box = r.boxes[0]

            cls = int(box.cls[0])

            confidence = float(box.conf[0])

            disease = self.model.names[cls]

            if confidence >= 0.85:
                risk = "HIGH"
            elif confidence >= 0.60:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            return {
                "status": "success",
                "crop": "unknown",
                "disease": disease,
                "confidence": round(confidence * 100, 2),
                "risk": risk
            }

        except Exception as e:

            print("AI ERROR =", e)

            return {
                "status": "error",
                "message": str(e)
            }