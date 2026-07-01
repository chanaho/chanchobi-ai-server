from ultralytics import YOLO
from PIL import Image
import torch
import io
import os
import traceback


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

            print("===== AI Predict Start =====")

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            print("✅ Image Loaded")

            with torch.inference_mode():

                print("🚀 YOLO Predict Start")

                results = self.model.predict(
                    source=image,
                    imgsz=640,
                    conf=0.25,
                    verbose=False
                )

            print("✅ YOLO Predict Finished")

            if len(results) == 0:
                print("⚠ No Results")

                return {
                    "status": "success",
                    "crop": "unknown",
                    "disease": "정상",
                    "confidence": 0,
                    "risk": "LOW"
                }

            r = results[0]

            print("Boxes =", r.boxes)

            if r.boxes is None or len(r.boxes) == 0:

                print("⚠ No Boxes")

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

            print("Prediction =", disease)
            print("Confidence =", confidence)

            if confidence >= 0.85:
                risk = "HIGH"
            elif confidence >= 0.60:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            result = {
                "status": "success",
                "crop": "unknown",
                "disease": disease,
                "confidence": round(confidence * 100, 2),
                "risk": risk
            }

            print("===== RESULT =====")
            print(result)
            print("==================")

            return result

        except Exception as e:

            print("===================================")
            print("❌ AI PREDICT ERROR")
            traceback.print_exc()
            print("===================================")

            return {
                "status": "error",
                "message": str(e)
            }