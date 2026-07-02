from ultralytics import YOLO
from PIL import Image
import torch
import io
import os

class Predictor:
    def __init__(self, model_path):
        print("🔥 MODEL LOADING:", model_path)

        if not os.path.exists(model_path):
            raise Exception("Model not found")

        self.model = YOLO(model_path)

        print("✅ MODEL LOADED:", self.model.names)

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            with torch.inference_mode():
                results = self.model.predict(
                    source=image,
                    imgsz=640,
                    conf=0.25,
                    verbose=False
                )

            if not results or len(results) == 0:
                return self._empty()

            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                return self._empty()

            box = r.boxes[0]

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            disease = self.model.names.get(cls, "unknown")

            return {
                "status": "success",
                "crop": "unknown",
                "disease": disease,
                "confidence": round(conf * 100, 2),
                "risk": self._risk(conf)
            }

        except Exception as e:
            print("AI ERROR:", e)
            return {
                "status": "error",
                "message": str(e)
            }

    def _risk(self, conf):
        if conf >= 0.85:
            return "HIGH"
        elif conf >= 0.60:
            return "MEDIUM"
        return "LOW"

    def _empty(self):
        return {
            "status": "success",
            "crop": "unknown",
            "disease": "정상",
            "confidence": 0,
            "risk": "LOW"
        }