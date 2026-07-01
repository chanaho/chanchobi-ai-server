import torch
from ultralytics import YOLO
from PIL import Image
import io
from services.disease_db import get_disease_info


class Predictor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.names = self.model.names

    def predict(self, image_bytes: bytes):

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            return {"status": "error", "message": "image decode failed"}

        image.thumbnail((640, 640))

        with torch.inference_mode():
            results = self.model.predict(
                source=image,
                imgsz=640,
                conf=0.25,
                device="cpu",
                verbose=False
            )

        if not results or len(results) == 0:
            return {
                "status": "success",
                "disease": "정상",
                "risk": "LOW",
                "confidence": 0
            }

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return {
                "status": "success",
                "disease": "정상",
                "risk": "LOW",
                "confidence": 0
            }

        box = r.boxes[0]
        cls_id = int(box.cls[0])
        label = self.names.get(cls_id, "unknown")
        confidence = float(box.conf[0]) * 100

        info = get_disease_info(label)

        return {
            "status": "success",
            "label": label,
            "disease": info["name"],
            "crop": info["crop"],
            "risk": info["risk"],
            "confidence": round(confidence, 2),
            "chemical": info.get("chemical"),
            "method": info.get("method"),
            "note": info.get("note"),
            "warning": info.get("warning")
        }