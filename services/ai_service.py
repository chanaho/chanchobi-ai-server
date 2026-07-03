from ultralytics import YOLO
from PIL import Image
import torch
import io
import os

from services.disease_db import get_disease_info


class Predictor:
    def __init__(self, model_path):
        print("🔥 MODEL LOADING:", model_path)

        if not os.path.exists(model_path):
            raise Exception("Model not found")

        self.model = YOLO(model_path)

        print("✅ MODEL LOADED:", self.model.names)

    # 🚀 crop 추정 엔진
    def _infer_crop(self, class_name: str):
        if class_name is None:
            return "unknown"

        name = str(class_name)

        if "고추" in name:
            return "고추"
        if "사과" in name:
            return "사과"
        if "자두" in name:
            return "자두"
        if "복숭아" in name:
            return "복숭아"
        if "포도" in name:
            return "포도"
        if "토마토" in name:
            return "토마토"
        if "오이" in name:
            return "오이"
        if "수박" in name:
            return "수박"
        if "블루베리" in name:
            return "블루베리"
        if "아로니아" in name:
            return "아로니아"
        if "체리" in name:
            return "체리"

        return "unknown"

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

            # 🔥 안전 변환
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            names = self.model.names

            # 🔥 class name 추출 (안정형)
            if isinstance(names, dict):
                class_name = names.get(cls_id, "unknown")
            elif isinstance(names, list):
                class_name = names[cls_id] if cls_id < len(names) else "unknown"
            else:
                class_name = "unknown"

            # 🚀 crop 자동 추정
            crop = self._infer_crop(class_name)

            # 🚀 DB 매핑
            info = get_disease_info(class_name, crop)

            return {
                "status": "success",
                "crop": info["crop"],
                "disease": info["name"],
                "confidence": round(conf * 100, 2),
                "risk": info["risk"],
                "chemical": info["chemical"],
                "method": info["method"],
                "note": info["note"],
                "warning": info["warning"]
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
            "risk": "LOW",
            "chemical": [],
            "method": "기본 관리",
            "note": "정상 상태",
            "warning": ""
        }