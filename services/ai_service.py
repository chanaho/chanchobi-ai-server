from ultralytics import YOLO
import os

# =========================
# MODEL LOAD (1회 로딩)
# =========================

MODEL_PATH = "runs/detect/train/weights/best.pt"

try:
    model = YOLO(MODEL_PATH)

    print("✅ AI MODEL LOAD SUCCESS")
    print("📌 MODEL PATH:", MODEL_PATH)
    print("📌 MODEL CLASS NAMES:", model.names)

    MODEL_READY = True

except Exception as e:

    print("❌ AI MODEL LOAD ERROR:", e)

    model = None
    MODEL_READY = False


# =========================
# PREDICT FUNCTION
# =========================

def run_ai_predict(image_path: str):

    # =========================
    # MODEL CHECK
    # =========================

    if not MODEL_READY:
        return {
            "success": False,
            "error": "Model not loaded"
        }

    # =========================
    # FILE CHECK
    # =========================

    if not os.path.exists(image_path):
        return {
            "success": False,
            "error": "Image file not found"
        }

    print("\n=========================")
    print("📷 AI PREDICT START")
    print("📌 IMAGE:", image_path)

    # =========================
    # YOLO PREDICT
    # =========================

    results = model.predict(
        source=image_path,
        conf=0.5,
        iou=0.5,
        verbose=False
    )

    names = model.names

    print("🔥 MODEL NAMES:", names)

    detections = []

    # =========================
    # RESULT LOOP
    # =========================

    for r in results:

        print("📦 BOXES:", r.boxes)

        if r.boxes is None:
            continue

        for box in r.boxes:

            try:

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                class_name = names.get(cls_id, str(cls_id))

                print(
                    f"✅ DETECTED: {class_name} / CONF: {conf}"
                )

                detections.append({
                    "name": class_name,
                    "confidence": round(conf, 2)
                })

            except Exception as e:

                print("❌ BOX PARSE ERROR:", e)

    # =========================
    # 탐지 실패
    # =========================

    if not detections:

        print("⚠ NO DETECTION")

        return {
            "success": True,
            "crop": "미확인",
            "disease": "미감지",
            "risk": "LOW",
            "recommendation": "추가 예찰 필요",
            "detections": []
        }

    # =========================
    # 신뢰도 정렬
    # =========================

    detections.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    top3 = detections[:3]

    best = top3[0]

    name = best["name"]
    conf = best["confidence"]

    # =========================
    # 위험도 계산
    # =========================

    if conf >= 0.85:
        risk = "HIGH"

    elif conf >= 0.60:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    result = {
        "success": True,
        "disease": name,
        "risk": risk,
        "max_confidence": conf,
        "detections": top3
    }

    print("🔥 FINAL RESULT:", result)
    print("=========================\n")

    return result