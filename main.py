@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown")
):
    try:
        contents = await file.read()

        import numpy as np
        import cv2

        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "failed", "error": "image decode failed"}

        # 🔥 YOLO 실제 실행
        results = model(img)

        r = results[0]

        if r.boxes and len(r.boxes) > 0:
            cls = int(r.boxes.cls[0])
            conf = float(r.boxes.conf[0])

            return {
                "status": "success",
                "farm": farm,
                "result": {
                    "disease": model.names.get(cls, "unknown"),
                    "confidence": round(conf * 100, 2),
                    "risk": "HIGH" if conf > 0.7 else "MEDIUM"
                }
            }

        return {
            "status": "success",
            "farm": farm,
            "result": {
                "disease": "no_detection",
                "confidence": 0,
                "risk": "LOW"
            }
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }