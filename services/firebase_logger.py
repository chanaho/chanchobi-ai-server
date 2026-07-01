# ai-server/services/firebase_logger.py

import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# =========================
# Firebase INIT (Render 안전 구조)
# =========================
if not firebase_admin._apps:
    try:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print("Firebase init skipped:", e)

db = firestore.client()


# =========================
# LOG RESULT
# =========================
def log_result(data: dict):
    """
    AI 분석 결과 Firebase 저장
    """

    try:
        doc = {
            "farm": data.get("farm", "unknown"),
            "crop": data.get("crop", "unknown"),
            "label": data.get("label"),
            "disease": data.get("disease"),
            "confidence": data.get("confidence"),
            "risk": data.get("risk"),
            "chemical": data.get("chemical"),
            "method": data.get("method"),
            "note": data.get("note"),
            "warning": data.get("warning"),
            "created_at": datetime.datetime.utcnow()
        }

        db.collection("pestResults").add(doc)

        return True

    except Exception as e:
        print("Firebase save error:", e)
        return False