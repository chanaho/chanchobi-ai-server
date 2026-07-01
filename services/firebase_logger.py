import datetime
import firebase_admin
from firebase_admin import credentials, firestore

db = None

# =========================
# Firebase INIT
# =========================

if not firebase_admin._apps:
    try:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
        db = firestore.client()

        print("✅ Firebase Connected")

    except Exception as e:
        print("⚠ Firebase Disabled :", e)
        db = None

else:
    db = firestore.client()


# =========================
# LOG RESULT
# =========================

def log_result(data: dict):

    if db is None:
        print("⚠ Firebase Skip")
        return False

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

        print("Firebase Save Error :", e)

        return False