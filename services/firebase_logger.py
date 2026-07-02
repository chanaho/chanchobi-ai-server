import datetime

try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        try:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
        except:
            print("Firebase ADC SKIP")

    db = firestore.client()
    FIREBASE_OK = True

except:
    print("Firebase completely disabled")
    FIREBASE_OK = False


def log_result(data: dict):

    if not FIREBASE_OK:
        return False

    try:
        db.collection("pestResults").add({
            "disease": data.get("disease"),
            "risk": data.get("risk"),
            "confidence": data.get("confidence"),
            "created_at": datetime.datetime.utcnow()
        })
        return True

    except Exception as e:
        print("Firebase save error:", e)
        return False