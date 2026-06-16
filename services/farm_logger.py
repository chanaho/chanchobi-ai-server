import firebase_admin
from firebase_admin import credentials, firestore

# Firebase 초기화
cred = credentials.Certificate("firebase-key.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()


def save_farm_log(data: dict):

    db.collection("farm_logs").add(data)

    print("🔥 Firebase 저장 완료:", data)