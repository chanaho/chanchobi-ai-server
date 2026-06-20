import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# 이미 초기화 되어있으면 재사용
if not firebase_admin._apps:

    # Render / 서버 환경 변수 방식 우선
    firebase_json = os.getenv("FIREBASE_KEY_JSON")

    if firebase_json:
        cred = credentials.Certificate(json.loads(firebase_json))
    else:
        # 로컬 fallback
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        KEY_PATH = os.path.join(BASE_DIR, "firebase-key.json")
        cred = credentials.Certificate(KEY_PATH)

    firebase_admin.initialize_app(cred)

db = firestore.client()

print("🔥 Firebase 연결 완료")