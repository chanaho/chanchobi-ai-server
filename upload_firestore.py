import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json


# Firebase 키 위치
cred = credentials.Certificate(
    "firebase-key.json"
)

firebase_admin.initialize_app(cred)

db = firestore.client()


# JSON 파일 읽기
with open(
    "disease_db/plum.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


crop = data["crop"]

print("UPLOAD CROP :", crop)


# crops/plum 생성

crop_ref = db.collection("crops").document(
    crop
)

crop_ref.set(
    {
        "crop": crop,
        "version": data["version"]
    }
)


# diseases 업로드

for disease in data["diseases"]:

    disease_id = disease["id"]

    print(
        "UPLOAD:",
        disease_id
    )

    db.collection("crops")\
      .document(crop)\
      .collection("diseases")\
      .document(disease_id)\
      .set(disease)


print("🔥 FIREBASE UPLOAD COMPLETE")