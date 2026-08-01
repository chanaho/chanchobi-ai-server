import os
import json


BASE_DIR = os.path.dirname(os.path.dirname(__file__))


DISEASE_CACHE = {}


def load_json_database():
    global DISEASE_CACHE

    if DISEASE_CACHE:
        return DISEASE_CACHE

    db_path = os.path.join(
        BASE_DIR,
        "disease_db"
    )

    if not os.path.exists(db_path):
        print("⚠️ disease_db folder not found")
        return {}

    for filename in os.listdir(db_path):

        if filename.endswith(".json"):

            file_path = os.path.join(
                db_path,
                filename
            )

            try:
                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                crop = data.get(
                    "crop",
                    ""
                )

                for disease in data.get(
                    "diseases",
                    []
                ):

                    disease_id = disease.get(
                        "id",
                        ""
                    )

                    name = disease.get(
                        "name",
                        ""
                    )

                    key = f"{crop}_{name}"

                    DISEASE_CACHE[key] = disease

                    if disease_id:
                        DISEASE_CACHE[disease_id] = disease

                print(
                    "✅ LOAD DB:",
                    filename
                )

            except Exception as e:
                print(
                    "❌ DB LOAD ERROR:",
                    filename,
                    e
                )

    return DISEASE_CACHE


def get_disease_info(
    label: str,
    crop: str = "unknown"
):

    db = load_json_database()


    # 1. crop + name 검색
    if crop != "unknown":

        key = f"{crop}_{label}"

        if key in db:
            return normalize(
                db[key],
                crop
            )


    # 2. id 검색
    if label in db:

        return normalize(
            db[label],
            crop
        )


    # 3. 이름 포함 검색

    for key, value in db.items():

        if value.get("name") == label:

            return normalize(
                value,
                crop
            )


    # 4. 실패

    return {

        "name": label,

        "crop": crop,

        "risk": "UNKNOWN",

        "chemical": [],

        "method": "",

        "note": "",

        "warning": "",

        "cause": "",

        "condition": [],

        "prevention": [],

        "spray_time": [],

        "pesticides": []

    }



def normalize(
    data,
    crop
):

    return {

        "name":
            data.get(
                "name",
                ""
            ),

        "crop":
            crop,

        "risk":
            data.get(
                "risk",
                "UNKNOWN"
            ),

        "chemical":
            data.get(
                "chemical",
                []
            ),

        "method":
            data.get(
                "method",
                ""
            ),

        "note":
            data.get(
                "note",
                ""
            ),

        "warning":
            data.get(
                "warning",
                ""
            ),

        "symptom":
            data.get(
                "symptom",
                []
            ),

        "cause":
            data.get(
                "cause",
                ""
            ),

        "condition":
            data.get(
                "condition",
                []
            ),

        "prevention":
            data.get(
                "prevention",
                []
            ),

        "spray_time":
            data.get(
                "spray_time",
                []
            ),

        "pesticides":
            data.get(
                "pesticides",
                []

            )
    }