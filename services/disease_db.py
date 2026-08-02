import os
import json

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

DISEASE_CACHE = {}


def load_json_database():

    global DISEASE_CACHE

    db_path = os.path.join(
        BASE_DIR,
        "disease_db"
    )

    if not os.path.exists(db_path):
        print("⚠ disease_db folder not found")
        return

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

                for d in data.get(
                    "diseases",
                    []
                ):

                    key = (
                        crop,
                        d.get("id")
                    )

                    DISEASE_CACHE[key] = d


                print(
                    "LOAD:",
                    filename
                )


            except Exception as e:

                print(
                    "JSON LOAD ERROR:",
                    e
                )


load_json_database()



def get_disease_info(
    crop,
    disease_id
):

    key = (
        crop,
        disease_id
    )

    info = DISEASE_CACHE.get(
        key
    )

    if info:
        return info

    return None