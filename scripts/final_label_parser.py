import json
from pathlib import Path

BASE = Path(r"D:\앱개발\식물병 유발")

FILES = list((BASE / "라벨링데이터").rglob("*.json"))

def extract_label(data):

    if not isinstance(data, dict):
        return None

    # ======================================
    # 1️⃣ annotations 구조
    # ======================================
    anno = data.get("annotations")
    if isinstance(anno, dict):

        if anno.get("disease"):
            return str(anno["disease"])

        if anno.get("object_class_code"):
            return str(anno["object_class_code"])

    # ======================================
    # 2️⃣ root 구조
    # ======================================
    if data.get("disease"):
        return str(data["disease"])

    if data.get("object_class_code"):
        return str(data["object_class_code"])

    # ======================================
    # 3️⃣ crop fallback
    # ======================================
    if data.get("crop"):
        return f"crop_{data['crop']}"

    # ======================================
    # 4️⃣ objects 구조 (AI-Hub 일부)
    # ======================================
    if "objects" in data:
        objs = data["objects"]
        if isinstance(objs, list) and len(objs) > 0:
            obj = objs[0]
            return obj.get("class", None)

    return None


count = {}
total = 0
fail = 0

for f in FILES[:300]:  # 샘플 300개
    try:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)

        label = extract_label(data)

        if label:
            count[label] = count.get(label, 0) + 1
            total += 1
        else:
            fail += 1

    except Exception:
        fail += 1

print("===== LABEL RESULT =====")
print("TOTAL:", total)
print("FAIL:", fail)
print("------------------------")

for k, v in sorted(count.items(), key=lambda x: -x[1]):
    print(k, v)