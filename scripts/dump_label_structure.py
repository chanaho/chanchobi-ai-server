import os
import json
from collections import defaultdict, Counter

ROOT = r"D:\앱개발\식물병 유발\라벨링데이터"

stats = {
    "total_files": 0,
    "valid_json": 0,
    "fail_json": 0,
    "keys": Counter(),
    "disease": Counter(),
    "crop": Counter(),
    "object_class_code": Counter(),
    "risk": Counter(),
    "bbox_exist": 0,
    "bbox_missing": 0,
    "samples": []
}

def extract_fields(data):
    disease = None
    crop = None
    risk = None
    obj_code = None
    bbox = False

    # 1. disease
    if isinstance(data, dict):
        if "disease" in data:
            disease = data["disease"]

        if "crop" in data:
            crop = data["crop"]

        if "risk" in data:
            risk = data["risk"]

        if "object_class_code" in data:
            obj_code = data["object_class_code"]

        # bbox 체크 (여러 구조 대응)
        if "annotations" in data:
            for ann in data["annotations"]:
                if "bbox" in ann:
                    bbox = True

        if "bbox" in data:
            bbox = True

    return disease, crop, risk, obj_code, bbox


for root, dirs, files in os.walk(ROOT):
    for file in files:
        if not file.endswith(".json"):
            continue

        path = os.path.join(root, file)
        stats["total_files"] += 1

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            stats["valid_json"] += 1

            disease, crop, risk, obj_code, bbox = extract_fields(data)

            if disease:
                stats["disease"][disease] += 1

            if crop:
                stats["crop"][crop] += 1

            if risk:
                stats["risk"][risk] += 1

            if obj_code:
                stats["object_class_code"][str(obj_code)] += 1

            if bbox:
                stats["bbox_exist"] += 1
            else:
                stats["bbox_missing"] += 1

            # 샘플 10개 저장
            if len(stats["samples"]) < 10:
                stats["samples"].append({
                    "file": file,
                    "disease": disease,
                    "crop": crop,
                    "risk": risk,
                    "object_class_code": obj_code,
                    "bbox": bbox
                })

        except Exception as e:
            stats["fail_json"] += 1


# ========================
# 결과 출력
# ========================

print("\n===== DATA STRUCTURE REPORT =====\n")

print(f"TOTAL FILES: {stats['total_files']}")
print(f"VALID JSON: {stats['valid_json']}")
print(f"FAILED JSON: {stats['fail_json']}")
print(f"BBOX EXISTS: {stats['bbox_exist']}")
print(f"BBOX MISSING: {stats['bbox_missing']}")

print("\n--- DISEASE TOP ---")
for k, v in stats["disease"].most_common(20):
    print(k, ":", v)

print("\n--- CROP TOP ---")
for k, v in stats["crop"].most_common(20):
    print(k, ":", v)

print("\n--- OBJECT CLASS CODE ---")
for k, v in stats["object_class_code"].most_common(20):
    print(k, ":", v)

print("\n--- RISK ---")
for k, v in stats["risk"].most_common(20):
    print(k, ":", v)

print("\n--- SAMPLE DATA ---")
for s in stats["samples"]:
    print(s)