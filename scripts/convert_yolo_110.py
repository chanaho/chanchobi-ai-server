import os
import json
import shutil

IMAGE_DIR = r"D:\앱개발\식물병 유발\원천데이터"
LABEL_DIR = r"D:\앱개발\식물병 유발\라벨링데이터"
OUTPUT = r"D:\dataset_yolo_110"

os.makedirs(f"{OUTPUT}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT}/labels/train", exist_ok=True)

class_map = {}
class_list = []

def get_class(label):
    label = str(label)
    if label not in class_map:
        class_map[label] = len(class_list)
        class_list.append(label)
    return class_map[label]

def find_bbox_and_label(obj):
    """
    어떤 구조든 bbox + label 추출
    """
    bbox = None
    label = None

    if not isinstance(obj, dict):
        return None, None

    # bbox 탐색
    for k in ["bbox", "box", "bounding_box"]:
        if k in obj:
            bbox = obj[k]

    # label 탐색
    for k in ["disease", "object_class_code", "crop", "category_id", "label"]:
        if k in obj:
            label = obj[k]

    return bbox, label


count = 0

for root, _, files in os.walk(LABEL_DIR):
    for file in files:
        if not file.endswith(".json"):
            continue

        path = os.path.join(root, file)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # annotation 후보 찾기 (완전 자동)
            candidates = []

            if isinstance(data, dict):
                for k in ["annotations", "objects", "labels"]:
                    if k in data and isinstance(data[k], list):
                        candidates.extend(data[k])

            elif isinstance(data, list):
                candidates = data

            if not candidates:
                continue

            img_name = file.replace(".json", ".jpg")

            img_path = None
            for r, _, fs in os.walk(IMAGE_DIR):
                if img_name in fs:
                    img_path = os.path.join(r, img_name)
                    break

            if not img_path:
                continue

            wrote = False

            for obj in candidates:
                bbox, label = find_bbox_and_label(obj)

                if bbox is None:
                    continue

                if label is None:
                    label = "unknown"

                # bbox 형식 보정
                try:
                    x, y, w, h = bbox
                except:
                    continue

                cls_id = get_class(label)

                shutil.copy(img_path, f"{OUTPUT}/images/train/{img_name}")

                label_file = img_name.replace(".jpg", ".txt")

                with open(f"{OUTPUT}/labels/train/{label_file}", "a") as f:
                    f.write(f"{cls_id} {x} {y} {w} {h}\n")

                wrote = True

            if wrote:
                count += 1

        except Exception as e:
            print("ERROR:", file, e)

print("\n===== DONE =====")
print("TOTAL IMAGES:", count)
print("CLASSES:", class_list)

yaml_path = f"{OUTPUT}/dataset.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {OUTPUT}\n")
    f.write("train: images/train\n")
    f.write("val: images/train\n")
    f.write(f"nc: {len(class_list)}\n")
    f.write(f"names: {class_list}\n")

print("DATASET READY")