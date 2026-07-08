import json
import shutil
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = Path(r"D:\앱개발\식물병 유발")

OUTPUT_DIR = BASE_DIR / "dataset_new"

TRAIN_IMAGES = OUTPUT_DIR / "images/train"
VAL_IMAGES = OUTPUT_DIR / "images/val"
TRAIN_LABELS = OUTPUT_DIR / "labels/train"
VAL_LABELS = OUTPUT_DIR / "labels/val"

for p in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
    p.mkdir(parents=True, exist_ok=True)

print("=== FIXED DATASET BUILD ===")

# ======================================
# 핵심: TL1 / TS1 구조 대응
# ======================================
LABEL_ROOT = SOURCE_DIR / "라벨링데이터"
IMAGE_ROOT = SOURCE_DIR / "원천데이터"

image_files = list(IMAGE_ROOT.rglob("*.jpg"))
json_files = list(LABEL_ROOT.rglob("*.json"))

print("images:", len(image_files))
print("labels:", len(json_files))

# ======================================
# 라벨 매칭 (핵심 수정)
# ======================================
def find_json_by_name(name):
    for js in json_files:
        if js.stem == name:
            return js
    return None

class_map = {}
class_index = 0

def get_class_id(name):
    global class_index
    if name not in class_map:
        class_map[name] = class_index
        class_index += 1
    return class_map[name]

def get_label(data):
    anno = data.get("annotations", {})
    if isinstance(anno, dict):
        if anno.get("disease"):
            return str(anno["disease"])
        if anno.get("object_class_code"):
            return str(anno["object_class_code"])
    return None

def convert_bbox(w, h, bbox):
    x = float(bbox["x"])
    y = float(bbox["y"])
    bw = float(bbox["w"])
    bh = float(bbox["h"])

    cx = (x + bw/2) / w
    cy = (y + bh/2) / h
    nw = bw / w
    nh = bh / h

    return cx, cy, nw, nh

items = []

for img in image_files:
    js = find_json_by_name(img.stem)
    if js:
        items.append((img, js))

random.shuffle(items)

split = int(len(items) * 0.8)
train = items[:split]
val = items[split:]

def process(data, img_out, label_out):
    ok = 0
    fail = 0

    for img, js in data:
        try:
            with open(js, "r", encoding="utf-8") as f:
                d = json.load(f)

            desc = d.get("description", {})
            anno = d.get("annotations", {})

            if not isinstance(anno, dict):
                fail += 1
                continue

            w = float(desc.get("width", 0))
            h = float(desc.get("height", 0))

            if w == 0 or h == 0:
                fail += 1
                continue

            bbox = anno.get("bbox", [])
            if not bbox:
                fail += 1
                continue

            label = get_label(d)
            if not label:
                fail += 1
                continue

            cls = get_class_id(label)

            cx, cy, nw, nh = convert_bbox(w, h, bbox[0])

            shutil.copy(img, img_out / img.name)

            with open(label_out / f"{img.stem}.txt", "w") as f:
                f.write(f"{cls} {cx} {cy} {nw} {nh}\n")

            ok += 1

        except Exception:
            fail += 1

    print("OK:", ok, "FAIL:", fail)

process(train, TRAIN_IMAGES, TRAIN_LABELS)
process(val, VAL_IMAGES, VAL_LABELS)

with open(OUTPUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
    for k, v in class_map.items():
        f.write(f"{v} {k}\n")

print("\nDONE")
print("CLASSES:", len(class_map))
print(class_map)