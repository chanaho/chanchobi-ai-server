import json
import random
import shutil
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

print("=== LABEL REBUILD START ===")

RAW_IMAGE_DIR = SOURCE_DIR / "원천데이터"
RAW_LABEL_DIR = SOURCE_DIR / "라벨링데이터"

image_files = list(RAW_IMAGE_DIR.rglob("*.jpg"))
json_files = list(RAW_LABEL_DIR.rglob("*.json"))

print("images:", len(image_files))
print("labels:", len(json_files))

# ==========================================
# 🔥 AI-Hub 핵심 복구 매핑
# ==========================================
DISEASE_MAP = {
    "a7": "감귤_궤양병",
    "a1": "감귤_정상",
    "a2": "감귤_귤응애",
    "a3": "감귤_진딧물",

    "b1": "고추_탄저병",
    "b2": "고추_정상",

    "c1": "키위_점무늬병",
    "c2": "키위_정상",
}

def get_label(anno):
    if not isinstance(anno, dict):
        return None

    raw = anno.get("disease")

    if not raw:
        raw = anno.get("object_class_code")

    if not raw:
        return None

    raw = str(raw).strip()

    # 🔥 핵심: 코드 → 실제 병명 변환
    if raw in DISEASE_MAP:
        return DISEASE_MAP[raw]

    # 이미 완성형이면 그대로 사용
    if "_" in raw:
        return raw

    return None


def convert_bbox(w, h, bbox):
    x = float(bbox.get("x", 0))
    y = float(bbox.get("y", 0))
    bw = float(bbox.get("w", 0))
    bh = float(bbox.get("h", 0))

    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    nw = bw / w
    nh = bh / h

    return cx, cy, nw, nh


class_map = {}
class_index = 0

def get_class_id(name):
    global class_index
    if name not in class_map:
        class_map[name] = class_index
        class_index += 1
    return class_map[name]


pairs = {}

for img in image_files:
    pairs[img.stem] = {"img": img, "json": None}

for js in json_files:
    pairs.setdefault(js.stem, {"img": None, "json": None})
    pairs[js.stem]["json"] = js

items = [v for v in pairs.values() if v["img"] and v["json"]]

random.shuffle(items)

split = int(len(items) * 0.8)
train_items = items[:split]
val_items = items[split:]


def process(items, img_out, label_out):
    ok, fail = 0, 0

    for it in items:
        try:
            img = it["img"]
            js = it["json"]

            with open(js, "r", encoding="utf-8") as f:
                data = json.load(f)

            desc = data.get("description", {})
            anno = data.get("annotations", {})

            if not isinstance(anno, dict):
                fail += 1
                continue

            w = float(desc.get("width", 0))
            h = float(desc.get("height", 0))

            if w == 0 or h == 0:
                fail += 1
                continue

            bbox_list = anno.get("bbox", [])
            if not bbox_list:
                fail += 1
                continue

            bbox = bbox_list[0]

            label = get_label(anno)
            if not label:
                fail += 1
                continue

            cx, cy, nw, nh = convert_bbox(w, h, bbox)

            cls = get_class_id(label)

            shutil.copy(img, img_out / img.name)

            with open(label_out / f"{img.stem}.txt", "w") as f:
                f.write(f"{cls} {cx} {cy} {nw} {nh}\n")

            ok += 1

        except Exception:
            fail += 1

    print("OK:", ok, "FAIL:", fail)


process(train_items, TRAIN_IMAGES, TRAIN_LABELS)
process(val_items, VAL_IMAGES, VAL_LABELS)

with open(OUTPUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
    for k, v in class_map.items():
        f.write(f"{v} {k}\n")

print("\n===================================")
print("DONE")
print("TOTAL CLASSES:", len(class_map))
print(class_map)
print("===================================")