import os
import json
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# ======================
# PATH 설정
# ======================
JSON_ROOT = r"D:\앱개발\식물병 유발\라벨링데이터"
IMG_ROOT = r"D:\앱개발\식물병 유발\원천데이터"

OUT_IMG = "dataset_new/images"
OUT_LABEL = "dataset_new/labels"

os.makedirs(OUT_IMG + "/train", exist_ok=True)
os.makedirs(OUT_IMG + "/val", exist_ok=True)
os.makedirs(OUT_LABEL + "/train", exist_ok=True)
os.makedirs(OUT_LABEL + "/val", exist_ok=True)

data = []

# ======================
# JSON 수집
# ======================
json_files = glob(JSON_ROOT + "/**/*.json", recursive=True)

classes = set()

for jf in json_files:
    try:
        with open(jf, "r", encoding="utf-8") as f:
            j = json.load(f)

        label = j["Annotations"]["OBJECT_CLASS_CODE"]
        img_id = j["Info"]["IMAGE_FILE_NM"]

        if not label:
            continue

        classes.add(label)

        img_path = glob(IMG_ROOT + f"/**/{img_id}.jpg", recursive=True)
        if not img_path:
            continue

        data.append((img_path[0], label))

    except Exception as e:
        continue

# ======================
# CLASS MAP
# ======================
classes = sorted(list(classes))
class_map = {c:i for i,c in enumerate(classes)}

# ======================
# SPLIT
# ======================
train, val = train_test_split(data, test_size=0.2, random_state=42)

def save(split, mode):
    for img_path, label in split:
        img_name = os.path.basename(img_path)

        shutil.copy(img_path, f"{OUT_IMG}/{mode}/{img_name}")

        txt_name = img_name.replace(".jpg", ".txt")
        with open(f"{OUT_LABEL}/{mode}/{txt_name}", "w") as f:
            f.write(f"{class_map[label]} 0.5 0.5 1.0 1.0")

save(train, "train")
save(val, "val")

# ======================
# YAML 생성
# ======================
with open("dataset_new/dataset.yaml", "w", encoding="utf-8") as f:
    f.write("train: dataset_new/images/train\n")
    f.write("val: dataset_new/images/val\n")
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {classes}\n")

print("===== DONE =====")
print("IMAGES:", len(data))
print("CLASSES:", classes)
print("DATASET READY")