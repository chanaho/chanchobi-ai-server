from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_new"

TRAIN_DIR = "images/train"
VAL_DIR = "images/val"

# ==========================================
# classes.txt 읽기
# ==========================================
classes_path = DATASET_DIR / "classes.txt"

if not classes_path.exists():
    raise Exception("classes.txt 없음 - 먼저 convert_dataset 실행 필요")

names = {}

with open(classes_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            idx, name = line.strip().split(" ", 1)
            names[int(idx)] = name

names_str = "\n".join([f"  {k}: {v}" for k, v in sorted(names.items())])

# ==========================================
# YAML 생성
# ==========================================
yaml_content = f"""path: {DATASET_DIR.as_posix()}

train: {TRAIN_DIR}
val: {VAL_DIR}

nc: {len(names)}

names:
{names_str}
"""

yaml_path = DATASET_DIR / "dataset.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("===================================")
print("YOLO DATASET YAML CREATED")
print("===================================")
print("PATH :", yaml_path)
print("CLASSES :", len(names))
print("===================================")