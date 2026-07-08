import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset_new"

TRAIN_IMG = DATASET_DIR / "images/train"
TRAIN_LBL = DATASET_DIR / "labels/train"

print("===================================")
print("YOLO DATASET CHECK")
print("===================================")

# ==========================================
# 1. 기본 개수 체크
# ==========================================
images = list(TRAIN_IMG.rglob("*.jpg"))
labels = list(TRAIN_LBL.rglob("*.txt"))

print("이미지 수 :", len(images))
print("라벨 수   :", len(labels))

# ==========================================
# 2. 라벨 파싱 검사
# ==========================================
class_count = {}
empty_labels = 0
error_files = []

for lbl in labels:
    try:
        with open(lbl, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                error_files.append(lbl.name)
                continue

            cls = parts[0]

            class_count[cls] = class_count.get(cls, 0) + 1

    except Exception:
        error_files.append(lbl.name)

# ==========================================
# 3. 결과 출력
# ==========================================
print("\n===================================")
print("CLASS DISTRIBUTION")
print("===================================")

for k, v in sorted(class_count.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
    print(f"Class {k} : {v}")

print("\n===================================")
print("EMPTY LABELS :", empty_labels)
print("ERROR FILES :", len(error_files))

if error_files:
    print("샘플 :", error_files[:10])

# ==========================================
# 4. 품질 판단
# ==========================================
if len(class_count) <= 1:
    print("\n⚠ WARNING: 클래스가 1개뿐입니다 (학습 의미 없음)")
elif max(class_count.values()) / (min(class_count.values() + [1])) > 50:
    print("\n⚠ WARNING: 클래스 불균형 심함")
else:
    print("\n✅ DATASET OK FOR TRAINING")

print("===================================")