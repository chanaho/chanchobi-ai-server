import os
from collections import defaultdict

label_paths = [
    r"dataset_clean/labels/train",
    r"dataset_clean/labels/val"
]

class_counter = defaultdict(int)
valid_files = 0
empty_files = 0

print("\n===== 클래스 분석 시작 =====\n")

for base in label_paths:
    if not os.path.exists(base):
        continue

    for f in os.listdir(base):
        path = os.path.join(base, f)

        if not f.endswith(".txt"):
            continue

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            cls = int(parts[0])
            class_counter[cls] += 1

        valid_files += 1

print("\n===== 결과 =====")
print("유효 파일:", valid_files)
print("빈 파일:", empty_files)
print("\n클래스 분포:")

for k in sorted(class_counter.keys()):
    print(f"class{k}: {class_counter[k]}")

print("\n===== 추천 =====")
print("실제 사용 클래스 수:", len(class_counter))
print("YOLO용 nc 추천값:", len(class_counter))