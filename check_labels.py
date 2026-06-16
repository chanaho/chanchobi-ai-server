import os
from collections import Counter

BASE_PATH = r"C:\Users\user\chanchobi-app\ai-server\dataset"

label_dirs = [
    os.path.join(BASE_PATH, "labels/train"),
    os.path.join(BASE_PATH, "labels/val")
]

class_counter = Counter()
total_files = 0

print("===== 라벨 경로 확인 =====")
for d in label_dirs:
    print("확인 경로:", d)

print("\n===== 분석 시작 =====\n")

for d in label_dirs:
    if not os.path.exists(d):
        print(f"❌ 폴더 없음: {d}")
        continue

    for file in os.listdir(d):
        if not file.endswith(".txt"):
            continue

        total_files += 1
        path = os.path.join(d, file)

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            cls = int(parts[0])
            class_counter[cls] += 1

print("\n===== 결과 =====")
print("총 라벨 파일 수:", total_files)
print("클래스 분포:", dict(class_counter))