import os
import random
import shutil

base = "dataset_clean/labels/train"

files = [f for f in os.listdir(base) if f.endswith(".txt")]

empty_files = []
valid_files = []

for f in files:
    path = os.path.join(base, f)
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if len(lines) == 0:
        empty_files.append(f)
    else:
        valid_files.append(f)

print("전체:", len(files))
print("정상:", len(valid_files))
print("빈 파일:", len(empty_files))

# 🔥 빈 파일 일부 삭제 (50%만 유지)
delete_count = len(empty_files) // 2
to_delete = random.sample(empty_files, delete_count)

for f in to_delete:
    os.remove(os.path.join(base, f))

print("\n삭제된 빈 라벨:", len(to_delete))