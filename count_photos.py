from pathlib import Path

root = Path("my_farm_photos")

print("\n===== 작물별 이미지 수 =====\n")

total = 0

for crop in sorted(root.iterdir()):
    if crop.is_dir():
        count = len(list(crop.rglob("*.jpg")))
        total += count
        print(f"{crop.name}: {count}")

print("\n-------------------")
print("총 이미지 수:", total)