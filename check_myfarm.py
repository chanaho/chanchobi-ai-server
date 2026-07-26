import os

base = "my_farm_photos"

grand_total = 0

for crop in sorted(os.listdir(base)):
    crop_path = os.path.join(base, crop)

    if not os.path.isdir(crop_path):
        continue

    crop_total = 0

    for root, dirs, files in os.walk(crop_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                crop_total += 1

    print(f"{crop:15} : {crop_total}")

    grand_total += crop_total

print("-" * 40)
print("전체 사진 :", grand_total)