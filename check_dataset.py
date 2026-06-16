import os

# =========================
# 설정 (여기만 바꾸면 재사용 가능)
# =========================
DATASET_PATH = "dataset_crop"

# =========================
# 체크 시작
# =========================
print("\n===== YOLO CLASSIFY DATASET CHECK =====")

base_path = os.path.abspath(DATASET_PATH)
print("📁 Dataset Path:", base_path)

# train / val 체크
for split in ["train", "val"]:
    split_path = os.path.join(base_path, split)

    print(f"\n🔎 [{split}]")
    print("경로:", split_path)
    print("존재:", os.path.exists(split_path))

    if not os.path.exists(split_path):
        print("❌ 폴더 없음")
        continue

    classes = os.listdir(split_path)

    if len(classes) == 0:
        print("❌ 클래스 폴더 없음")
        continue

    print("📂 클래스 목록:", classes)

    total_images = 0

    for cls in classes:
        cls_path = os.path.join(split_path, cls)

        if not os.path.isdir(cls_path):
            continue

        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"   └ {cls}: {len(images)}장")
        total_images += len(images)

    print("📊 총 이미지 수:", total_images)

print("\n===== CHECK END =====\n")