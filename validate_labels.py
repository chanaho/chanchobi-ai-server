import os

# =========================
# CONFIG
# =========================
DATASET_DIR = "dataset_clean"

IMG_DIRS = [
    f"{DATASET_DIR}/images/train",
    f"{DATASET_DIR}/images/val",
]

LBL_DIRS = [
    f"{DATASET_DIR}/labels/train",
    f"{DATASET_DIR}/labels/val",
]

# =========================
# CHECK RESULTS
# =========================
errors = []
class_count = {}

def check_label_file(label_path):
    if not os.path.exists(label_path):
        errors.append(f"❌ Missing label: {label_path}")
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            errors.append(f"❌ Format error: {label_path} -> {line}")
            continue

        cls, x, y, w, h = parts

        try:
            cls = int(cls)
            x, y, w, h = map(float, (x, y, w, h))
        except:
            errors.append(f"❌ Parse error: {label_path} -> {line}")
            continue

        # class count
        class_count[cls] = class_count.get(cls, 0) + 1

        # class id check
        if cls < 0:
            errors.append(f"❌ Invalid class id: {label_path} -> {cls}")

        # bbox range check
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            errors.append(f"❌ bbox out of range: {label_path} -> {line}")

# =========================
# MAIN
# =========================
def main():
    print("🔍 VALIDATING DATASET...\n")

    total_images = 0

    for lbl_dir in LBL_DIRS:
        if not os.path.exists(lbl_dir):
            print(f"⚠️ Missing folder: {lbl_dir}")
            continue

        for file in os.listdir(lbl_dir):
            if file.endswith(".txt"):
                path = os.path.join(lbl_dir, file)
                check_label_file(path)

    for img_dir in IMG_DIRS:
        if os.path.exists(img_dir):
            total_images += len(os.listdir(img_dir))

    # =========================
    # REPORT
    # =========================
    print("\n=========================")
    print("📊 DATASET REPORT")
    print("=========================")

    print(f"Total Images: {total_images}")
    print(f"Total Classes: {len(class_count)}")
    print(f"Class Distribution: {class_count}")

    if len(errors) == 0:
        print("\n✅ NO ERRORS FOUND")
    else:
        print(f"\n❌ ERRORS FOUND: {len(errors)}")
        for e in errors[:20]:
            print(e)

    print("\nDONE")

if __name__ == "__main__":
    main()