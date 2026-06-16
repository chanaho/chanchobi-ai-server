import os
import cv2
import albumentations as A

img_dir = "dataset_clean/images/train"
out_dir = "dataset_clean/images/train_aug"

os.makedirs(out_dir, exist_ok=True)

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Blur(p=0.2),
    A.Rotate(limit=15, p=0.5),
])

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    augmented = transform(image=img)["image"]

    cv2.imwrite(os.path.join(out_dir, "aug_" + img_name), augmented)

print("DONE AUGMENT")