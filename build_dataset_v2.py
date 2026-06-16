from pathlib import Path
import shutil
import random

SRC = Path("my_farm_photos")
DST = Path("dataset_crop")

if DST.exists():
    shutil.rmtree(DST)

for phase in ["train", "val"]:
    (DST / phase).mkdir(parents=True, exist_ok=True)

for crop in SRC.iterdir():

    if not crop.is_dir():
        continue

    for disease in crop.iterdir():

        if not disease.is_dir():
            continue

        cls = f"{crop.name}_{disease.name}"

        files = list(disease.glob("*.jpg"))

        if len(files) < 2:
            continue

        random.shuffle(files)

        split = int(len(files) * 0.8)

        train = files[:split]
        val = files[split:]

        (DST / "train" / cls).mkdir(parents=True, exist_ok=True)
        (DST / "val" / cls).mkdir(parents=True, exist_ok=True)

        for f in train:
            shutil.copy(f, DST / "train" / cls / f.name)

        for f in val:
            shutil.copy(f, DST / "val" / cls / f.name)

print("완료")