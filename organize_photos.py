from pathlib import Path
import shutil

ROOT = Path("my_farm_photos")

for crop_dir in ROOT.iterdir():

    if not crop_dir.is_dir():
        continue

    crop_name = crop_dir.name

    for file in list(crop_dir.glob("*.jpg")):

        filename = file.stem

        parts = filename.split("_")

        disease = None

        if len(parts) >= 2:

            if parts[0] == crop_name:
                disease = "_".join(parts[1:])

        if disease:

            target_dir = crop_dir / disease

        else:

            target_dir = crop_dir / "정상"

        target_dir.mkdir(exist_ok=True)

        target_file = target_dir / file.name

        shutil.move(str(file), str(target_file))

        print(f"[MOVE] {file.name} -> {target_dir}")