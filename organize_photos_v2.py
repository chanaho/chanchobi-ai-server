from pathlib import Path
import shutil

ROOT = Path("my_farm_photos")

# 영문 파일명 매핑
disease_map = {
    "sagwa_tanjeo": ("사과", "탄저병"),
    "sagwa_hingaru": ("사과", "흰가루병"),
    "sagwa_scab": ("사과", "갈색무늬병"),

    "aronia_botrytis": ("아로니아", "잿빛곰팡이병"),
    "aronia_leafspot": ("아로니아", "갈색무늬병"),

    "gochu_tanjeobyeong": ("고추", "탄저병"),
}

for crop_dir in ROOT.iterdir():

    if not crop_dir.is_dir():
        continue

    crop_name = crop_dir.name

    files = list(crop_dir.glob("*.jpg"))

    for file in files:

        name = file.stem

        moved = False

        # 영문 패턴 처리
        for key, (crop, disease) in disease_map.items():

            if key.lower() in name.lower():

                target = ROOT / crop / disease
                target.mkdir(parents=True, exist_ok=True)

                shutil.move(
                    str(file),
                    str(target / file.name)
                )

                print(
                    f"[영문병명] {file.name} -> {crop}/{disease}"
                )

                moved = True
                break

        if moved:
            continue

        # 한글 패턴 처리
        parts = name.split("_")

        if len(parts) >= 2 and parts[0] == crop_name:

            disease = "_".join(parts[1:])

            target = crop_dir / disease

        else:

            target = crop_dir / "정상"

        target.mkdir(exist_ok=True)

        shutil.move(
            str(file),
            str(target / file.name)
        )

        print(
            f"[이동] {file.name} -> {target}"
        )

print("\n완료")