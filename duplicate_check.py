from pathlib import Path
import hashlib

folder = Path("my_farm_photos/사과")

hashes = {}
duplicates = []

for file in folder.rglob("*.jpg"):
    try:
        h = hashlib.md5(file.read_bytes()).hexdigest()

        if h in hashes:
            duplicates.append((file.name, hashes[h]))
        else:
            hashes[h] = file.name

    except:
        pass

print("전체 사진:", len(list(folder.rglob("*.jpg"))))
print("중복 사진:", len(duplicates))

for d in duplicates[:20]:
    print(d)