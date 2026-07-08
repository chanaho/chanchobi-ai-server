import os

base = r"D:\앱개발\식물병 유발\라벨링데이터"

found = None

for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith(".json"):
            found = os.path.join(root, f)
            break
    if found:
        break

print("FOUND FILE:")
print(found)