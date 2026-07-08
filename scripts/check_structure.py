from pathlib import Path

BASE = Path(r"D:\앱개발\식물병 유발")

print("=== STRUCTURE CHECK ===")

for p in BASE.rglob("*.json"):
    print(p)
    break

for p in BASE.rglob("*.jpg"):
    print(p)
    break