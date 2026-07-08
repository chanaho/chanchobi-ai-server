import json

path = r"D:\앱개발\식물병 유발\라벨링데이터\TL1.감귤\열매_궤양병\HF01_01FT_000001.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("===== JSON TOP LEVEL KEYS =====")
print(list(data.keys()))

print("\n===== FULL JSON SAMPLE (CUT) =====")
print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])