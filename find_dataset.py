import os

base_path = "C:/Users/user/chanchobi-app/ai-server"

print("\n===== 데이터셋 위치 탐색 =====")

for root, dirs, files in os.walk(base_path):
    if "labels" in root.lower():
        txt_files = [f for f in files if f.endswith(".txt")]
        if len(txt_files) > 0:
            print("\n[발견됨]")
            print("경로:", root)
            print("라벨 수:", len(txt_files))