import os

base = "runs/detect"

for folder in os.listdir(base):
    path = os.path.join(base, folder)
    if os.path.isdir(path):
        print(folder)