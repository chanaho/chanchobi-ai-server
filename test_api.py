import requests

url = "http://127.0.0.1:8000/predict"

files = {"file": open("test.jpg", "rb")}

res = requests.post(url, files=files)

print(res.json())