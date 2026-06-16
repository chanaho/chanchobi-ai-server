from fastapi import APIRouter
import requests
import os

router = APIRouter()

# =========================
# WEATHER API KEY
# =========================

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# 👉 fallback (임시 테스트용)
if not WEATHER_API_KEY:
    WEATHER_API_KEY = "TEST_MODE_NO_KEY"

# =========================
# WEATHER API
# =========================

@router.get("/weather")
def weather(lat: float, lon: float):

    # 키 없을 때 안내 (크래시 방지)
    if WEATHER_API_KEY == "TEST_MODE_NO_KEY":
        return {
            "success": False,
            "error": "NO_API_KEY",
            "message": "환경변수 WEATHER_API_KEY 설정 필요"
        }

    url = "https://api.openweathermap.org/data/2.5/weather"

    try:
        res = requests.get(url, params={
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "lang": "kr"
        })

        data = res.json()

        return {
            "success": True,
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"],
            "condition": data["weather"][0]["main"]
        }

    except Exception as e:
        return {
            "success": False,
            "error": "WEATHER_API_FAIL",
            "detail": str(e)
        }