import requests
import os

def get_weather_data(lat, lon):

    key = os.getenv("WEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    res = requests.get(url, params={
        "lat": lat,
        "lon": lon,
        "appid": key,
        "units": "metric"
    })

    data = res.json()

    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"]
    }