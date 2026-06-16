from fastapi import APIRouter
from services.weather_service import get_weather_data
from services.pest_service import get_pest_data

router = APIRouter()

@router.get("/farm/status")
def farm_status(farm_id: str, lat: float, lon: float):

    try:
        # =========================
        # DEFAULT SAFE DATA
        # =========================
        weather = get_weather_data(lat, lon) or {}
        pest = get_pest_data(farm_id) or {}

        temp = weather.get("temp", 25)
        humidity = weather.get("humidity", 50)

        pest_level = pest.get("level", "LOW")

        # =========================
        # SCORE CALC
        # =========================
        score = 0

        if pest_level == "HIGH":
            score += 60
        elif pest_level == "MEDIUM":
            score += 35
        else:
            score += 10

        if temp > 30:
            score += 20

        if humidity > 80:
            score += 15

        # =========================
        # STATUS DECISION
        # =========================
        if score >= 70:
            status = "HIGH"
            action = "즉시 방제"
        elif score >= 40:
            status = "MEDIUM"
            action = "예찰 강화"
        else:
            status = "LOW"
            action = "정상 관리"

        return {
            "success": True,
            "farm_id": farm_id,
            "risk_score": score,
            "status": status,
            "action": action,
            "weather": weather,
            "pest": pest
        }

    except Exception as e:
        return {
            "success": False,
            "error": "FARM_STATUS_ERROR",
            "detail": str(e),
            "farm_id": farm_id,
            "risk_score": 0,
            "status": "LOW",
            "action": "데이터 확인 필요"
        }