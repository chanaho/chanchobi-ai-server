# ai-server/services/farm_logic.py

from datetime import datetime, timedelta

# =========================
# 위험도 기준
# =========================
RISK_LEVEL = {
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "UNKNOWN": 0
}


# =========================
# 농장별 태그 정리
# =========================
def parse_farm(farm: str):
    if not farm:
        return "unknown"

    farm = farm.lower()

    if "군위" in farm:
        return "gunwi"
    if "안동" in farm:
        return "andong"

    return "unknown"


# =========================
# 위험도 판단
# =========================
def is_high_risk(risk: str):
    return RISK_LEVEL.get(risk, 0) >= 3


# =========================
# 일정 자동 생성
# =========================
def create_schedule(risk: str):
    """
    HIGH → 즉시 방제
    MEDIUM → 3일 내
    LOW → 관찰
    """

    now = datetime.utcnow()

    if risk == "HIGH":
        return {
            "action": "immediate_spray",
            "date": now.strftime("%Y-%m-%d"),
            "message": "즉시 방제 필요"
        }

    elif risk == "MEDIUM":
        return {
            "action": "spray",
            "date": (now + timedelta(days=3)).strftime("%Y-%m-%d"),
            "message": "3일 내 방제 권장"
        }

    else:
        return {
            "action": "observe",
            "date": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            "message": "관찰 유지"
        }


# =========================
# 알림 생성
# =========================
def create_alert(disease: str, risk: str):
    if risk == "HIGH":
        return f"🚨 긴급: {disease} 발생 (즉시 방제 필요)"

    elif risk == "MEDIUM":
        return f"⚠️ 주의: {disease} 확산 가능성 있음"

    else:
        return f"ℹ️ {disease} 상태 관찰 필요"