# ai-server/services/automation.py

from collections import defaultdict
from datetime import datetime

# =========================
# 월별 통계 집계
# =========================
def build_monthly_stats(records):
    """
    Firebase 데이터 기반 월별 통계 생성
    """

    stats = defaultdict(lambda: {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
        "TOTAL": 0
    })

    for r in records:
        try:
            created_at = r.get("created_at")
            if not created_at:
                continue

            # YYYY-MM 추출
            month = created_at.strftime("%Y-%m")

            risk = r.get("risk", "LOW")

            stats[month][risk] += 1
            stats[month]["TOTAL"] += 1

        except:
            continue

    return dict(stats)


# =========================
# 농장별 분리 통계
# =========================
def build_farm_stats(records):
    """
    군위 / 안동 농장 분리 통계
    """

    farms = defaultdict(lambda: {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
        "TOTAL": 0
    })

    for r in records:
        farm = r.get("farm", "unknown")
        risk = r.get("risk", "LOW")

        farms[farm][risk] += 1
        farms[farm]["TOTAL"] += 1

    return dict(farms)


# =========================
# 위험도 트렌드 분석
# =========================
def risk_trend(records):
    trend = []

    for r in records:
        trend.append({
            "date": r.get("created_at"),
            "risk": r.get("risk"),
            "crop": r.get("crop"),
            "disease": r.get("disease")
        })

    return trend