# ai-server/services/dashboard_builder.py

from datetime import datetime, timedelta


# =========================
# 오늘 요약
# =========================
def build_today_summary(records):
    today = datetime.utcnow().date()

    summary = {
        "total": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
        "diseases": {}
    }

    for r in records:
        try:
            created_at = r.get("created_at")
            if not created_at:
                continue

            if created_at.date() != today:
                continue

            risk = r.get("risk", "LOW")
            disease = r.get("disease", "unknown")

            summary["total"] += 1
            summary[risk] += 1

            summary["diseases"][disease] = summary["diseases"].get(disease, 0) + 1

        except:
            continue

    return summary


# =========================
# 이번주 요약
# =========================
def build_week_summary(records):
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)

    summary = {
        "total": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }

    for r in records:
        try:
            created_at = r.get("created_at")
            if not created_at:
                continue

            if created_at < week_ago:
                continue

            risk = r.get("risk", "LOW")

            summary["total"] += 1
            summary[risk] += 1

        except:
            continue

    return summary


# =========================
# 핵심 대시보드
# =========================
def build_dashboard(records):
    return {
        "today": build_today_summary(records),
        "week": build_week_summary(records),
        "total_records": len(records)
    }