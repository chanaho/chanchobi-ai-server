# =========================
# 🌾 농업 AI 결과 표준화 엔진
# =========================

def disease_map(class_name: str):
    """
    YOLO class → 농업 병해명 변환
    """

    mapping = {
        "apple_scab": "사과 검은별무늬병",
        "anthracnose": "탄저병",
        "blight": "마름병",
        "rust": "녹병",
        "powdery_mildew": "흰가루병",

        # 기본값
        "default": "미감지"
    }

    return mapping.get(class_name, mapping["default"])


# =========================
# risk 계산
# =========================
def calculate_risk(conf: float):

    if conf is None:
        return "LOW"

    if conf >= 0.85:
        return "HIGH"
    elif conf >= 0.6:
        return "MEDIUM"
    else:
        return "LOW"


# =========================
# 농약 추천
# =========================
def pesticide_advice(disease: str):

    advice = {
        "탄저병": {
            "recommendation": "카브리오 살포",
            "method": "5~7일 간격 살포"
        },
        "사과 검은별무늬병": {
            "recommendation": "다이센M-45",
            "method": "7일 간격 예방 살포"
        },
        "마름병": {
            "recommendation": "기본 방제제",
            "method": "환기 및 제거"
        }
    }

    return advice.get(disease, {
        "recommendation": "기본 관리",
        "method": "정기 관찰"
    })


# =========================
# 🔥 최종 AI 결과 표준화
# =========================
def format_ai_result(yolo_result: dict):

    class_name = yolo_result.get("class", "")
    confidence = float(yolo_result.get("conf", 0))

    disease = disease_map(class_name)
    risk = calculate_risk(confidence)

    pesticide = pesticide_advice(disease)

    return {
        "disease": disease,
        "crop": "고추",  # 필요시 확장 가능

        "risk": risk,
        "confidence": confidence,

        "recommendation": pesticide["recommendation"],
        "method": pesticide["method"],

        "status": "infected" if risk in ["HIGH", "MEDIUM"] else "safe"
    }