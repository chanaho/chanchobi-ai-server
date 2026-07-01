# disease_db.py

DISEASE_DB = {
    "고추_탄저병": {
        "name": "탄저병",
        "crop": "고추",
        "risk": "HIGH",
        "chemical": ["아족시스트로빈", "디페노코나졸"],
        "method": "교호살포",
        "note": "고온다습 환경에서 급속 확산",
        "warning": "비 온 후 24시간 내 방제 권장"
    },

    "사과_갈색무늬병": {
        "name": "갈색무늬병",
        "crop": "사과",
        "risk": "HIGH",
        "chemical": ["카벤다짐", "만코제브"],
        "method": "예방 위주 방제",
        "note": "장마철 집중 발생",
        "warning": "잎 조기 낙엽 유발"
    },

    "자두_세균성구멍병": {
        "name": "세균성구멍병",
        "crop": "자두",
        "risk": "HIGH",
        "chemical": ["구리제", "스트렙토마이신"],
        "method": "예방 살포 중요",
        "note": "비 후 전염 확산",
        "warning": "과실 품질 저하"
    },

    "아로니아_잿빛곰팡이병": {
        "name": "잿빛곰팡이병",
        "crop": "아로니아",
        "risk": "MEDIUM",
        "chemical": ["프로클로라즈"],
        "method": "환기 및 전정",
        "note": "과습 환경 주의",
        "warning": "수확량 감소"
    },

    "복숭아_세균성구멍병": {
        "name": "세균성구멍병",
        "crop": "복숭아",
        "risk": "HIGH",
        "chemical": ["구리제"],
        "method": "비 전후 예방 살포",
        "note": "잎 구멍 발생",
        "warning": "조기 낙엽"
    },

    "체리_갈색무늬병": {
        "name": "갈색무늬병",
        "crop": "체리",
        "risk": "MEDIUM",
        "chemical": ["만코제브"],
        "method": "기본 방제",
        "note": "고온다습 조건 발생",
        "warning": "광합성 저하"
    },

    "포도_노균병": {
        "name": "노균병",
        "crop": "포도",
        "risk": "HIGH",
        "chemical": ["메탈락실"],
        "method": "예방 중심",
        "note": "습도 매우 중요",
        "warning": "급속 확산"
    },

    "블루베리_잿빛곰팡이병": {
        "name": "잿빛곰팡이병",
        "crop": "블루베리",
        "risk": "MEDIUM",
        "chemical": ["프로클로라즈"],
        "method": "환기 강화",
        "note": "밀식 주의",
        "warning": "수확 직전 피해"
    },

    "토마토_잎마름병": {
        "name": "잎마름병",
        "crop": "토마토",
        "risk": "HIGH",
        "chemical": ["클로로탈로닐"],
        "method": "주기적 방제",
        "note": "잎부터 시작",
        "warning": "전체 고사 가능"
    },

    "오이_노균병": {
        "name": "노균병",
        "crop": "오이",
        "risk": "HIGH",
        "chemical": ["디메토모르프"],
        "method": "예방 필수",
        "note": "습도 영향 큼",
        "warning": "급격한 확산"
    },

    "수박_덩굴마름병": {
        "name": "덩굴마름병",
        "crop": "수박",
        "risk": "HIGH",
        "chemical": ["티오파네이트메틸"],
        "method": "토양 관리",
        "note": "연작 피해",
        "warning": "덩굴 전체 고사"
    }
}


def get_disease_info(label: str):
    """YOLO label → 농업 정보 변환"""
    return DISEASE_DB.get(label, {
        "name": "알 수 없음",
        "crop": "unknown",
        "risk": "UNKNOWN",
        "chemical": [],
        "method": "추가 분석 필요",
        "note": "모델 미학습 데이터",
        "warning": "수동 확인 필요"
    })