# class_map.py

DISEASE_MAP = {
    # =========================
    # 감귤
    # =========================
    "a1": "감귤_정상",
    "a7": "감귤_궤양병",
    "a2": "감귤_귤응애",
    "a3": "감귤_진딧물",

    # =========================
    # 고추
    # =========================
    "b1": "고추_탄저병",
    "b2": "고추_정상",

    # =========================
    # 키위
    # =========================
    "c1": "키위_점무늬병",
    "c2": "키위_정상",

    # =========================
    # fallback
    # =========================
}

def normalize_label(label: str):
    if not label:
        return "unknown"

    label = str(label).strip()

    if label in DISEASE_MAP:
        return DISEASE_MAP[label]

    return label