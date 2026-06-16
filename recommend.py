RECOMMEND_MAP = {
    "자두_잉크병": "구리제 + 항균제",
    "사과_탄저병": "보르도액",
    "포도_노균병": "메탈락실",
}

def get_recommend(disease):
    return RECOMMEND_MAP.get(disease, "-")