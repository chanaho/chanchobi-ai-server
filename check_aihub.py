import os
import json


base = r"D:\앱개발\식물병 유발"


folders = [
    r"라벨링데이터\TL1.감귤\열매_궤양병",
    r"라벨링데이터\TL1.감귤\열매_정상",
    r"라벨링데이터\TL1_고추\병해",
]


for f in folders:

    path = os.path.join(base, f)

    print("\n================")
    print(path)

    files = os.listdir(path)

    jsons = [
        x for x in files
        if x.endswith(".json")
    ]

    print("JSON:", len(jsons))

    if jsons:

        sample = os.path.join(path, jsons[0])

        with open(sample, encoding="utf-8") as fp:
            data=json.load(fp)

        print(
            "CLASS:",
            data["Annotations"]["OBJECT_CLASS_CODE"]
        )