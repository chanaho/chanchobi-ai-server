import os

base = "dataset_chanchobi_cls_v2/train"

classes = sorted(
    [
        x for x in os.listdir(base)
        if os.path.isdir(
            os.path.join(base,x)
        )
    ]
)

print("클래스 수:", len(classes))

for i,c in enumerate(classes):
    print(i, c)


with open(
    "classes.txt",
    "w",
    encoding="utf-8"
) as f:

    for c in classes:
        f.write(c+"\n")


print("classes.txt 생성 완료")