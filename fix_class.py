import glob

files = glob.glob(
    "dataset_chanchobi_final/labels/**/*.txt",
    recursive=True
)

count = 0

for f in files:

    with open(f, "r", encoding="utf-8") as x:
        lines = x.readlines()

    new_lines = []

    for line in lines:

        p = line.strip().split()

        if len(p) == 5:

            if p[0] == "11":
                p[0] = "0"
                count += 1

            new_lines.append(" ".join(p) + "\n")

    with open(f, "w", encoding="utf-8") as x:
        x.writelines(new_lines)

print("변경 완료:", count)