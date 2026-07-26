import glob


files = glob.glob(
    "dataset_aihub_pepper/labels/**/*.txt",
    recursive=True
)


count = 0


for f in files:

    with open(f, "r") as file:
        lines = file.readlines()


    new_lines = []


    for line in lines:

        parts = line.strip().split()

        if len(parts) == 5:

            # AI HUB 고추 병해 class 0 -> 11
            parts[0] = "11"

            new_lines.append(
                " ".join(parts)
            )


    with open(f, "w") as file:
        file.write(
            "\n".join(new_lines)
        )


    count += 1


print("변경 완료:", count)