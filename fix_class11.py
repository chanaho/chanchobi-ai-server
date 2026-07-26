import glob

count=0

files=glob.glob(
    "dataset_chanchobi_final/labels/**/*.txt",
    recursive=True
)

for f in files:

    new=[]

    with open(f,encoding="utf-8") as x:
        lines=x.readlines()

    for line in lines:

        p=line.strip().split()

        if len(p)==5:

            if p[0]=="11":
                p[0]="0"
                count+=1

            new.append(" ".join(p)+"\n")


    with open(f,"w",encoding="utf-8") as x:
        x.writelines(new)


print("변경:",count)