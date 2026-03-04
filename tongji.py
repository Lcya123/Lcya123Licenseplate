import os

total=0
for files in os.listdir("./datatset"):
    for file in os.listdir("./datatset/"+files):
        for f in os.listdir("./datatset/"+files+"/"+file):
            total+=1
print(total)
