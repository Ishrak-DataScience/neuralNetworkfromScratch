import numpy as np

input=[0, 1, 2, 3,-10,5,0.5,-0.5]
output=[]
for i in input:
    if i>0:
        output.append(i)
    elif i<=0:
        output.append(0)
print(output)
