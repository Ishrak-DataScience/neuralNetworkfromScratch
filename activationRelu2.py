import numpy as np

input=[0, 1, 2, 3,-10,5,0.5,-0.5]
output=[]
for i in input:
    output.append(max(i,0))
print(output)
