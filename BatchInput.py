import numpy as np
input=[[0, 1, 2, 3],[2,5,1,2],[1.5,3.3,-.8,0.7]]
weights= [[9, 8, 7, 6],[9, 8, 7, 6],[9, 8, 7, 6]]
biases= [10,20,30]

Output= np.dot(input,np.array(weights).T)+biases
print(Output)