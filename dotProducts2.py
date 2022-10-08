import numpy as np
input=[0, 1, 2, 3]
weights= [[9, 8, 7, 6],[9, 8, 7, 6],[9, 8, 7, 6]]
biases= [10,20,30]

Output= np.dot(weights,input)+biases
print(Output)