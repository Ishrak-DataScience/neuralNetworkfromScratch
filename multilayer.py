import numpy as np

input_batch=[[0, 1, 2, 3],
       [2,5,1,2],
       [1.5,3.3,-.8,0.7]]

weights1= [[9, 8, 7, 6],
          [9, 8, 7, 6],
          [9, 8, 7, 6]]
biases1= [10,20,30]

weights2= [[.7, .3, .8],
          [1.2, 1.8, 0.7],
          [1.3, -.8, 1]]
biases2= [5,-20,10]


layer1_output= np.dot(input_batch,np.array(weights1).T)+biases1
layer2_output= np.dot(layer1_output,np.array(weights2).T)+biases2

print(layer2_output)
