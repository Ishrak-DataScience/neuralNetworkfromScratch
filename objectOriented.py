import numpy as np
np.random.seed(0)

X=[[0, 1, 2, 3],
       [2,5,1,2],
       [1.5,3.3,-.8,0.7]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
       
    def forward(self, inputs):
       self.outputs=np.dot(inputs,self.weights)+self.biases
       
layer1=Layer_Dense(4,3)
layer2=Layer_Dense(3,2)

layer1.forward(X)
#print(layer1.outputs)
layer2.forward(layer1.outputs)
print(layer2.outputs)