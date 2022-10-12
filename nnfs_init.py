import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
X, y = spiral_data(100, 3) 
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
        #if many zeros after activation pass , set positive/larger biases
       
    def forward(self, inputs):
       self.outputs=np.dot(inputs,self.weights)+self.biases
class Activation_Relu:
    def forward(self,input):
        self.output=np.maximum(0,input)
       
layer1=Layer_Dense(2,10)
activation1=Activation_Relu()
layer1.forward(X)
#print(layer1.outputs)
activation1.forward(layer1.outputs)
print(activation1.output)
