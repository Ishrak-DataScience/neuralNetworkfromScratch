import numpy as np
np.random.seed(0)
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
       
    def forward(self, inputs):
       self.outputs=np.dot(inputs,self.weights)+self.biases
class Activation_Relu:
    def forward(self,input):
        self.output=np.maximum(0,input)
        
class Activation_Softmax:
    def forward(self,input):
        exp_values= np.exp(input-np.max(input,axis=1,keepdims=True))
        norm_base=np.sum(exp_values,axis=1,keepdims=True)
        self.output=exp_values/norm_base 

    
X, y= spiral_data(samples=100, classes=3)

dense1= Layer_Dense(2,3)
Activation1=Activation_Relu()
dense2=Layer_Dense(3,3)
Activation2=Activation_Softmax()

dense1.forward(X)
Activation1.forward(dense1.outputs)

dense2.forward(Activation1.output)
Activation2.forward(dense2.outputs)

print(Activation2.output[:5])
