import numpy as np

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

class Loss:
    def calculate(self,output,y):
        sample_losses= self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss

class Loss_Categorical_Crossentropy(Loss):
    def forward(self,y_pred, y_true):
        samples=len(y_pred)
        y_pred_clipped= np.clip(y_pred,1e-7,1-1e-7)
        
        if len(y_true.shape)==1:
            correct_confidences=y_pred_clipped[range(samples),y_true]
        
        elif len(y_true.shape)==2:
            correct_confidences=np.sum(y_pred_clipped*y_true,axis=1)
        NegativeLogLikelihood=-np.log(correct_confidences)
        return NegativeLogLikelihood
    

    
X, y= spiral_data(samples=100, classes=3)

dense1= Layer_Dense(2,3)
Activation1=Activation_Relu()
dense2=Layer_Dense(3,3)
Activation2=Activation_Softmax()

dense1.forward(X)
Activation1.forward(dense1.outputs)

dense2.forward(Activation1.output)
Activation2.forward(dense2.outputs)



loss_function= Loss_Categorical_Crossentropy()
loss= loss_function.calculate(Activation2.output,y)
print("loss", loss)
