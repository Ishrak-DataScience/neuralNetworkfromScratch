import numpy as np
import nnfs
nnfs.init()
layer_output=[[4.1,3,2,1.8],
              [3,1,2,5],
              [1.2,3.1,4.2,3.3]]

exp_values=np.exp(layer_output)

norm_base=np.sum(exp_values,axis=1,keepdims=True)
norm_values=exp_values/norm_base 


print(norm_values)
