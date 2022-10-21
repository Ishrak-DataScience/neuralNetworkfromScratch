import math
math.log
softmax_output=[0.3,0.7,0.5]
tagret_output=[1,0,0]
loss=-(math.log(softmax_output[0])*tagret_output[0]+ math.log(softmax_output[1])*tagret_output[1]+math.log(softmax_output[2])*tagret_output[2])
print(loss)
print(-math.log(0.9))