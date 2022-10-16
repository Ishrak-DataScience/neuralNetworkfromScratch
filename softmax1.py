import math
layer_output=[4.1,3,2,1.8]
E=math.e
exp_values=[]
for output in layer_output:
    exp_values.append(E**output)

norm_base=sum(exp_values)
norm_values= []

for values in exp_values:
    norm_values.append(values/norm_base)
print(norm_values)
print(sum(norm_values))