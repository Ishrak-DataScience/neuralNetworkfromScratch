input=[0, 1, 2, 3]

weights= [[9, 8, 7, 6],[9, 8, 7, 6],[9, 8, 7, 6]]
biases= [10,20,30]
layer_output= []
for neural_weights,neural_biases in zip(weights,biases):
    neuron_output=0
    for n_input,weight in zip(input,neural_weights):
        neuron_output+=n_input*weight
    neuron_output+=neural_biases
    layer_output.append(neuron_output)
print(layer_output)
    