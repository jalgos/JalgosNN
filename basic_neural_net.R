## JalgosNN 
## A feed forward Neural Net is simply a sequence of matrix multiplication and activation function

## Multilayer perceptron is a NN where each layer is fully connected to the previous one
## Using S4 class for representation

library(Matrix)
library(ggplot2)
setClass("NNLayer",
         slots = c(layer_id = "character",
         neurons = "index",
         input_neurons = "index",    
         activation_function = "function",
         activation_arguments = "list",
         synapses = "list"))
         
NNLayer <- function(L,
                    layer_id,
                    input_neurons,
                    last_neuron)
{
    if(!is.null(L$layer_id)) layer_id = L$layer_id
    nb_neurons = L$nb_neurons
    neurons = last_neuron + 1:nb_neurons
    
    synapses <- list(i = rep(1:nb_neurons, each = length(input_neurons)),
                     j = rep(1:length(input_neurons), nb_neurons))
    
    if(is.null(L$activation_arguments)) L$activation_arguments = list()
    new("NNLayer",
        layer_id = layer_id,
        neurons = neurons,
        input_neurons = input_neurons,
        activation_function = L$activation_function,
        activation_arguments = L$activation_arguments,
        synapses = synapses)
}

NNLayer.set_synaptic_weights <- function(NN,
                                         W)
{
    if(length(NN@synapses$i) != length(W)) stop("Dimension of weights and synapses don't match")
    NN@synapses$weight = W
    NN
}

NNLayer.get_matrix_representation <- function(NNL)
{
    sparseMatrix(i = NNL@synapses$i,
                 j = NNL@synapses$j,
                 x = NNL@synapses$weight,
                 dims = c(length(NNL@neurons), length(NNL@input_neurons)))
}

NNLayer.activate <- function(NNL,
                             signal)
{
    do.call(NNL@activation_function, c(list(signal), NNL@activation_arguments))
}

NNLayer.get_num_synapses <- function(NN)
{
    return(length(NN@synapses$i))
}
   
setGeneric("set_synaptic_weights", function(NN, W) standardGeneric("set_synaptic_weights"))
setMethod("set_synaptic_weights", c("NNLayer", "numeric"), NNLayer.set_synaptic_weights)

setGeneric("get_matrix_representation", function(NNL) standardGeneric("get_matrix_representation"))
setMethod("get_matrix_representation", c("NNLayer"), NNLayer.get_matrix_representation)
           
setGeneric("activate", function(NNL, ...) standardGeneric("activate"))
setMethod("activate", c("NNLayer"), NNLayer.activate)

setGeneric("get_num_synapses", function(NN, ...) standardGeneric("get_num_synapses"))
setMethod("get_num_synapses", c("NNLayer"), NNLayer.get_num_synapses)

setClass("JalgosNN",
         slots = c(neurons = "index", ## Each neuron is ided by an integer
         input_size = "index",
         layers = "list"))

JalgosNN <- function(L)
{
    input_size = L$input_size
    layers = list()
    last_neuron = input_size
    input_neurons = 1:input_size
    neurons = input_neurons
    for(layer_def in L$layers)
    {
        layer <- NNLayer(layer_def,
                         layer_id = as.character(length(layers) + 1),
                         input_neurons = input_neurons,
                         last_neuron = last_neuron)
        neurons = c(neurons, layer@neurons)
        input_neurons = layer@neurons
        last_neuron = max(neurons)
        layers = c(layers, layer)
    }
    new("JalgosNN",
        neurons = neurons,
        input_size = input_size,
        layers = layers)
}

JalgosNN.get_num_synapses <- function(NN)
{
    sum(sapply(NN@layers, get_num_synapses))
}

JalgosNN.set_synaptic_weights <- function(NN,
                                          W)
{
    nsyn = 0
    new_layers = list()
    for(layer in NN@layers)
    {
        nbs = get_num_synapses(layer)
        layer = set_synaptic_weights(layer, W[nsyn + 1:nbs])        
        nsyn = nsyn + nbs
        new_layers = c(new_layers, layer)
    }
    NN@layers = new_layers
    NN
}

JalgosNN.process <- function(JNN,
                             input)
{
    layer_state = input
    for(layer in JNN@layers)
    {
        M = get_matrix_representation(layer)
        next_raw_layer_state = M %*% layer_state
        layer_state = activate(layer,  next_raw_layer_state)
    }
    layer_state
}

setMethod("get_num_synapses", c("JalgosNN"), JalgosNN.get_num_synapses)
setMethod("set_synaptic_weights", c("JalgosNN"), JalgosNN.set_synaptic_weights)


setGeneric("process", function(JNN, ...) standardGeneric("process"))
setMethod("process", "JalgosNN", JalgosNN.process)


### Test

## A 3 layers network with an input of size 1 and output of size 1
## Weights and samples are randomly drawn.
## Last call prints the input vs output.
NNExample <- function()
{
    layer1 = list(nb_neurons = 10, activation_function = tanh)
    layer2 = list(nb_neurons = 5, activation_function = tanh)
    layer3 = list(nb_neurons = 1, activation_function = tanh)
    JNNDef <- list(input_size = 1,
                   layers = list(layer1, layer2, layer3))
    
    JNN = JalgosNN(JNNDef)
    W = rnorm(get_num_synapses(JNN)) * 2
    JNN = set_synaptic_weights(JNN, W)
    input = Matrix(rnorm(1000), 1, 1000)
    output = process(JNN, input)
    print(qplot(as.vector(input), as.vector(output)), title = "input vs output")
    JNN
}
