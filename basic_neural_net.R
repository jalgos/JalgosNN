## JalgosNN 
## A feed forward Neural Net is simply a sequence of matrix multiplication and activation function

## Using S4 class for representation
setClass("JalgosNN",
         slots = c(neurones = "integer",
         input_size = "integer",
         activation_functions = "list",
         layers = list(),
         network = "Matrix"))

JalgosNN <- function(nb_layers,
                     nb_neurons,
                     act_fun)
