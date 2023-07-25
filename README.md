1.0.0:

This version was written in january 2021 during winter Camp. However, it is lost due to accidental cleaning up of my files.

In this version, I only realized input layer, Dense layer, and output layer. There are no other special layers. The way of constructing a network is simply putting a list into the network, telling how many nodes in each Dense layer. I use sigmoid and MSE as default activation function and loss function. There are no optimizer policy.

1.0.1:

Updates
1. Construct Dense layer class, Input layer class. Omit output layer
2. Add ReLU function as activation function, softmax as output function(Though it's still problematic)
3. Add Dropout class


This version is written in the summer of 2022. In this version, I started to use structual way. 
Instead of constructing Dense layer class, input layer class, output layer in a single class, I choose to separate them into different classes. It is better to maintain and clearer for me to write codes.



1.0.2: 

Updates
1. Add LeaklyRelu and tanh activation Function. Add CrossEntropy+Softmax loss function(it can no function properly). Add SGD and Momentum optimizers.
2. Decompose Dense class into Linear, Activate Function, and Loss Function classes. Activate Function is now a layer instead of a part of Linear Layer.
3. Each layer(except sigmoid and tanh layer) will store the input, not its output.
4. Solve the problem of calculating large gradients. Now can use larger learning rate
5. Linear layer will now take both input_size and output_size in order to intialize weights more easily.



I decided to intimitate the structual Neural network in Pytorch. 

Although I don't know the code behind pytorch, it seems that it treats activate function as a layer, the Dense layer as an activation function (Linear activation function). In this case, the training of Neural network is all about automatic differentiation. Every layer is a type of function. The combination of these function can theoretically imitate any function. 

I believe such structure can be easily maintained and really flexible. So I separate the Dense layer into different classes. 

I still don't know how pytorch use optimizer to update parameters. Optimizer in Pytorch takes in all the parameters of Neural network and updates these parameters. I don't know how to realise it. In my project, every Linear layer will be coupled with optimizer object and the layer will update its parameter through the optimizer(as far as I know, optimizer is used to update the weights in Linear layer)


