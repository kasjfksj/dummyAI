import numpy as np


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self,input):
        self.input = input
        return np.maximum(self.input,0)
    def backward(self, error):

        return np.array(np.maximum(self.input,0)>0,dtype = np.int32)*error

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self,input):
        """  Better use the output of sigmoid function to calculate gradient. If use
        the input to calculate gradient, have to calculate exponential function
        twice, which costs a lot of time. If use the output to calculate gradient,
        only need to multiply. 
        """
        self.output = 1 / (1 + np.exp(-input))
        return self.output  
    def backward(self,error):
        return self.output*(1-self.output)*error

class Tanh:
    def __init__(self):
        self.output = None
    def forward(self,input):
        self.output = np.tanh(input)
        return self.output

    def backward(self,error):
        return (1-self.output*self.output)*error
    
class Leakly_ReLU:
    def __init__(self,alpha=0.5):
        self.alpha =alpha
        self.input = None

    def forward(self,input):
        self.input = input
        return np.maximum(self.input,self.input*self.alpha)
    def backward(self, error):
        a = np.array(np.maximum(self.input,0)>0,dtype = np.int32)+np.array(np.maximum(self.input,self.input*0.5)<0,dtype = np.int32)*0.5
        return a*error



