import numpy as np
from Optimizer import Momentum,SGD


class Linear:
    def __init__(self, input_size, output_size,optimizer,initialization="normal"):
        self.size = output_size
        if(initialization == "normal"):
            self.weight = np.random.normal(loc = 0,scale=np.sqrt(2 / (input_size + self.size)),size=(input_size, output_size))
        else:
            self.weight = np.random.rand(input_size,output_size)
        self.input = None
        if type(optimizer).__name__=="Momentum":
            self.optimizer = Momentum()
        else:
            self.optimizer = SGD()

        
        
    def forward(self, input):
        self.input = input
        return np.dot(input,self.weight)/input.shape[0]

    def backward(self, error):
        tmp = self.weight
        grad=np.dot(self.input.T, error)/self.input.shape[0]

        self.weight = self.optimizer.update(self.weight,grad)
        return np.dot(tmp, error.T).T

class Dropout:
    def __init__(self, dropout_rate):
        self.rate=dropout_rate
        self.arr=None
        self.count=0
    def set_Optimizer(self,optimizer):
            pass
        
    def forward(self,input):
        if self.count%100==0:
            self.arr=np.random.binomial(1, 1-self.rate, (input.shape[1]))
        self.count=(self.count+1)%101
        return input*self.arr
    def backward(self,error):
        return (error*self.arr)

class normalization:
    def __init__(self):
        self.epsilon = 1e-4
        self.mu_data  = None
        self.var_data = None
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
    def forward(self,input):
        self.mu_data = input-np.sum(input,axis=1)/input.shape[0]
        self.var_data = np.var(input,axis=1)
        return self.mu_data/np.sqrt(self.var_data+self.epsilon)
    def backward(self,prev_record,error):
        return 


class Net:
    def __init__(self):
        self.sequence = list()
    def Sequential(self, *other):
        for i in other:

            self.sequence.append(i)
    def forward(self, data):
        for i in self.sequence:
            data = i.forward(data)
        return data
    def backward(self,loss):
        for i in range(len(self.sequence)-1,-1,-1):
            loss = self.sequence[i].backward(loss)
        
