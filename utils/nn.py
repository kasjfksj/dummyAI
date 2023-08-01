import numpy as np



class Linear:
    def __init__(self, input_size, output_size,optimizer,initialization="normal"):

        if(initialization == "normal"):
            self.weight = np.random.normal(loc = 0,scale=np.sqrt(2 / (input_size + output_size)),size=(input_size, output_size))
        else:
            self.weight = np.random.rand(input_size,output_size)
        self.input = None
        self.optimizer = optimizer

    def forward(self, input):
        self.input = input
        return np.dot(input,self.weight)

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

        
    def forward(self,input):
        if self.count%1000==0:
            self.arr=np.random.binomial(1, 1-self.rate, input.shape[-1])
        self.count=(self.count+1)%1001
        return input*self.arr
    def backward(self,error):
        return (error*self.arr)

class BatchNorm1d:
    def __init__(self,gamma=1, beta=0, momentum=0.01,epsilon = 1e-4, lr =1e-2):
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = gamma
        self.beta = beta
        self.lr = lr
        self.mu_data  = None
        self.var_data = None
        self.global_var = None
        self.global_mu = None
        self.out_put = None
        self.batch_size = 0
    def forward(self,input,training = True):
        if self.global_mu is None:
            self.global_mu = np.zeros(input.shape[1])
            self.global_var = np.zeros(input.shape[1])
        if training:
            self.mu_data = input - input.mean(axis=0)
            self.var_data = np.var(input,axis=1)
            self.batch_size = input.shape[0]
            self.global_mu = self.momentum*self.global_mu + (1-self.momentum)*self.mu_data
            self.global_var = self.momentum*self.global_var + (1-self.momentum)*self.var_data
            self.output =  self.mu_data/(np.sqrt(self.var_data)+self.epsilon)
        else:
            self.output = (input-self.global_mu)/(np.sqrt(self.var_data)+self.epsilon)
        return self.gamma*self.output+self.beta
    def backward(self,error):
        self.beta -= self.lr*error.sum(axis=0)
        self.gamma -= self.lr*np.sum(self.output*error,axis=0)
        self.output = self.gamma*error
        return


class Net:
    def __init__(self):
        self.sequence = list()
    def Sequential(self, *other):
        for i in other:
            self.sequence.append(i)
    def forward(self, data,training = True):
        if training:
            for i in self.sequence:
                data = i.forward(data)
        else:
            for i in self.sequence:
                if isinstance(i,Dropout):
                    continue
                data = i.forward(data)
        return data
    def backward(self,loss):
        for i in range(len(self.sequence)-1,-1,-1):
            loss = self.sequence[i].backward(loss)
    def valuate(self,test_data,labels):
        c=0
        for i in range(len(test_data)):
            outputs = self.forward(test_data[i],training = False)
            if np.argmax(outputs) == np.argmax(labels[i]):
                c += 1
        return str(c/len(labels)*100)+"%"
        
