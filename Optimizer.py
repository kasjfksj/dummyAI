import numpy as np

class Momentum:
    def __init__(self,lr = 1e-2, momentum = 0.06):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self,parameters, grad):
        if self.v is None:
            self.v = np.zeros((grad.shape[0],grad.shape[1]))

        self.v = self.momentum*self.v-(1-self.momentum)*self.lr*grad
        parameters += self.v
        return parameters
 
class Adam:
    def __init__(self, 
                 epsilon =1e-10,
                 lr = 1e-2,
                 gamma=0.9,
                 theta = 1e-8,
                 beta1 = 0.9,
                 beta2 = 0.999):
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.theta = theta
        self.beta1 =beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
    def update(self,parameters,grad):
        if self.m is None:
            self.m = np.zeros((grad.shape[0],grad.shape[1]))
        if self.v is None:
            self.v = np.zeros((grad.shape[0],grad.shape[1]))
        self.m = self.beta1*self.m+(1-self.beta1)*grad
        self.v = self.beta2*self.v+(1-self.beta2)*grad
        parameters-=self.lr*(self.m/(1-self.beta1))/(np.sqrt(self.v/(1-self.beta2))+self.epsilon)
        return parameters
        
    
class SGD:
    def __init__(self,lr = 1e-2):
        self.lr =  lr
    def update(self,parameters,grad):
        return parameters-self.lr*(grad)