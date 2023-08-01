import numpy as np

class Momentum:
    def __init__(self,lr = 1e-2, momentum = 0.1):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self,parameters, grad):
        if self.v is None:
            self.v = np.zeros((grad.shape[0],grad.shape[1]))

        self.v = self.momentum*self.v-(1-self.momentum)*grad
        parameters = parameters-self.lr*self.v
        return parameters
 
class Adagrad:
    def __init__(self,lr = 1e-2):
        self.lr = lr
        self.epsilon = 1e-10
        self.v = None


    def update(self,parameters,grad):
        if self.v is None:
            self.v = np.zeros((grad.shape[0],grad.shape[1]))
        self.v = self.v+ grad*grad
        parameters = parameters-self.lr*(grad/(np.sqrt(self.v)+self.epsilon))
        return parameters

class Adam:
    def __init__(self, 
                 epsilon =1e-10,
                 lr = 1e-3,
                 gamma=0.9,
                 beta1 = 0.9,
                 beta2 = 0.999):
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma

        self.beta1 =beta1
        self.beta2 = beta2
        self.beta1t = self.beta1
        self.beta2t = self.beta2
        self.m = None
        self.v = None

    def update(self,parameters,grad):
        if self.m is None:
            self.m = np.zeros((grad.shape[0],grad.shape[1]))
        if self.v is None:
            self.v = np.zeros((grad.shape[0],grad.shape[1]))
        self.m = self.beta1*self.m+(1-self.beta1)*grad
        self.v = self.beta2*self.v+(1-self.beta2)*grad*grad
        
        parameters = parameters-self.lr*(self.m/(1-self.beta1t))/(np.sqrt(self.v/(1-self.beta2t))+self.epsilon)
        self.beta1t = self.beta1t*self.beta1
        self.beta2t = self.beta2t*self.beta2
        return parameters
        
    
class SGD:
    def __init__(self,lr = 1e-2,decay_rate=None):
        self.lr =  lr
        if decay_rate is None:
            self.decay_rate = 0
        else:
            self.decay_rate = decay_rate
    def update(self,parameters,grad):
        return (1-self.decay_rate)*parameters-self.lr*(grad)