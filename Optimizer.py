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
    def __init__(self) -> None:
        pass
    
class SGD:
    def __init__(self,lr = 1e-2):
        self.lr =  lr
    def update(self,parameters,grad):
        return parameters-self.lr*grad