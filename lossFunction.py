import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        self.outputs = None
    
    def forward(self,outputs, labels):
        # for i in range(outputs.shape[0]):
        #     outputs[i] -= np.max(outputs[i], axis=1)
        maxn = np.max(outputs,axis=1)
        maxn = np.reshape(maxn.shape[0],1)
        outputs = outputs-maxn
        exp = np.exp(outputs)
        total = np.sum(exp, axis=1)
        
            
        for i in range(outputs.shape[0]):
            if(total[i]<1e-9):
                print(outputs)
                print(total)
            outputs[i] = exp[i]/total[i]
        self.outputs = outputs
        return -np.log(outputs[i])*labels
    def backward(self,labels):
        return self.outputs-labels

class MSE:
    def __init__(self):
        self.outputs = None
    def forward(self,outputs, labels):
        self.outputs = outputs
        return (outputs-labels)*(outputs-labels)/2
    
    def backward(self,labels):
        return self.outputs-labels