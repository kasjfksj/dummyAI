import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        self.outputs = None
    
    def forward(self,outputs, labels):
        '''
        softmax function
        '''
        maxn = np.max(outputs,axis=len(outputs.shape)-1)
        maxn = np.reshape(maxn.shape[0],1)
        outputs = outputs-maxn
        exp = np.exp(outputs)
        total = np.sum(exp, len(outputs.shape)-1)    
        for i in range(outputs.shape[0]):
            outputs[i] = exp[i]/total[i]

        '''
        CrossEntropy Function
        '''
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