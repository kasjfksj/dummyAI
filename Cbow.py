import re
import numpy as np
from utils.nn import Net,Linear
from utils.activateFunction import ReLU
from utils.Optimizer import SGD
from utils.lossFunction import CrossEntropyLoss
f = open("example.txt","r")
lines = f.read()

class TextProcessor:
    def __init__(self):
        self.data = None
        self.dictionary  = dict()
    def processing(self, lines):
        lines = lines.replace(".","*")
        lines = lines.replace("?","*")
        lines = lines.replace("!","*")
        lines = re.sub("[\s+\.\!\/_,$%(+\"\'""<>]+|[+-?,.~&#:;']"," ",lines)
        self.data = lines.split("*")
# print(lines[:500])

        for i in range(len(self.data)):
            self.data[i] = self.data[i].strip()
            self.data[i] = self.data[i].split(" ")
        for i in range(len(self.data)):
            for j in self.data[i]:
                if j not in self.dictionary:
                    self.dictionary[j] = len(self.dictionary)

process = TextProcessor()
process.processing(lines)

net = Net()
net.Sequential(
    Linear(190,20,optimizer=SGD()),
    ReLU(),
    Linear(20,190,optimizer=SGD())
)
loss = CrossEntropyLoss()
n = 3
class Cbow:
    def __init__(self):
        pass
# for k in range(20):
#     for i in range(n+1,len(lines)-n-2):
#         target = np.zeros(190)
#         target[dictionary[lines[i]]]=1
#         for j in range(i-n,i):
#             tmp = list()
#             tmp.append(np.zeros(190))

#             tmp[0][dictionary[lines[j]]]=1
            
#             tmp = np.array(tmp,dtype = np.float32)
#             outputs = net.forward(tmp)
#             Loss = loss.forward(outputs,target)

#             error = loss.backward(target)
#             net.backward(error)
#         for j in range(i+1,i+n+1):
#             tmp = list()
#             tmp.append(np.zeros(190))
#             tmp[0][dictionary[lines[j]]]=1
#             tmp = np.array(tmp,dtype = np.float32)
#             outputs = net.forward(tmp)

#             Loss = loss.forward(outputs,target)
#             error = loss.backward(target)
#             net.backward(error)

# tmp = list()
# tmp.append(np.zeros(190))

# tmp[0][dictionary[lines[10]]]=1

# tmp = np.array(tmp,dtype = np.float32)
# outputs = net.forward(tmp)   



    



