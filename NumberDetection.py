import numpy as np

from keras.datasets import mnist
from utils.Optimizer import Adam,Adagrad,SGD
from utils.nn import Net, Linear, Dropout
from utils.activateFunction import Swish,ReLU
from utils.lossFunction import MSE, CrossEntropyLoss

import requests
import ssl
requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

batch_size = 5000


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train1=x_train
x_test1=x_test
x_train = x_train.reshape((-1, 28 * 28)).astype("float32") / 255.0-0.5
x_train = np.array_split(x_train,batch_size)


y_train = np.eye(10)[y_train.reshape(-1)]
y_train = np.array_split(y_train,batch_size)

x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255.0-0.5
y_test = np.eye(10)[y_test.reshape(-1)]


Epoch = 3
if __name__=="__main__":

    net =Net()
    net.Sequential(
        Linear(784,128,optimizer=Adam()),
        ReLU(),
        Dropout(0.1),
        Linear(128,64,optimizer=Adam()),
        ReLU(),
        Linear(64,10,optimizer=Adam())
    )
    loss = CrossEntropyLoss()
    count = 1
    record_loss = 
    for j in range(Epoch):
        for i in range(len(x_train)):
            
            outputs = net.forward(x_train[i])
            Loss = loss.forward(outputs,y_train[i])
            if (j*len(x_train)+i)%1000 == 0: 
                print(count,": ",np.sum(Loss))
                count += 1
            error = loss.backward(y_train[i])
            
            net.backward(error)

    acc = net.valuate(x_test,y_test)
    print(acc)
 
        
