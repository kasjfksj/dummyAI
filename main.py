import numpy as np

from keras.datasets import mnist
from nn import Net, Linear
from activateFunction import Tanh,Leakly_ReLU,ReLU
from lossFunction import MSE, CrossEntropyLoss

import requests
import ssl
requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

batch_size = 3000

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train1=x_train
x_test1=x_test
x_train = x_train.reshape((-1, 28 * 28)).astype("float32") / 255.0-0.5
x_train = np.array_split(x_train,batch_size)


y_train = np.eye(10)[y_train.reshape(-1)]
y_train = np.array_split(y_train,batch_size)

x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255.0-0.5
y_test = np.eye(10)[y_test.reshape(-1)]



if __name__=="__main__":

    net =Net()
    net.Sequential(
        Linear(784,256,optimizer="Momentum"),
        Leakly_ReLU(),
        Linear(256,128,optimizer="Momentum"),
        Leakly_ReLU(),
        Linear(128,10,optimizer="Momentum")
    )
    loss = CrossEntropyLoss()
    count=1
    for i in range(100):
        for j in range(100):
            if((i*100+j)%1000==0):
                print(count)
                count+=1
            outputs = net.forward(x_train[i])
            Loss = loss.forward(outputs,y_train[i])
            
            error = loss.backward(y_train[i])
            
            net.backward(error)
    
    c=0
    for i in range(len(x_test)):

        outputs = net.forward(x_test[i])
        if np.argmax(outputs)==np.argmax(y_test[i]):
            c+=1
    print(c)
 
        
