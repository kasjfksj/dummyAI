import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from nn import Input, Dropout, Dense, ANN
import requests

import ssl

requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



if __name__=="__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train1=x_train
    x_test1=x_test
    x_train = x_train.reshape((-1, 28 * 28)).astype("float32") / 255.0-0.5
    x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255.0-0.5
    y_train = np.eye(10)[y_train.reshape(-1)]
    sample_size=40
    Neural = ANN(sample_size,epochs=7000)
    Neural.add(Input(28 * 28))
    Neural.add(Dropout(0.2))
    Neural.add(Dense(128, "relu",0.0013))
    Neural.add(Dropout(0.2))
    Neural.add(Dense(10, "relu",0.0008))

    print(Neural.train(x_train,y_train))
    # print(Neural.loss)
    # print(Neural.sequence[-1].weight)
    plt.plot(Neural.loss)
    # print(Neural.sequence[-1].weight)
    print("final loss:",Neural.loss[-1])
    plt.show()


    c=0
    tmp=0

    for b in range(10000):
        a=Neural.predict(x_train[b:b+1])
        maxn=0
        for i in range(len(a[0])):
            if a[0][maxn]<a[0][i]:
                maxn=i
    #     print(a,y_test[b])
    #     print(y_train.shape)
        for j in range(len(y_train[b])):
            if y_train[b][j]==1:
                tmp=j
                break
        if maxn==tmp:
            c+=1

    print(c/10000)
