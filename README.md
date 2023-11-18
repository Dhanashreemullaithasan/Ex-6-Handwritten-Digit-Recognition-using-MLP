# Skill Assisessment-Handwritten Digit Recognition using MLP
## Aim:
To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
/*Describe here*/


## Algorithm :
1.Import the necessary libraries of python.

2.After that, create a dataframe and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3.Divide the dataset into two parts. Where the first part is for training and the second is for testing.

4.Define all the basic functions needed to create an MLP.

5.Find the weights and bias of each neuon using the gradient descent algorithm.

6.Make predictions using the defined functions.

7.Create a function to test the predictions which also contains the algorithm to plot the image.

8.Test the predictions and find the accuracy.

## Program:
```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) ## shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
Y_train


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
```

## Output :

### Y_train:
![image](https://github.com/Dhanashreemullaithasan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94165415/68e3b877-f3e1-454a-b3bc-30a1ef77d401)

### ITERATIONS FROM 1-100:

![image](https://github.com/Dhanashreemullaithasan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94165415/7ded565c-0f42-4f2c-95c1-a8b5ae5520d0)

### ITERATIONS FROM 240-340:

![image](https://github.com/Dhanashreemullaithasan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94165415/293c65a0-fbe6-4588-b88f-4014fba84fe0)

### ITERATIONS FROM 350-450 :

![image](https://github.com/Dhanashreemullaithasan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94165415/04614e41-e1a4-45c7-bf1f-bdd9bf73bf17)

### ITERATIONS FROM 460-490 :

![image](https://github.com/Dhanashreemullaithasan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94165415/4509de4a-e4dd-4855-bac2-031c6bc84c58)

### PREDICTING 5:

![image](https://github.com/Dhanashreemullaithasan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94165415/1898d928-7197-4873-84f6-7291a5989dde)

### PREDICTING 2 :

![image](https://github.com/Dhanashreemullaithasan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94165415/bd85cc48-8ae8-432b-b092-2b8c14ddc43c)

### DEV_PREDICTIONS :

![image](https://github.com/Dhanashreemullaithasan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94165415/d1bb83e2-29af-4cee-8706-fce6cfb052f4)

### ACCURACY :

![image](https://github.com/Dhanashreemullaithasan/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94165415/4f149184-7074-45c4-be02-f682a37fda7e)

## Result:

Thus, the program to recognize the handwritten digits using the multi-layer perceptron (MLP) is developed and executted successfully.
