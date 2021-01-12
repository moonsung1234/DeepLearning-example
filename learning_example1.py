
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_data = np.array([0, 1, 1, 0])

w_data = np.random.rand(2, 6)
b_data = np.random.rand(6)

w1_data = np.random.rand(6, 1)
b1_data = np.random.rand(1)

delta = 1e-5
learning_rate = 1e-2

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def diff(f, x):
    final_x = np.zeros_like(x)
    
    iter = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    
    while not iter.finished:
        index = iter.multi_index        
        temp_x_value = x[index]

        x[index] = temp_x_value + delta
        fx1 = f(x)

        x[index] = temp_x_value
        fx2 = f(x)

        final_x[index] = (fx1 - fx2) / delta
         
        iter.iternext()   
        
    return final_x

def rid() :
    z = np.dot(x_data, w_data) + b_data
    a = sigmoid(z)

    #print(a, "\n")

    z1 = np.dot(a, w1_data) + b1_data
    a1 = sigmoid(z1)

    #print(a1, "\n")

    y = a1

    return -np.sum(t_data * np.log(y + delta) + (1 - t_data) * np.log((1 - y) + delta))

def predict(x_data):    
    z = np.dot(x_data, w_data) + b_data
    a = sigmoid(z)
    
    z1 = np.dot(a, w1_data) + b1_data
    a1 = sigmoid(z1)

    y = a1

    if y > 0.5:
        result = 1
    
    else :
        result = 0

    return y, result

def run() :
    global w_data, b_data, w1_data, b1_data

    function = lambda x : rid()

    for i in range(10001) :
        print(diff(function, w1_data))
        print(diff(function, b1_data))

        w_data -= learning_rate * diff(function, w_data)
        b_data -= learning_rate * diff(function, b_data)
        w1_data -= learning_rate * diff(function, w1_data)
        b1_data -= learning_rate * diff(function, b1_data)

        if i % 400 == 0 :
            print("loss(cost) : ", rid())

run()