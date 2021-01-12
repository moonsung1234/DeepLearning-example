
import numpy as np

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

class Network :
    def __init__(self, input_nodes, middle_nodes, output_nodes, learning_rate, delta) :
        self.input_nodes = input_nodes
        self.middle_nodes = middle_nodes
        self.output_nodes = output_nodes

        self.w1_data = np.random.rand(input_nodes, middle_nodes)
        self.w2_data = np.random.rand(middle_nodes, output_nodes)

        self.b1_data = np.random.rand(middle_nodes)
        self.b2_data = np.random.rand(output_nodes)

        self.A1 = np.zeros([1, input_nodes])
        self.A2 = np.zeros([1, middle_nodes])
        self.A3 = np.zeros([1, output_nodes])

        self.Z1 = np.zeros([1, input_nodes])
        self.Z2 = np.zeros([1, middle_nodes])
        self.Z3 = np.zeros([1, output_nodes])

        self.learning_rate = learning_rate
        self.delta = delta

    def getLoss(self) :
        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.w1_data) + self.b1_data
        self.A2 = sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.w2_data) + self.b2_data
        self.A3 = sigmoid(self.Z3)

        return -np.sum(self.target_data * np.log(self.A3 + self.delta) + (1 - self.target_data) * np.log((1 - self.A3) + self.delta))

    def run(self, input_data, target_data) :
        self.input_data = input_data
        self.target_data = target_data

        default_loss = self.getLoss()

        loss2 = (self.A3 - self.target_data) * self.A3 * (1 - self.A3)
        
        self.w2_data -= self.learning_rate * np.dot(self.A2.T, loss2)
        self.b2_data -= self.learning_rate * loss2.copy().reshape(self.output_nodes,)

        loss1 = np.dot(loss2, self.w2_data.T) * self.A2 * (1 - self.A2)
        
        self.w1_data -= self.learning_rate * np.dot(self.A1.T, loss1)
        self.b1_data -= self.learning_rate * loss1.copy().reshape(self.middle_nodes,)

    def predict(self, input_data) :
        #middle(hidden) node
        A2 = sigmoid(np.dot(input_data, self.w1_data) + self.b1_data)

        #output node
        A3 = sigmoid(np.dot(A2, self.w2_data) + self.b2_data)

        return A3

input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], ndmin=2)
target_data = np.array([0, 1, 1, 1], ndmin=2).reshape(4, 1)

N = Network(2, 6, 1, 1e-1, 1e-5)

for i in range(1000) :
    for ii in range(4) :
        N.run(input_data[ii].reshape(1, 2), target_data[ii].reshape(1, 1))

    if i % 100 :
        print("loss : ", N.getLoss())    

print("predict : ", N.predict(np.array([0, 1])))

