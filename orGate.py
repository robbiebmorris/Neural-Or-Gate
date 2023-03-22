import math
import numpy as np

#my savior:
#https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c

class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        #default weight value
        self.weight = 0.5
    
    def setWeight(self, weight):
        self.weight = weight
    
    def getWeight(self):
        return self.weight

class Neuron:
    learningRate = 1.0 #not over 20

    def __init__(self, layer):
        self.arr = []
        self.error = 0.0
        self.gradient = 0.0
        self.p_hat = 0.0
        self.bias = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                link = Connection(neuron)
                self.arr.append(link)

    def addErr(self, err):
        self.error = self.error + err

    def setErr(self, err):
        self.error = err

    def activationFunction(self, x):
        return 1 / (1 + math.exp(- x)) #*1.0?

    def AFderivative(self, x):
        return x * (1.0 - x)

    def setP_hat(self, p_hat):
        self.p_hat = p_hat

    def getP_hat(self):
        return self.p_hat

    def getBias(self):
        return self.bias
    
    def setBias(self, bias):
        self.bias = bias

    def activate(self):
        sum = 0
        if (len(self.arr) == 0):
            return
        for link in self.arr:
            sum = sum + link.connectedNeuron.getP_hat() * link.weight
        sum = sum + self.bias
        self.p_hat = self.activationFunction(sum)

    def lossFunction(self, actual, pred):
        bce = 0
        for i in range(len(actual)):
            bce = actual[i] * np.log(pred[i]) + (1 - actual[i]) * np.log(1 - pred[i])
        return bce / (- len(actual))

    def lossDerivative(self, actual, pred):
        return (- actual / pred) + (1 - actual) /(1 - pred)

    #type 0 is loss function, type 1 is activation function
    def backpropogate(self, upstreamGradient, neuronType):
        #upstream gradient * localgradient (stochastic gradiant descent)
        if neuronType == 1:
            self.gradient = upstreamGradient * self.AFderivative(self.p_hat)
        elif neuronType == 0:
            self.gradient = upstreamGradient * self.lossDerivative(self.p_hat)
        

    def gradientDescent(self):
        for link in self.arr:
            link.weight = link.weight - Neuron.learningRate * self.gradient
        self.bias = self.bias - Neuron.learningRate * self.gradient


class Network:
    def __init__(self, dimensions):
        self.layers = []
        #height is the number of (neurons????) in one layer
        for height in dimensions:
            layer = []
            for i in range(height):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setP_hat(1)
            self.layers.append(layer)

    #https://www.analyticsvidhya.com/blog/2021/03/binary-cross-entropy-log-loss-for-binary-classification/

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setP_hat(inputs[i])

    #call activate function for each neuron
    def activate(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.activate()

    def backpropogate(self, target):

        print(self.layers)
        #starting from the output layer, propogate back through layers
        for layer in self.layers[::-1]:
            print(self.layers)
            if layer == self.layers[0]:
                layerType = 1
            else:
                layerType = 0
            for neuron in layer: 
                    neuron.backpropogate(1.0, layerType)
        


        # for i in range(len(target)):
        #     self.layers[-1][i].setErr(target[i] - self.layers[-1][i].getP_hat()) #error is target - predicted

    def getP_hat(self):
        p_hat = []
        for neuron in self.layers[-1]:
            p_hat.append(neuron.getP_hat())
        p_hat.pop()
        return p_hat

def createData(size):
    data = []
    for i in range(size):
        temp1 = np.random.randint(0, 2)
        temp2 = np.random.randint(0, 2)
        test = temp1 or temp2
        data.append([temp1, temp2, test])
    return data

def main():
    #code to generate data:
    #data = createData(1000)
    #np.savetxt('data.csv', data)

    #code to manipulate data:
    data = np.loadtxt('data.csv')
    inputs = []
    actual = []
    
    #separate data into inputs array and y_actual array
    for j in range(len(data)):
        inputs.append([data[j][0], data[j][1]])
        actual.append(data[j][2])

    orGate = Network([2, 1])
    targetLoss = 0.1

    #iterate over the data multiple times
    #use the weights generated from each iteraction in the next iteration

    #change weights and repeat until loss is less than target loss
    while True:
        loss = 0  
        pred = []
        for i in range(len(data)):
            orGate.setInput(inputs[i])
            orGate.activate()
            orGate.backpropogate([actual[i]])
            print(orGate.final())
            pred.append(orGate.final())
        loss = orGate.lossFunction(actual, pred)
        print("error: " + str(loss))
        print(orGate.layers[0][1].arr)
        
        if (loss < targetLoss):
            break

    #code to let someone test it
    while True:
        input1 = input("Choose 1 or 0: ")
        input2 = input("Choose 1 or 0: ")
        orGate.setInput([int(input1), int(input2)])
        orGate.activate()
        result = orGate.final()
        print(result)
        if (result[0] > 0.5):
            print("classified as true")
        else:
            print("classified as false")

#measure accuracy and loss

if __name__ == '__main__':
    main()
