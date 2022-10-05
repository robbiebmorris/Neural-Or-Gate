import math
import numpy as np


class LinearLayer:
    #datashape: array of length 2 hold width and height of dataset respectively
    #numNurons: number of neurons in the layer
    def __init__(self, dataShape):
        
        #initialize weights as array with input shape 0 values stored
        self.weights = [0.0] * dataShape
        self.bias = 0.0
        #note that output here is conventionally called Z: w1x1 + w2x2 + ... + b
        self.output = 0.0

    #x: an array of input data points
    def activate(self, x):
        #store input data for use later
        self.x = x
        #calculate Z: w1x1 + w2x2 + ... + b
        self.output = 0
        for i in range(len(self.weights)):
            self.output = self.output + self.weights[i] * self.x[i]
        self.output = self.output + self.bias

    #upstreamGradient: 
    def backpropogate(self, upstreamGradient):
        
        #derivative with respect to weights and bias:
        #z = x1w1 + x2w2 + b, dz/dw1 = x1, dz/dw2 = x2, dz/db = 1
        self.dLdw = []
        for i in range(len(self.weights)):
            dzdw = self.x[i]
            localGradient = dzdw
            self.dLdw.append(localGradient * upstreamGradient)

        #redundant, but here for understandings sake
        dzdb = 1 
        self.dLdb = dzdb * upstreamGradient

    #learningRate: arbitrary constant also known as alpha
    def gradientDescent(self, learningRate):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learningRate * self.dLdw[i]
        self.bias = self.bias - learningRate * self.dLdb

    def setParams(self, weights, bias):
        self.weights = weights
        self.bias = bias