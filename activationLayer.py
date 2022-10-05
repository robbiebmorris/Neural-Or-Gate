import math
import numpy as np

class ActivationLayer:
    def __init__(self):
        self.output = 0.0

    def activate(self, input):
        #call activation function on Z node
        self.output = self.activationFunction(input)

    def backpropogate(self, upstreamGradient):
        localGradient = self.AFderivative(self.output)
        #chain rule: product of the local and upstream gradient
        self.dp_hatdz = localGradient * upstreamGradient

    #helpers:

    #x: any number
    def activationFunction(self, x):
        #code implementation of sigmoid function
        return 1.0 / (1.0 + math.exp(- x))

    def AFderivative(self, x):
        #code implementation of sigmoid function's derivative
        return x * (1.0 - x)