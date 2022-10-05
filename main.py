import numpy as np
import linearLayer as ll
import activationLayer as al
import matplotlib as plt
#INFO

#Best Trained model: 
#weights: [7.908250953632633, 7.85234582284727]
#bias: -3.4343647453407344
#accuracy: 99.6%
#loss (varies for different inputs): ~0.01 to 0.03

#Set weights using 

def main():
    #code to generate data:
    # data = createData(1000)
    # np.savetxt('data.csv', data)

    #code to manipulate data:
    data = np.loadtxt('data.csv')
    inputs = []
    actual = []

    #separate data into inputs array and y_actual array
    for j in range(len(data)):
        inputs.append([data[j][0], data[j][1]])
        actual.append(data[j][2])


    #create network layers, consisting of one sigmoid node, one linear, two input nodes, and a bias node
    layer1 = ll.LinearLayer(len(inputs[0]))
    layer2 = al.ActivationLayer()
    
    #write predetermined weights / bias down here:
    fixedWeights = [7.908250953632633, 7.85234582284727]
    fixedBias = -3.4343647453407344

    train = False
    if (train):
        history = []
        iterations = []
        epochs = 1
        learningRate = 0.7
        for epoch in range(epochs):
            #accuracy constants
            correct = 0.0
            total = 0.0

            #stochastic gradient descent: one data point is put thropugh the model at a time
            for i in range(len(inputs)):
                print("\ninput: " + str(inputs[i]))
                
                #call activate linear layer: calculate Z
                layer1.activate(inputs[i])
                #put Z through the activation function (sigmoid)
                layer2.activate(layer1.output)
                
                print("p_hat for true: " + str(layer2.output))

                #increment for accuracy calculations:
                total = total + 1
                if ((layer1.output > 0.5 and actual[i] == 1.0) or layer1.output < 0.5 and actual[i] == 0.0):
                    correct = correct + 1

                #matplotlib shenanigans
                iterations.append[total]

                #calculate derivative of loss with respect to p_hat to as first upstream gradient
                #note that this comes from the first redundant backpropogation step, where dL/dL = 1 (so it doesn't impact anything)
                dLdp_hat = lossDerivative(actual[i], layer2.output)
                loss = lossFunction([actual[i]], [layer2.output])
                
                #matplotlib shenanigans
                history.append(loss)
                
                print("loss: " + str(loss))

                #propogate back through the layers using the newly generated chain rule gradient as the upstream each time
                layer2.backpropogate(dLdp_hat)
                layer1.backpropogate(layer2.dp_hatdz)
                #calculate change in weights and biases using gradient descent
                layer1.gradientDescent(learningRate)


                print("iteration: " + str(i + 1))
                print("weights: " + str(layer1.weights))
                print("bias: " + str(layer1.bias) + "\n")

            accuracy = calculateAccuracy(total, correct)
            print("accuracy: " + str(accuracy))
    else:
        layer1.setParams(fixedWeights, fixedBias)

    #loss visualization for jupyter notebook: doesn't work in VSCode
    # plt.plot(iterations, history, 'r.-', label="Loss")
    # plt.title("Iterations VS Loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Iteration")
    # plt.show()


    #simple loop to test the model
    while True:
        input1 = float(input("Choose 1 or 0: "))
        input2 = float(input("Choose 1 or 0: "))
        #simply goes forward through the model, and never goes backwards to change the values
        layer1.activate([input1, input2])
        layer2.activate(layer1.output)
        print("probability of being true (p_hat): " + str(layer2.output))
        if (layer2.output > 0.5):
            print("Classified as true")
        else:   
            print("Classified as false")

#given by Vikram
def createData(size):
    data = []
    for i in range(size):
        temp1 = np.random.randint(0, 2)
        temp2 = np.random.randint(0, 2)
        test = temp1 or temp2
        data.append([temp1, temp2, test])
    return data

#actual: real double value result of an or gate
#pred: the models p_hat double value for the same or gate inputs
def lossFunction(actual, pred):
        bce = 0
        #for loop for potential batch use later down the road
        for i in range(len(actual)):
            #using max to prevent any divisions by 0, capping out at an arbitrarily small number
            #binary cross entropy formula from the internet
            bce = actual[i] * np.log(max(pred[i], 1e-07)) + (1 - actual[i]) * np.log(max(1 - pred[i], 1e-07))
        return bce / (- len(actual))

#actual: real double value result of an or gate
#pred: the models p_hat double value for the same or gate inputs
def lossDerivative(actual, pred):
        #note that the max functions are to prevent any diviosns by 0
        #binary cross entropy derivative formula from the internet
        return (- actual / max(pred, 1e-07)) + (1 - actual) / max(1 - pred, 1e-07)

#total: incrementor integer value
#correct: incrementor integer value
def calculateAccuracy(total, correct):
    return 1.0 * correct / total

if __name__ == '__main__':
    main()

