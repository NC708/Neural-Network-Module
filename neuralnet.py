import pandas as pd
import numpy as np
import math

def sigmoid(x):
    y = 1/(1 + math.exp(-x))
    return y

def relu(x):
    if (x < 0):
        y = 0
    else:
        y = x
    return y

class neuron:
    def __init__(self, inpWeights = 0, bias = 0):
        self.inpWeights = inpWeights
        self.bias = bias
        self.output = 0

    def modWeights(self, weightCnt):
        self.inpWeights = np.random.rand(weightCnt)
        
class layer:
    def __init__(self, length, prevLength = 0):
        self.length = length
        self.neurons = []
        for i in range(length):
            self.neurons.append(neuron(np.random.rand(prevLength)))

class network:
    def __init__(self, hiddenLayers, layerLength, outputCnt):
        self.layers = []
        for i in range(hiddenLayers):
            if (i == 0):
                self.layers.append(layer(layerLength))
            else:
                self.layers.append(layer(layerLength, layerLength))
        self.layers.append(layer(outputCnt, layerLength))
        self.initialized = False
        self.firstLayer = self.layers[0]
        print(self.layers)

    def initData(self, data):
        self.initialized = True
        for i in range(self.firstLayer.length):
            self.firstLayer.neurons[i].modWeights(len(data))

    def fwdProp(self, dataPoint):
        for layerIdx, layer in enumerate(self.layers):
            for neuronIdx, neuron in enumerate(layer.neurons):
                if (layerIdx == 0):
                    for weightIdx, inpWeight in enumerate(neuron.inpWeights):
                        print(inpWeight)
                        neuron.output += inpWeight * dataPoint[weightIdx] + neuron.bias
                    neuron.output = sigmoid(neuron.output)
                    print("Layer {}, Neuron {}; OUTPUT = {}".format(layerIdx, neuronIdx, neuron.output))

                else:
                    for weightIdx, inpWeight in enumerate(neuron.inpWeights):
                        print(inpWeight)
                        neuron.output += inpWeight * self.layers[layerIdx - 1].neurons[weightIdx].output + neuron.bias
                    neuron.output = sigmoid(neuron.output)
                    print("Layer {}, Neuron {}; OUTPUT = {}".format(layerIdx, neuronIdx, neuron.output))

    def bckProp(self, targetVec, learnRate):
        print("temp")
        # Datasets are passed as a parameter to the train/test function.
        # This allows for networks to be created independent of datasets
        # but demands for initialization of the number of weights on
        # first-layer neurons upon first introducing a dataset.
    def train(self, data, targets, learnRate):
        if (self.initialized == False):
            initData(data)
        if (len(data[0]) != self.firstLayer.neurons[0].inpWeights):
            print("Data format incompatible: number of inputs do not match.")
            return 0
        for idx, dataPoint in enumerate(data):
            self.fwdProp(dataPoint)
            self.bckProp(targets[idx], learnRate)
    def test(self, data):
        if (self.initialized == False):
            self.initData(data)
        if (len(data[0]) != len(self.firstLayer.neurons[0].inpWeights)):
            print("Data format incompatible: number of features do not match initial dataset.")
            return 0
        for dataPoint in data:
            self.fwdProp(dataPoint)

def main():
    data = list(np.random.rand(3, 3))
    firstNetwork = network(2, 3, 2)
    print(data[2])
    firstNetwork.test(data)
    data = list(np.random.rand(4, 3))
    firstNetwork.test(data)

        
main()
