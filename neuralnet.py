import pandas as pd
import numpy as np
import math

def sigmoid(x):
    y = 1/(1 + math.exp(-x))
    return y

def relu(x):
    return np.maximum(0, x)

class neuron:
    def __init__(self, inpWeights = 0, bias = 0):
        self.inpWeights = inpWeights
        self.bias = bias
        self.output = 0
        self.delta = 0

    def modWeights(self, weightCnt):
        self.inpWeights = np.random.rand(weightCnt)


        
class layer:
    def __init__(self, length, prevLength = 0, activation = sigmoid):
        self.length = length
        self.activation = activation
        self.neurons = []
        for i in range(length):
            self.neurons.append(neuron(np.random.rand(prevLength)))

class network:
    def __init__(self, hiddenLayers, layerLength, outputCnt):
        self.hiddenLayers = hiddenLayers
        self.layers = []
        self.layerLength = layerLength
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
        for i in range(self.firstLayer.length):
            self.firstLayer.neurons[i].modWeights(len(data))
        self.initialized = True
        print("Network intialized.")

    def fwdProp(self, dataPoint):
        layerIdx = 0
        for layer in self.layers:
            neuronIdx = 0
            for neuron in layer.neurons:
                if (layerIdx == 0):
                    weightIdx = 0
                    for inpWeight in neuron.inpWeights:
                        print(inpWeight)
                        neuron.output += inpWeight * dataPoint[weightIdx] + neuron.bias
                        weightIdx += 1
                    neuron.output = layer.activation(neuron.output)
                    print("Layer {}, Neuron {}; OUTPUT = {}".format(layerIdx, neuronIdx, neuron.output))
                else:
                    weightIdx = 0
                    for inpWeight in neuron.inpWeights:
                        print(inpWeight)
                        neuron.output += inpWeight * self.layers[layerIdx - 1].neurons[weightIdx].output + neuron.bias
                        weightIdx += 1
                    neuron.output = layer.activation(neuron.output)
                    print("Layer {}, Neuron {}; OUTPUT = {}".format(layerIdx, neuronIdx, neuron.output))
                    neuronIdx += 1
            layerIdx += 1

    def bckProp(self, targetVec, learnRate, dataPoint):
        def derivative(layerIdx, output):
            if self.layers[layerIdx].activation == sigmoid:
                return output * (1 - output)
            if self.layers[layerIdx].activation == relu:
                if output > 0:
                    return 1
                else:
                    return 0

        def updateDeltas():
            neuronIdx = 0
            for neuron in self.layers[-1].neurons:
                coefficient = targetVec[neuronIdx] - neuron.output
                neuron.delta = derivative(-1, neuron.output) * coefficient
                neuronIdx += 1
            layerIdx = self.hiddenLayers - 1
            while layerIdx >= 0:
                neuronIdx = 0
                for neuron in self.layers[layerIdx].neurons:
                    coefficient = 0
                    for prevNeuron in self.layers[layerIdx + 1].neurons:
                        coefficient += prevNeuron.delta * prevNeuron.inpWeights[neuronIdx]
                    neuron.delta = derivative(layerIdx, neuron.output) * (coefficient)
                    neuronIdx += 1
                layerIdx -= 1

        def updateWeights():
            layerIdx = self.hiddenLayers
            while layerIdx >= 1:
                neuronIdx = 0
                for neuron in self.layers[layerIdx].neurons:
                    weightIdx = 0
                    for weight in neuron.inpWeights:
                        weight += (learnRate * neuron.delta * self.layers[layerIdx - 1].neurons[weightIdx].output)
                        weightIdx += 1
                    neuronIdx += 1
                layerIdx -= 1
            for neuron in self.layers[layerIdx].neurons:
                weightIdx = 0
                for weight in neuron.inpWeights:
                    weight += (learnRate * neuron.delta * dataPoint[weightIdx])
                    weightIdx += 1
                neuronIdx += 1

        updateDeltas()
        updateWeights()

        # Datasets are passed as a parameter to the train/test function.
        # This allows for networks to be created independent of datasets
        # but demands for initialization of the number of weights on
        # first-layer neurons upon first introducing a dataset.

    def train(self, data, targets, learnRate):
        if (self.initialized == False):
            initData(data)
        if (len(data[0]) != len(self.firstLayer.neurons[0].inpWeights)):
            print("Data format incompatible: number of features do not match initial dataset.")
            return 0
        idx = 0
        for dataPoint in data:
            self.fwdProp(dataPoint)
            self.bckProp(targets[idx], learnRate, dataPoint)
            idx += 1

    def test(self, data):
        if (self.initialized == False):
            self.initData(data)
        if (len(data[0]) != len(self.firstLayer.neurons[0].inpWeights)):
            print("Data format incompatible: number of features do not match initial dataset.")
            return 0
        for dataPoint in data:
            self.fwdProp(dataPoint)

def main():
    data = [[1, 4, 3], [2, 4, 6], [7, 8, 2]]
    targets = [[1, 0], [0, 1], [1, 0]]
    firstNetwork = network(hiddenLayers = 2, layerLength = 3, outputCnt = 2)
    firstNetwork.test(data)
    firstNetwork.train(data, targets, learnRate = 0.1)

main()