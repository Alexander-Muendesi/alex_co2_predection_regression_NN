from dataReader import DataReader
from neuralNetwork import NeuralNetwork
import random

random_generator = random.Random(0)
datareader = DataReader(random_generator)
datareader.readFile()
hidden_layers = [10,10,10]
neuralNet = NeuralNetwork(hidden_layers, datareader)