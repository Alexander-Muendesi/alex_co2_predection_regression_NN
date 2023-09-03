from dataReader import DataReader
from neuralNetwork import NeuralNetwork
import random

random_generator = random.Random(0)
datareader = DataReader(random_generator)
datareader.readFile()

neuralNet = NeuralNetwork(None, datareader)