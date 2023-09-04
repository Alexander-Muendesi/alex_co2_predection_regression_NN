from dataReader import DataReader
from neuralNetwork import NeuralNetwork
import random

random_generator = random.Random(0)
datareader = DataReader(random_generator)
datareader.readFile()
hidden_layers = [10,10,10]
learning_rate = 0.0001
num_epochs = 100
batch_size = 10
neuralNet = NeuralNetwork(hidden_layers, datareader,num_epochs,batch_size,learning_rate)
neuralNet.train()