from dataReader import DataReader
from neuralNetwork import NeuralNetwork
from sobolReader import SobolReader
import random

# random_generator = random.Random(0)
# datareader = DataReader(random_generator)
# datareader.readFile()
# hidden_layers = [10,10,10]
# # learning_rate = 0.0001
# learning_rate = 0.001
# num_epochs = 100
# batch_size = 10
# neuralNet = NeuralNetwork(hidden_layers, datareader,num_epochs,batch_size,learning_rate)
# neuralNet.train()

sobolReader = SobolReader()

while True:
    row = None
    try:
        row = sobolReader.getRow()
    except Exception:
        break

    batch_size = int(5 + (150 - 5) * row.iloc[0]) #range for batch size: [5, 150]
    learning_rate = 0.00001 + (0.01-0.00001) * row.iloc[1] #range for learning rate: [0.00001,0.01]
    params = str(batch_size) + "," + str(learning_rate) + ","

    random_number_generator = random.Random(2)
    data_reader = DataReader(random_number_generator=random_number_generator)
    data_reader.readFile()
    hidden_layers = [100]
    num_epochs = 100

    neuralNet = NeuralNetwork(hidden_layers, data_reader,num_epochs,batch_size,learning_rate)
    neuralNet.train()
    params += str(neuralNet.train())
    print(params)


#NB the max number of neurons you are allowed to have is 1000. Since they are 2171 we want to keep it in the ratio 1:2 to try prevent overfitting
#max number of neurons per layer we will cap it at 100 for now..So 10 hidden layers max