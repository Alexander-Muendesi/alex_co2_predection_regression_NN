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
random_number_generator = random.Random(0)
data_reader = DataReader(random_number_generator=random_number_generator)
data_reader.readFile()

counter = 439
while counter < 1000:
    hidden_layers = [84]
    num_epochs = 100
    batch_size = 14
    learning_rate = 0.006887880859375

    random_number_generator = random.Random(counter)
    data_reader = DataReader(random_number_generator=random_number_generator)
    data_reader.readFile()


    neuralNet = NeuralNetwork(hidden_layers, data_reader,num_epochs,batch_size,learning_rate)
    neuralNet.set_np_seed(counter)

    result = str(counter) + ": " + str(neuralNet.train())
    print(result)
    counter += 1    



def parameter_tuning():
    while True:
        row = None
        try:
            row = sobolReader.getRow()
        except Exception:
            break

        batch_size = 14
        learning_rate = 0.006887880859375
        num_neurons_hidden_layer = int(2 + (1000-2) * row.iloc[0])
        params = str(batch_size) + "," + str(learning_rate) + "," + str(num_neurons_hidden_layer) + ","

        
        hidden_layers = [num_neurons_hidden_layer]
        num_epochs = 100

        neuralNet = NeuralNetwork(hidden_layers, data_reader,num_epochs,batch_size,learning_rate)
        params += str(neuralNet.train())
        print(params)


#NB the max number of neurons you are allowed to have is 1000. Since they are 2171 we want to keep it in the ratio 1:2 to try prevent overfitting
#max number of neurons per layer we will cap it at 100 for now..So 10 hidden layers max
# best parameter values: batch size: 14.0 learning rate: 0.006887880859375 Number of hidden layer neurons: 84
# Reason for using one hidden layer: universal approximation theorem says one hidden layer can approximate many functions that would typically require
# multiple hidden layers in the NN

#ranges for parameters
# batch size [5,150]
# learning rate [0.00001, 0.01]
# num neurons in hidden layer [2,1000] #1000 because we want to ensure ratio of 1:2 of weights to training instances

# Adagrad NN run performance
# Average: 0.08776255463808776
# Standard Deviation: 0.029636709596042164

# Adam NN run performance
# Average: 0.08641217925399541
# Standard Deviation: 0.028282631184210816

# ASGD NN run performance
# Average: 0.08530568008497358
# Standard Deviation: 0.028516834976621358

