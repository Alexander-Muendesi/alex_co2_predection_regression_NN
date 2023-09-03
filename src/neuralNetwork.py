import numpy as np
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self,hidden_layer_sizes,data_reader):
        super(NeuralNetwork, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes            # keeps the number of neurons for each hidden layer
        self.data_reader = data_reader
        self.num_inputs = data_reader.get_num_inputs()

        train_data = data_reader.get_train_data().copy()
        train_data_target = train_data["Value_co2_emissions_kt_by_country"]
        del train_data["Value_co2_emissions_kt_by_country"]

        validation_data = data_reader.get_validation_data().copy()
        validation_data_target = validation_data["Value_co2_emissions_kt_by_country"]
        del validation_data["Value_co2_emissions_kt_by_country"]

        test_data = data_reader.get_test_data().copy()
        test_data_target = test_data["Value_co2_emissions_kt_by_country"]
        del test_data["Value_co2_emissions_kt_by_country"]

        prediction_data = data_reader.get_prediction_data().copy()
        prediction_data_target = prediction_data["Value_co2_emissions_kt_by_country"]
        del prediction_data["Value_co2_emissions_kt_by_country"]

        self.train_tensor_x = torch.tensor(train_data.values, dtype=torch.float32)              # x for the inputs
        self.train_tensor_y = torch.tensor(train_data_target.values, dtype=torch.float32).reshape(-1,1)       # y for the target value

        self.validation_tensor_x = torch.tensor(validation_data.values, dtype=torch.float32)
        self.validation_tensor_y = torch.tensor(validation_data_target.values, dtype=torch.float32).reshape(-1,1)

        self.test_tensor_x = torch.tensor(test_data.values, dtype=torch.float32)
        self.test_tensor_y = torch.tensor(test_data_target.values, dtype=torch.float32).reshape(-1,1)
        
        self.prediction_tensor_x = torch.tensor(prediction_data.values, dtype=torch.float32)
        self.prediction_tensor_y = torch.tensor(prediction_data_target.values, dtype=torch.float32).reshape(-1,1)

        layers = []

        layers.append(nn.Linear(self.num_inputs,self.hidden_layer_sizes[0])) #input layer with the identity function

        for i in range(len(hidden_layer_sizes) - 1): #create the hiddden layers
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_layer_sizes[-1], 1))   # 1 attribute is being predicted
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)
