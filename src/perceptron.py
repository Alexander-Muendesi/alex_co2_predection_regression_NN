import torch
import torch.nn as nn
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import pandas as pd



class Perceptron(nn.Module):
    def __init__(self,data_reader, num_epochs, batch_size, learning_rate):
        super(Perceptron, self).__init__()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_inputs = data_reader.get_num_inputs()
        self.data_reader = data_reader
        np.random.seed(29)

        train_data = data_reader.get_train_data().copy()
        train_data_target = train_data["Value_co2_emissions_kt_by_country"]
        del train_data["Value_co2_emissions_kt_by_country"]

        validation_data = data_reader.get_validation_data().copy()
        validation_data_target = validation_data["Value_co2_emissions_kt_by_country"]
        del validation_data["Value_co2_emissions_kt_by_country"]

        prediction_data = data_reader.get_prediction_data().copy()
        prediction_data_target = prediction_data["Value_co2_emissions_kt_by_country"]
        del prediction_data["Value_co2_emissions_kt_by_country"]

        self.train_tensor_x = torch.tensor(train_data.values, dtype=torch.float32)              # x for the inputs
        self.train_tensor_y = torch.tensor(train_data_target.values, dtype=torch.float32).reshape(-1,1)       # y for the target value

        self.validation_tensor_x = torch.tensor(validation_data.values, dtype=torch.float32)
        self.validation_tensor_y = torch.tensor(validation_data_target.values, dtype=torch.float32).reshape(-1,1)

        self.prediction_tensor_x = torch.tensor(prediction_data.values, dtype=torch.float32)
        self.prediction_tensor_y = torch.tensor(prediction_data_target.values, dtype=torch.float32).reshape(-1,1)

        #south africa data elements
        # lattitude = (float(-30.559482) - self.data_reader.train_mean_values["Latitude"]) / self.data_reader.train_std_dev_values["Latitude"]
        # longitude = (float(22.937506) - self.data_reader.train_mean_values["Longitude"]) / self.data_reader.train_std_dev_values["Longitude"]

        #sri lanka
        # lattitude = (float(7.873054) - self.data_reader.train_mean_values["Latitude"]) / self.data_reader.train_std_dev_values["Latitude"]
        # longitude = (float(80.771797) - self.data_reader.train_mean_values["Longitude"]) / self.data_reader.train_std_dev_values["Longitude"]

        #thailand
        lattitude = (float(15.870032) - self.data_reader.train_mean_values["Latitude"]) / self.data_reader.train_std_dev_values["Latitude"]
        longitude = (float(100.992541) - self.data_reader.train_mean_values["Longitude"]) / self.data_reader.train_std_dev_values["Longitude"]

        raw_data = pd.concat([data_reader.get_train_data(),data_reader.get_validation_data()],ignore_index=True)
        raw_data = raw_data.copy()
        south_africa_train_target_data = raw_data["Value_co2_emissions_kt_by_country"]
        del raw_data["Value_co2_emissions_kt_by_country"]

        south_africa_mask = (raw_data["Latitude"] == lattitude) & (raw_data["Longitude"] == longitude)
        south_africa_train_data = raw_data[south_africa_mask]
        south_africa_train_target_data = south_africa_train_target_data[south_africa_mask]

        self.south_africa_train_tensor_x = torch.tensor(south_africa_train_data.values, dtype=torch.float32)
        self.south_africa_train_tensor_y = torch.tensor(south_africa_train_target_data.values, dtype=torch.float32)

        self.model = nn.Sequential(
            nn.Linear(self.num_inputs,1)
        )

    def forward(self, x):
        return self.model(x)
    
    def set_np_seed(self, seed):
        np.random.seed(seed)

    def train(self):
        loss_function = nn.MSELoss()
        optimizer = optim.ASGD(self.model.parameters(),lr=self.learning_rate)
        # optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
        # optimizer = optim.Adagrad(self.model.parameters(),lr=self.learning_rate)

        batch_start = torch.arange(0, len(self.train_tensor_x), self.batch_size)

        best_mse = np.inf
        best_train_mse = np.inf
        best_weights = None
        history = []

        for epoch in range(self.num_epochs):
            self.model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    #select a batch
                    train_batch_x = self.train_tensor_x[start:start+self.batch_size]
                    train_batch_y = self.train_tensor_y[start:start+self.batch_size]
                    #perform a forward pass
                    train_prediction_y = self.forward(train_batch_x)
                    loss = loss_function(train_prediction_y, train_batch_y)
                    # perform the backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update the weights
                    optimizer.step()
                    # show the progress
                    bar.set_postfix(mse=float(loss))
            # evaluate the training error
            train_mse = self.evaluate_model(self.train_tensor_x,self.train_tensor_y, loss_function)
            if train_mse < best_train_mse:
                best_train_mse = train_mse
            
            #evaluate the accuracy after each epoch on the validation set
            mse = self.evaluate_model(self.validation_tensor_x, self.validation_tensor_y, loss_function)
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(self.model.state_dict())

            #shuffle the dataset
            shuffled_indices = np.random.permutation(len(self.train_tensor_x))
            self.train_tensor_x = self.train_tensor_x[shuffled_indices]
            self.train_tensor_y = self.train_tensor_y[shuffled_indices]

        self.model.load_state_dict(best_weights)
        print(best_train_mse)
        # self.test()
        return best_mse
    
    def test(self):
        with torch.no_grad():
            y_prediction = self.model(self.south_africa_train_tensor_x)
            for i in range(len(self.south_africa_train_tensor_x)):
                print(y_prediction[i].item() * self.data_reader.train_std_dev_values["Value_co2_emissions_kt_by_country"] + self.data_reader.train_mean_values[["Value_co2_emissions_kt_by_country"]])

            # for i in range(len(self.prediction_tensor_x)):
            #     print(y_prediction[i].item() * self.data_reader.train_std_dev_values["Value_co2_emissions_kt_by_country"] + self.data_reader.train_mean_values[["Value_co2_emissions_kt_by_country"]])
    
    # will be used to evalulate the validation and test sets
    def evaluate_model(self, x,y,loss_function):
        self.model.eval()
        with torch.no_grad():
            y_prediction = self.model(x)
            # print(y_prediction)
            mse = loss_function(y_prediction, y)
            return float(mse)
