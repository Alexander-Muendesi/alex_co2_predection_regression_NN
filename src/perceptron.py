import torch
import torch.nn as nn
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import copy
import torch.optim as optim


class Perceptron(nn.Module):
    def __init__(self,data_reader, num_epochs, batch_size, learning_rate, momentum):
        super(Perceptron, self).__init__()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_inputs = data_reader.get_num_inputs()
        np.random.seed(0)

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

        self.model = nn.Sequential(
            nn.Linear(self.num_inputs,1)
        )

    def forward(self, x):
        return self.model(x)
    
    def set_np_seed(self, seed):
        np.random.seed(seed)

    def train(self):
        loss_function = nn.MSELoss()
        optimizer = optim.ASGD(lr=self.learning_rate)

        batch_start = torch.arange(0, len(self.train_tensor_x), self.batch_size)

        best_mse = np.inf
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
        return best_mse
    
    def test(self):
        loss_function = nn.MSELoss()
        return self.evaluate_model(self.validation_tensor_x,self.validation_tensor_y,loss_function)
    
    # will be used to evalulate the validation and test sets
    def evaluate_model(self, x,y,loss_function):
        self.model.eval()
        with torch.no_grad():
            y_prediction = self.model(x)
            print(y_prediction)
            mse = loss_function(y_prediction, y)
            return float(mse)
