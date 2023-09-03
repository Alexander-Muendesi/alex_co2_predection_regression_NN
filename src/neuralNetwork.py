import numpy as np
import torch
import torch.nn as nn

class NeuralNetwork(nn.module):
    def __init__(self,hidden_layer_sizes,data_reader):
        self.hidden_layer_sizes = hidden_layer_sizes