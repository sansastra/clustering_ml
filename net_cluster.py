import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self, dropout=False, input_size=4, output_size=2):
        super().__init__()
        hidden_layer_size = 32
        self.fc1 = nn.Linear(input_size,hidden_layer_size)
        self.use_dropout = dropout
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.use_dropout = dropout
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size//2)
        self.use_dropout = dropout
        self.fc4 = nn.Linear(hidden_layer_size//2, output_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = torch.sigmoid(self.fc4(x))
        # x = torch.relu(self.fc4(x))
        return x
