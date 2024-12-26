import torch
import pickle
import warnings
import numpy as np
from torch import nn

class BookRegressor(nn.Module):
    def __init__(self, input_dim, device):
        super(BookRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()

        self.fc_out = nn.Linear(128, 1)

        self.residual = nn.Linear(input_dim, 1)
        self._device = device
        self = self.to(self._device)

    def forward(self, x):
        residual = self.residual(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc_out(x)
        return x + residual

    def load(self, path):
        self.load_state_dict(torch.load("best_model_0_758.pt", map_location=self._device))
        self.eval()
