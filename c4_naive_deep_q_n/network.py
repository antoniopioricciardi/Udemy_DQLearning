import gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_inputs, n_outputs):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*n_inputs, 128)
        self.fc2 = nn.Linear(128, n_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)  # this has n_actions dimension

        return actions
