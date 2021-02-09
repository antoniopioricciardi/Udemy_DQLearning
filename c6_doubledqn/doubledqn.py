import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DDQN(nn.Module):
    def __init__(self, lr, input_n, output_n, name, checkpoint_dir):
        """

        :param lr:
        :param input_n:
        :param output_n: number of actions
        :param name: name of the network, for saving
        :param checkpoint_dir: directory in which to save the network
        """
        super(DDQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        # input_n[0] is the number of channels for the input images (4x1, 4 frames by one channel since we have grayscaled images)
        # 32 number of outgoing filters
        self.conv1 = nn.Conv2d(input_n[0], 32, kernel_size=8, stride=8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fc_input_dims = self.calc_conv_output_dims(input_n)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, output_n)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        # self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        # np.prod returns the product of array elements, in this case we multiply all the values of the size of a tensor
        # meaning that we are "flattening" all the output channels of the conv and multiplying their dimension,
        # the give them in input as ONE long layer into fc1.
        return int(np.prod(dims.size()))

    def forward(self, state):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))  # shape of (batch_size*num_filters*height*width) - h & w are of the final convolved image, not input img
        # reshape conv3 out to need compatible with the input of the fc1
        conv_state = state.view(state.size()[0], -1)  # select 1st dim (batch_size) and with -1 flatten the rest of the dimensions
        state = F.relu(self.fc1(conv_state))
        actions = self.fc2(state)  # action values

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))