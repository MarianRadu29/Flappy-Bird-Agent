import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_channels=4, n_actions=2):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # calculez dimensiunea output ului conv
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, 84, 84)
            x = self._forward_conv(dummy)
            self.fc_input_dim = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        return x

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()

        # input normalization
        x = x / 255.0

        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # (batch_size, fc_input_dim)

        x = F.relu(self.fc1(x), inplace=True)

        q = self.fc2(x)

        return q
