import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_channels, n_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, 84, 84)
            x = self._forward_conv(dummy)
            self.fc_input_dim = x.view(1, -1).size(1)

        # trunk comun
        self.fc1 = nn.Linear(self.fc_input_dim, 512)

        # Value stream: estimeaza V(s) => cat de buna este starea curenta
        self.fc_val = nn.Linear(512, 1)

        # Advantage stream: estimeaza A(s, a) => cat de buna este fiecare actiune fata de medie
        self.fc_adv = nn.Linear(512, n_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        return x

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()

        # normalizare img 0..255 -> 0..1
        x = x / 255.0

        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x), inplace=True)

        # Calculam Value si Advantage
        val = self.fc_val(x)        # (Batch, 1)
        adv = self.fc_adv(x)        # (Batch, n_actions)

        # Agregare Dueling: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Scadem media pentru stabilitate (identifiability)
        q = val + (adv - adv.mean(dim=1, keepdim=True))

        return q
