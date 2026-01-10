import torch
import torch.nn as nn
import torch.nn.functional as F


class FlappyCNN(nn.Module):
    """
    CNN care primește:
        (B, C, 84, 84)
    și scoate:
        (B, 2)  -> Q(nu_sari), Q(sari)
    """

    def __init__(self, input_channels=4, num_actions=2):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculăm dimensiunea flatten automat
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            x = self._forward_conv(dummy)
            self.flatten_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        # x: (B, C, 84, 84)
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q
