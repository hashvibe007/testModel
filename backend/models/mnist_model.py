import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    def __init__(self, conv_layers):
        super(MNISTNet, self).__init__()
        self.conv_layers = nn.ModuleList()

        # Input channels for first conv layer
        in_channels = 1

        # Create conv layers based on config
        for out_channels in conv_layers:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            in_channels = out_channels

        # Calculate final features for FC layer
        self.fc_features = conv_layers[-1] * 7 * 7

        self.fc1 = nn.Linear(self.fc_features, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.max_pool2d(x, 2)

        x = x.view(-1, self.fc_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
