import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActResNet(nn.Module):
    def __init__(self, input_dim):
        super(PreActResNet, self).__init__()

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.conv1 = nn.Conv2d(input_channel, 64, 3, padding=1)

        self.bn = nn.BatchNorm2d(64)

        self.resBlock1 = PreActResBlock(64)
        self.resBlock2 = PreActResBlock(128, first_conv_stride=True)
        self.resBlock3 = PreActResBlock(256, first_conv_stride=True)
        self.resBlock4 = PreActResBlock(512, first_conv_stride=True)

        self.relu = nn.ReLU()

        self.avg_pool = nn.AvgPool2d(512)

        self.fc = nn.Linear(512, 10)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs, labels):
        x = self.relu(self.bn(self.conv1(inputs)))

        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)

        x = self.avg_pool(x)
        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)

        return loss, outputs


class PreActResBlock(nn.Module):
    def __init__(self, channel, first_conv_stride=False):
        super(PreActResBlock, self).__init__()
        self.first_conv_stride = first_conv_stride

        if self.first_conv_stride:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, stride=2)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)

        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(self.relu(self.bn(inputs)))
        x = self.conv2(self.relu(self.bn(self.conv2(x))))

        if self.first_conv_stride:
            dim_size = inputs.shape[1]
            inputs = F.pad(inputs, (0, 0, 0, 0, 0, dim_size), "constant", 0)
        outputs = x + inputs

        return outputs
