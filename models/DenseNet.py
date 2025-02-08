import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    def __init__(self, input_dim, growth_rate):
        super(DenseNet, self).__init__()
        self.input_dim = input_dim
        self.growth_rate = growth_rate
        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = self.input_dim[1]
        input_size = self.input_dim[2]
        self.conv = nn.Conv2d(input_channel, 16, 3, padding=1)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

        # Custom?
        layers = [4, 8, 6]
        self.denseBlock1 = DenseBlock(16, layers[0])
        self.trans1 = TransitionLayer(16 + self.growth_rate * layers[0], 32)
        self.denseBlock2 = DenseBlock(32, layers[1])
        self.trans2 = TransitionLayer(32 + self.growth_rate * layers[1], 64)
        self.denseBlock3 = DenseBlock(64, layers[2])

        self.avg_pool = nn.AvgPool2d(8)

        self.fc = nn.Linear(64 + self.growth_rate * layers[2], 10)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs, labels):
        x = self.conv(inputs)
        x = self.denseBlock1(x)
        x = self.trans1(x)
        x = self.denseBlock2(x)
        x = self.trans2(x)
        x = self.denseBlock3(x)

        x = self.avg_pool(x)

        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs


class DenseBlock(nn.Module):
    def __init__(self, in_channel, layers, k):
        super(DenseBlock, self).__init__()
        self.in_channel = in_channel
        self.layers = layers
        self.k = k

        self.compfuncs = []
        for layer in range(self.layers - 1):
            comp = CompositeFunction(in_channel + k * layer, k)
            self.compfuncs.append(comp)

    def forward(self, inputs):
        for layer in range(self.layers - 1):
            x = self.compfuncs[layer](inputs)
            inputs = torch.cat((inputs, x), 1)

        return inputs


class CompositeFunction(nn.Module):
    def __init__(self, channel, k):
        super(CompositeFunction, self).__init__()
        self.channel = channel
        self.k = k

        self.bn = nn.BatchNorm2d(self.channel)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(self.channel, self.k, 3, padding=1)

        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs):
        return self.conv(self.relu(self.bn(inputs)))


class TransitionLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TransitionLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv = nn.Conv2d(self.in_channel, self.out_channel, 1)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, inputs):
        return self.pool(self.conv(inputs))
