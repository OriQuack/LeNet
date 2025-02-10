import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    def __init__(
        self, input_dim, layer_layout=[12, 12, 12], growth_rate=8, dropout=0.2
    ):
        super(DenseNet, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout if self.training else 0

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = self.input_dim[1]
        input_size = self.input_dim[2]
        self.conv = nn.Conv2d(input_channel, 16, 3, padding=1)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

        self.denseBlock1 = DenseBlock(16, layer_layout[0], growth_rate, self.dropout)
        block1_out_chan = 16 + growth_rate * (layer_layout[0] - 1)

        self.trans1 = TransitionLayer(block1_out_chan, 1, self.dropout)

        self.denseBlock2 = DenseBlock(
            block1_out_chan, layer_layout[1], growth_rate, self.dropout
        )
        block2_out_chan = block1_out_chan + growth_rate * (layer_layout[1] - 1)

        self.trans2 = TransitionLayer(block2_out_chan, 1, self.dropout)

        self.denseBlock3 = DenseBlock(
            block2_out_chan, layer_layout[2], growth_rate, self.dropout
        )
        block3_out_chan = block2_out_chan + growth_rate * (layer_layout[2] - 1)

        self.avg_pool = nn.AvgPool2d(8)

        self.fc = nn.Linear(block3_out_chan, 10)
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
    def __init__(self, in_channel, layers, k, dropout):
        super(DenseBlock, self).__init__()
        self.device = torch.device(os.environ["TORCH_DEVICE"])
        self.in_channel = in_channel
        self.layers = layers
        self.k = k
        self.dropout = dropout

        self.compfuncs = []
        for layer in range(self.layers - 1):
            comp = CompositeFunction(
                in_channel + self.k * layer, self.k, self.dropout
            ).to(self.device)
            self.compfuncs.append(comp)

    def forward(self, inputs):
        for layer in range(self.layers - 1):
            x = self.compfuncs[layer](inputs)
            inputs = torch.cat((inputs, x), 1)

        return inputs


class CompositeFunction(nn.Module):
    def __init__(self, channel, k, dropout):
        super(CompositeFunction, self).__init__()
        self.channel = channel
        self.k = k
        self.dropout = dropout

        self.bn = nn.BatchNorm2d(self.channel)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(self.channel, self.k, 3, padding=1)
        self.dropout_layer = nn.Dropout2d(self.dropout)

        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs):
        return self.dropout_layer(self.conv(self.relu(self.bn(inputs))))


class TransitionLayer(nn.Module):
    def __init__(self, in_channel, theta, dropout):
        super(TransitionLayer, self).__init__()
        self.theta = theta
        self.dropout = dropout

        out_channel = math.floor(in_channel * theta)
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.pool = nn.AvgPool2d(2, 2)
        self.dropout_layer = nn.Dropout2d(self.dropout)

    def forward(self, inputs):
        return self.pool(self.dropout_layer(self.conv(self.bn(inputs))))
