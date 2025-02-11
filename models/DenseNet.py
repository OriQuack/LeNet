import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    def __init__(
        self,
        input_dim,
        layers_layout=[12, 12, 12],
        growth_rate=8,
        dropout=0.2,
        theta=0.5,
        bottleneck=True,
    ):
        super(DenseNet, self).__init__()
        self.input_dim = input_dim
        self.num_blocks = len(layers_layout)

        # Theta only active for bottleneck
        theta = theta if bottleneck else 1
        dropout = dropout if self.training else 0

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = self.input_dim[1]
        input_size = self.input_dim[2]
        self.conv = nn.Conv2d(input_channel, 2 * growth_rate, 3, padding=1)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

        # Define DenseBlocks
        block_chan = 2 * growth_rate
        self.denseBlocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        for i, layers in enumerate(layers_layout):
            denseBlock = DenseBlock(
                block_chan, layers, growth_rate, dropout, bottleneck
            )
            self.denseBlocks.append(denseBlock)
            block_chan = block_chan + growth_rate * (layers - 1)

            # Transition layer only between desnse blocks
            if i == self.num_blocks - 1:
                break
            trans = TransitionLayer(block_chan, theta, dropout)
            self.trans_layers.append(trans)
            block_chan = math.floor(block_chan * theta)

        self.avg_pool = nn.AvgPool2d(input_size // (2 ** (self.num_blocks - 1)))

        self.fc = nn.Linear(block_chan, 10)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs, labels):
        x = self.conv(inputs)

        for i in range(self.num_blocks):
            x = self.denseBlocks[i](x)
            if i == self.num_blocks - 1:
                break
            x = self.trans_layers[i](x)

        x = self.avg_pool(x)

        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs


class DenseBlock(nn.Module):
    def __init__(self, in_channel, layers, k, dropout, bottleneck):
        super(DenseBlock, self).__init__()

        self.compfuncs = nn.ModuleList()
        for layer in range(layers - 1):
            comp = CompositeFunction(in_channel + k * layer, k, dropout, bottleneck)
            self.compfuncs.append(comp)

    def forward(self, inputs):
        for compfunc in self.compfuncs:
            x = compfunc(inputs)
            inputs = torch.cat((inputs, x), 1)

        return inputs


class CompositeFunction(nn.Module):
    def __init__(self, channel, k, dropout, bottleneck=False):
        super(CompositeFunction, self).__init__()
        self.bottleneck = bottleneck

        self.relu = nn.ReLU()
        if bottleneck:
            self.conv_bottleneck = nn.Conv2d(channel, k * 4, 1)
            nn.init.kaiming_normal_(
                self.conv_bottleneck.weight, mode="fan_in", nonlinearity="relu"
            )
            self.bn_bottleneck = nn.BatchNorm2d(channel)
            channel = k * 4

        self.bn = nn.BatchNorm2d(channel)
        self.conv = nn.Conv2d(channel, k, 3, padding=1)
        self.dropout_layer = nn.Dropout2d(dropout)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs):
        if self.bottleneck:
            inputs = self.dropout_layer(
                self.conv_bottleneck(self.relu(self.bn_bottleneck(inputs)))
            )
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
