import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    def __init__(self, input_dim, layers_layout=[64, 128, 256, 512], prob_L=0.5):
        super(StochasticDepth, self).__init__()
        self.num_blocks = len(layers_layout)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.conv1 = nn.Conv2d(input_channel, layers_layout[0], 3, padding=1)

        self.bn = nn.BatchNorm2d(layers_layout[0])
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

        # Build ResBlocks
        self.resBlocks = nn.ModuleList()
        for layer, channel in enumerate(layers_layout):
            resBlock = ResBlock(
                channel,
                self.survival_prob(self.num_blocks, layer, prob_L),
                first_conv_stride=(layer != 0),
            )
            self.resBlocks.append(resBlock)

        self.avg_pool = nn.AvgPool2d(input_size // 2 ** (self.num_blocks - 1))

        self.fc = nn.Linear(layers_layout[self.num_blocks - 1], 10)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs, labels):
        x = self.relu(self.bn(self.conv1(inputs)))

        for resBlock in self.resBlocks:
            x = resBlock(x)

        x = self.avg_pool(x)
        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs

    def survival_prob(self, num_layers, layer, prob_L=0.5):
        return 1 - layer / num_layers * (1 - prob_L)


class ResBlock(nn.Module):
    def __init__(self, channel, surv_prob, first_conv_stride=False):
        super(ResBlock, self).__init__()
        self.surv_prob = surv_prob
        self.first_conv_stride = first_conv_stride

        if self.first_conv_stride:
            self.conv1 = nn.Conv2d(channel // 2, channel, 3, padding=1, stride=2)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)

        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)

        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        residuals = inputs
        if self.first_conv_stride:
            dim_size = inputs.shape[1]
            padded_inputs = F.pad(inputs, (0, 0, 0, 0, 0, dim_size), "constant", 0)
            residuals = self.pool(padded_inputs)

        # In case of drop
        if self.training and random.random() > self.surv_prob:
            return residuals

        # Normal residual block
        x = self.relu(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))

        # In case of eval
        if not self.training:
            x = x * self.surv_prob

        outputs = self.relu(x + residuals)

        return outputs
