import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    def __init__(self, input_dim):
        super(StochasticDepth, self).__init__()

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.conv1 = nn.Conv2d(input_channel, 64, 3, padding=1)

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

        if self.training:
            self.survivalProb = SurvivalProbability(layers=4, prob_L=0.5)
        else:
            # In eval mode, survival probability is always 1
            self.survivalProb = SurvivalProbability(layers=4, prob_L=1)

        self.resBlock1 = ResBlock(
            64,
            surv_prob=self.survivalProb(0),
            first_conv_stride=False,
            training=self.training,
        )
        self.resBlock2 = ResBlock(
            128,
            surv_prob=self.survivalProb(1),
            first_conv_stride=True,
            training=self.training,
        )
        self.resBlock3 = ResBlock(
            256,
            surv_prob=self.survivalProb(2),
            first_conv_stride=True,
            training=self.training,
        )
        self.resBlock4 = ResBlock(
            512,
            surv_prob=self.survivalProb(3),
            first_conv_stride=True,
            training=self.training,
        )

        self.avg_pool = nn.AvgPool2d(4)

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
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs


class ResBlock(nn.Module):
    def __init__(self, channel, surv_prob, first_conv_stride=False, training=True):
        super(ResBlock, self).__init__()
        self.training = training
        self.surv_prob = surv_prob
        self.first_conv_stride = first_conv_stride

        # Drop with drop probability
        self.drop = True if random.random() > self.surv_prob else False

        if self.first_conv_stride:
            self.conv1 = nn.Conv2d(channel // 2, channel, 3, padding=1, stride=2)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)

        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)

        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        if self.first_conv_stride:
            dim_size = inputs.shape[1]
            padded_inputs = F.pad(inputs, (0, 0, 0, 0, 0, dim_size), "constant", 0)
            residuals = self.pool(padded_inputs)

        # In case of drop
        if self.drop:
            return residuals

        # Normal residual block
        x = self.relu(self.bn(self.conv1(inputs)))
        x = self.bn(self.conv2(x))

        # In case of eval
        if not self.training:
            x = x * self.surv_prob

        outputs = self.relu(x + residuals)

        return outputs


class SurvivalProbability(nn.Module):
    def __init__(self, layers, prob_L=0.5):
        super(SurvivalProbability, self).__init__()
        self.layers = layers
        self.prob_L = prob_L

    def forward(self, layer):
        return 1 - layer / self.layers * (1 - self.prob_L)
