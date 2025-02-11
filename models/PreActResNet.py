import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActResNet(nn.Module):
    def __init__(self, input_dim, layers_layout=[64, 128, 256, 512]):
        super(PreActResNet, self).__init__()
        num_blocks = len(layers_layout)

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
            resBlock = PreActResBlock(
                channel,
                first_conv_stride=(layer != 0),
            )
            self.resBlocks.append(resBlock)

        self.avg_pool = nn.AvgPool2d(input_size // 2 ** (num_blocks - 1))

        self.fc = nn.Linear(layers_layout[num_blocks - 1], 10)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs, labels):
        x = self.conv1(inputs)

        for resBlock in self.resBlocks:
            x = resBlock(x)

        x = self.avg_pool(x)
        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs


class PreActResBlock(nn.Module):
    def __init__(self, channel, first_conv_stride=False):
        super(PreActResBlock, self).__init__()
        self.first_conv_stride = first_conv_stride

        if self.first_conv_stride:
            self.conv1 = nn.Conv2d(channel // 2, channel, 3, padding=1, stride=2)
            self.bn1 = nn.BatchNorm2d(channel // 2)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)

        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

        self.bn2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        x = self.conv1(self.relu(self.bn1(inputs)))
        x = self.conv2(self.relu(self.bn2(x)))

        if self.first_conv_stride:
            dim_size = inputs.shape[1]
            inputs = F.pad(inputs, (0, 0, 0, 0, 0, dim_size), "constant", 0)
            inputs = self.pool(inputs)
        outputs = x + inputs

        return outputs
