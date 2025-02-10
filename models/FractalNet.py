import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class FractalNet(nn.Module):
    def __init__(
        self,
        input_dim,
        channel_layout=[64, 128, 256, 512, 512],
        columns=3,
        loc_drop=0.15,
        drop_path=True,
    ):
        super(FractalNet, self).__init__()
        self.drop_path = drop_path
        self.num_blocks = len(channel_layout)
        self.selected_col = -1
        loc_drop = 0

        # If drop_path is true, 50% local 50% global
        if self.training and self.drop_path:
            if random.random() > 0.5:
                self.selected_col = random.choice(range(columns)) + 1
            loc_drop = loc_drop

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.conv = nn.Conv2d(input_channel, channel_layout[0], 3, padding=1)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

        in_channel = channel_layout[0]
        self.fractalBlocks = nn.ModuleList()
        self.drop_layers = nn.ModuleList()
        for i, channel in enumerate(channel_layout):
            fractalBlock = FractalBlock(
                in_channel,
                channel,
                columns,
                loc_drop,
                self.selected_col,
            )
            self.fractalBlocks.append(fractalBlock)
            in_channel = channel

            drop_layer = nn.Dropout2d(0.1 * i)
            self.drop_layers.append(drop_layer)

        self.fc = nn.Linear(512, 10)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs, labels):
        x = self.relu(self.bn(self.conv(inputs)))

        for block in range(self.num_blocks):
            x = self.fractalBlocks[block](x)
            x = self.max_pool(x)
            x = self.drop_layers[block](x) if self.training and self.drop_path else x

        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs


class FractalBlock(nn.Module):
    def __init__(self, in_chan, out_chan, columns, loc_drop, selected_col=-1):
        super(FractalBlock, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.columns = columns
        self.loc_drop = loc_drop
        self.selected_col = selected_col

        self.conv1 = nn.Conv2d(self.in_chan, self.out_chan, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_chan, self.out_chan, 3, padding=1)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

        self.bn = nn.BatchNorm2d(self.out_chan)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # Local
        if self.selected_col == -1:
            x = self.build_block(1, inputs, True)
            outputs = torch.mean(x, dim=2)
        # Global
        else:
            outputs = self.build_block_global(inputs)
        return outputs

    def build_block(self, col, inputs, first):
        if col == self.columns:
            if first:
                x = self.relu(self.bn(self.conv1(inputs)))
            else:
                x = self.relu(self.bn(self.conv2(inputs)))
            return torch.unsqueeze(x, 2)

        if first:
            x = self.relu(self.bn(self.conv1(inputs)))
        else:
            x = self.relu(self.bn(self.conv2(inputs)))
        x = torch.unsqueeze(x, 2)

        y = torch.mean(self.build_block(col + 1, inputs, first), dim=2)
        first = False
        y = self.build_block(col + 1, y, first)

        outputs = torch.concat((x, y), dim=2)
        outputs = self.drop_input(outputs)
        return outputs

    def drop_input(self, inputs):
        paths = list(range(inputs.shape[2]))
        random.shuffle(paths)

        for path in paths:
            # Keep at least one path
            if inputs.shape[2] == 1:
                break

            if random.random() < self.loc_drop:
                inputs = torch.cat(
                    (inputs[:, :, :path, :, :], inputs[:, :, path + 1 :, :, :]), dim=2
                )

        return inputs

    def build_block_global(self, inputs):
        num_convs = 2 ** (self.selected_col - 1)
        x = self.relu(self.bn(self.conv1(inputs)))

        for _ in range(num_convs - 1):
            x = self.relu(self.bn(self.conv2(x)))

        return x
