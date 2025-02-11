import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class FractalNet(nn.Module):
    def __init__(
        self,
        input_dim,
        layers_layout=[64, 128, 256, 512, 512],
        columns=4,
        loc_drop=0.15,
        drop_path=True,
    ):
        super(FractalNet, self).__init__()
        self.drop_path = drop_path
        self.num_blocks = len(layers_layout)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.conv = nn.Conv2d(input_channel, layers_layout[0], 3, padding=1)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

        self.bn = nn.BatchNorm2d(layers_layout[0])
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

        # Define Fractal Blocks
        in_channel = layers_layout[0]
        self.fractalBlocks = nn.ModuleList()
        self.drop_layers = nn.ModuleList()
        for i, channel in enumerate(layers_layout):
            fractalBlock = FractalBlock(
                in_channel, channel, columns, loc_drop, self.drop_path
            )
            self.fractalBlocks.append(fractalBlock)
            in_channel = channel

            drop_layer = nn.Dropout2d(0.1 * i)
            self.drop_layers.append(drop_layer)

        self.fc = nn.Linear(layers_layout[self.num_blocks - 1], 10)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs, labels):
        x = self.relu(self.bn(self.conv(inputs)))

        for i in range(self.num_blocks):
            x = self.fractalBlocks[i](x)
            x = self.max_pool(x)
            x = self.drop_layers[i](x) if self.training and self.drop_path else x

        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs


class FractalBlock(nn.Module):
    def __init__(self, in_chan, out_chan, columns, loc_drop, drop_path):
        super(FractalBlock, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.columns = columns
        self.loc_drop = loc_drop
        self.drop_path = drop_path

        self.convs = nn.ModuleList()
        first = 0
        for i in range(2**columns - 1):
            if i == first:
                # First conv of each column
                conv = nn.Conv2d(self.in_chan, self.out_chan, 3, padding=1)
                first = (first + 1) * 2 - 1
            else:
                conv = nn.Conv2d(self.out_chan, self.out_chan, 3, padding=1)
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            bn = nn.BatchNorm2d(self.out_chan)
            relu = nn.ReLU()
            comp = nn.Sequential(conv, bn, relu)
            self.convs.append(comp)

    def forward(self, inputs):
        is_local = True
        if self.training and self.drop_path:
            if random.random() > 0.5:
                selected_col = random.choice(range(self.columns)) + 1
                is_local = False

        # Local
        if is_local:
            x = self.traverse_block(1, inputs, 0)
            outputs = torch.mean(x, dim=2)
        # Global
        else:
            outputs = self.traverse_block_global(inputs, selected_col)
        return outputs

    def traverse_block(self, col, inputs, idx):
        if col == self.columns:
            x = self.convs[2 ** (col - 1) - 1 + idx](inputs)
            return torch.unsqueeze(x, 2)

        x = self.convs[2 ** (col - 1) - 1 + idx](inputs)
        x = torch.unsqueeze(x, 2)

        y = torch.mean(self.traverse_block(col + 1, inputs, idx), dim=2)
        idx += 1
        y = self.traverse_block(col + 1, y, idx)

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

            if self.training and self.drop_path and random.random() < self.loc_drop:
                inputs = torch.cat(
                    (inputs[:, :, :path, :, :], inputs[:, :, path + 1 :, :, :]), dim=2
                )

        return inputs

    def traverse_block_global(self, inputs, selected_col):
        num_convs = 2 ** (selected_col - 1)

        for i in range(num_convs):
            inputs = self.convs[2 ** (selected_col - 1) - 1 + i](inputs)

        return inputs
