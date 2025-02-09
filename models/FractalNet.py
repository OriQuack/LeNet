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
        dropout=True,
    ):
        super(FractalNet, self).__init__()
        self.ch_layout = channel_layout
        self.columns = columns
        self.loc_drop = 0
        self.dropout = dropout
        self.selected_col = -1

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.conv = nn.Conv2d(input_channel, self.ch_layout[0], 3, padding=1)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

        # If dropout is true, 50% local 50% global
        if self.training and self.dropout:
            if random.random() > 0.5:
                self.selected_col = random.choice(range(columns)) + 1
            self.loc_drop = loc_drop

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.2)
        self.drop3 = nn.Dropout2d(0.3)
        self.drop4 = nn.Dropout2d(0.4)

        self.fracBlock1 = FractalBlock(
            self.ch_layout[0],
            self.ch_layout[0],
            self.columns,
            self.loc_drop,
            self.selected_col,
        )
        self.fracBlock2 = FractalBlock(
            self.ch_layout[0],
            self.ch_layout[1],
            self.columns,
            self.loc_drop,
            self.selected_col,
        )
        self.fracBlock3 = FractalBlock(
            self.ch_layout[1],
            self.ch_layout[2],
            self.columns,
            self.loc_drop,
            self.selected_col,
        )
        self.fracBlock4 = FractalBlock(
            self.ch_layout[2],
            self.ch_layout[3],
            self.columns,
            self.loc_drop,
            self.selected_col,
        )
        self.fracBlock5 = FractalBlock(
            self.ch_layout[3],
            self.ch_layout[4],
            self.columns,
            self.loc_drop,
            self.selected_col,
        )

        self.fc = nn.Linear(512, 10)
        # Kaiming Initialization
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs, labels):
        x = self.relu(self.bn(self.conv(inputs)))

        x = self.fracBlock1(x)
        x = self.max_pool(x)

        x = self.fracBlock2(x)
        x = self.max_pool(x)
        x = self.drop1(x) if self.training and self.dropout else x

        x = self.fracBlock3(x)
        x = self.max_pool(x)
        x = self.drop2(x) if self.training and self.dropout else x

        x = self.fracBlock4(x)
        x = self.max_pool(x)
        x = self.drop3(x) if self.training and self.dropout else x

        x = self.fracBlock5(x)
        x = self.max_pool(x)
        x = self.drop4(x) if self.training and self.dropout else x

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
