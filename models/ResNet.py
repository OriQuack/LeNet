import torch


class ResNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(ResNet, self).__init__()

        # Loss
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Net
        # Assume input Bx3x32x32
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.conv1 = torch.nn.Conv2d(input_channel, 64, 3)  # Bx64x30x30

        self.bn = torch.nn.BatchNorm2d(64)

        self.conv_res = torch.nn.Conv2d(64, 64, 4)
        # Kaiming Initialization
        torch.nn.init.kaiming_normal_(
            self.conv_res.weight, mode="fan_in", nonlinearity="relu"
        )

        self.res_block = torch.nn.Sequential(
            self.conv_res,
            self.bn,
        )  # 10 Blocks: Bx64x28x28 ... Bx64x10x10

        self.relu = torch.nn.ReLU()

        dim = input_size - 2 - 2 * 10
        self.avg_pool = torch.nn.AvgPool2d(dim)

        self.fc = torch.nn.Linear(64, 10)
        torch.nn.init.kaiming_normal_(
            self.fc.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, inputs, labels):
        x = self.relu(self.bn(self.conv1(inputs)))

        for _ in range(10):
            x = self.relu(self.res_block(x) + x)

        x = self.avg_pool(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)

        return loss, outputs
