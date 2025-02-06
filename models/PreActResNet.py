import torch


class PreActResNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(PreActResNet, self).__init__()

        # Loss
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Net
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.conv1 = torch.nn.Conv2d(input_channel, 64, 3)

        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()

        self.conv_res = torch.nn.Conv2d(64, 64, 3, padding=1)
        # Kaiming Initialization
        torch.nn.init.kaiming_normal_(
            self.conv_res.weight, mode="fan_in", nonlinearity="relu"
        )
        # Residual block
        self.res_block = torch.nn.Sequential(
            self.bn,
            self.relu,
            self.conv_res,
        )

        dim = input_size - 2
        self.avg_pool = torch.nn.AvgPool2d(dim)

        self.fc = torch.nn.Linear(64, 10)
        torch.nn.init.kaiming_normal_(
            self.fc.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, inputs, labels):
        x = self.conv1(inputs)

        for _ in range(10):
            x = self.res_block(x) + x

        x = self.avg_pool(x)
        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)

        return loss, outputs
