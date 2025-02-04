import torch


class LeNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(LeNet, self).__init__()

        # LOSS
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # NET
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.conv1 = torch.nn.Conv2d(input_channel, 6, 5, stride=1)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, stride=1)

        self.pool = torch.nn.MaxPool2d(2)

        self.relu = torch.nn.ReLU()

        dim = ((input_size - 5 + 1) // 2 - 5 + 1) // 2

        self.fc1 = torch.nn.Linear(16 * dim * dim, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, inputs, labels):
        x = self.pool(self.relu(self.conv1(inputs)))
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten feature map into 1D
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        outputs = self.fc3(x)

        loss = self.loss_fn(outputs, labels)

        return loss, outputs

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
