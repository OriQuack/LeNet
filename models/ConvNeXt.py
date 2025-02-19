import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: ResNeXt의 depthwise convolution 구현 안되어 있음
# TODO: 각종 augmentation & regularization 구현
class ConvNeXt(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=10,
        channels_layout=[96, 192, 384, 768],
        blocks_layout=[3, 3, 9, 3],
    ):
        super(ConvNeXt, self).__init__()
        num_stages = len(blocks_layout)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        input_channel = input_dim[1]
        input_size = input_dim[2]
        self.patchify_stem = nn.Conv2d(input_channel, channels_layout[0], 4, stride=4)
        self.ln1 = LayerNorm(channels_layout[0])

        # Build ConvBlocks
        self.convBlocks = nn.Sequential()
        for i in range(num_stages):
            for block in range(blocks_layout[i]):
                convBlock = ConvNeXtBlock(channels_layout[i])
                self.convBlocks.append(convBlock)
            # Between blocks
            if i == num_stages - 1:
                break
            ln = LayerNorm(channels_layout[i])
            downsampling = nn.Conv2d(channels_layout[i], channels_layout[i + 1], 2, 2)
            self.convBlocks.append(ln)
            self.convBlocks.append(downsampling)

        self.ln2 = LayerNorm(channels_layout[-1])
        self.avg_pool = nn.AvgPool2d(input_size // 4 // 2 ** (num_stages - 1))

        self.fc = nn.Linear(channels_layout[-1], num_classes)

    def forward(self, inputs, labels):
        x = self.ln1(self.patchify_stem(inputs))
        x = self.ln2(self.convBlocks(x))

        x = self.avg_pool(x)
        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs


class ConvNeXtBlock(nn.Module):
    def __init__(self, channel):
        super(ConvNeXtBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 7, padding=3)
        self.ln = LayerNorm(channel)

        self.conv2 = nn.Conv2d(channel, 4 * channel, 1)
        self.gelu = nn.GELU()

        self.conv3 = nn.Conv2d(4 * channel, channel, 1)

    def forward(self, inputs):
        x = self.ln(self.conv1(inputs))
        x = self.gelu(self.conv2(x))
        x = self.conv3(x)

        outputs = inputs + x

        return outputs


# Apply normalization for each (H, W) along channel dimension
# Shape: (Batch, Channel, H, W)
class LayerNorm(nn.Module):
    def __init__(self, channel):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(channel))
        self.bias = nn.Parameter(torch.zeros(channel))
        self.eps = 1e-6

    def forward(self, x):
        m = x.mean(1, keepdim=True)
        s = (x - m).pow(2).mean(1, keepdim=True)
        x = (x - m) / torch.sqrt(s + self.eps)

        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
