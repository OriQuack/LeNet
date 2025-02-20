import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# TODO: Depthwise convolution 구현 안되어 있음
# TODO: fine-tuning 모드에서 이미지 resolution이 다를 때 interpolation 구현
# TODO: 각종 augmentation & regularization 구현
class ConvMixer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=10,
        hidden_dim=512,
        nlayer=12,
        patch_size=4,
        kernel_size=7,
        dropout=0.1,
    ):
        super(ConvMixer, self).__init__()
        input_channel = input_dim[1]
        input_size = input_dim[2]

        assert input_size % patch_size == 0

        self.patch_size = patch_size
        self.seqlen = input_size**2 // patch_size**2
        self.num_features = patch_size**2 * input_channel

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        self.emb_conv = nn.Conv2d(
            input_channel, hidden_dim, patch_size, stride=patch_size
        )
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(hidden_dim)

        self.mixerLayers = nn.ModuleList()
        for i in range(nlayer):
            mixerLayer = MixerLayer(hidden_dim, kernel_size)
            self.mixerLayers.append(mixerLayer)

        self.avg_pool = nn.AvgPool2d(input_size // patch_size)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, labels):
        x = self.bn(self.gelu(self.emb_conv(inputs)))

        for layer in self.mixerLayers:
            x = layer(x)

        x = self.avg_pool(x)
        x = torch.squeeze(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs


class MixerLayer(nn.Module):
    def __init__(self, channel, k):
        super(MixerLayer, self).__init__()
        self.depthconv = nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(channel)

        self.pointconv = nn.Conv2d(channel, channel, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, inputs):
        x = self.bn1(self.gelu(self.depthconv(inputs)))
        x = inputs + x
        x = self.bn2(self.gelu(self.pointconv(x)))
        return x
