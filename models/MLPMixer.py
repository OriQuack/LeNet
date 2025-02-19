import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# TODO: 각종 augmentation & regularization 구현
class MLPMixer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=10,
        nlayer=12,
        hidden_dim=768,
        patch_size=8,
        dropout=0.1,
    ):
        super(MLPMixer, self).__init__()
        self.device = torch.device(os.environ["TORCH_DEVICE"])

        input_channel = input_dim[1]
        input_size = input_dim[2]

        assert input_size % patch_size == 0

        self.patch_size = patch_size
        self.seqlen = input_size**2 // patch_size**2
        self.num_features = patch_size**2 * input_channel

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        self.linear_proj = nn.Linear(self.num_features, hidden_dim)

        self.mixerLayers = nn.ModuleList()
        for i in range(nlayer):
            mixerLayer = MixerLayer(self.seqlen, hidden_dim)
            self.mixerLayers.append(mixerLayer)

        self.avg_pool = nn.AvgPool1d(self.seqlen)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, labels):
        patches = self.patch_partition(inputs)
        x = self.linear_proj(patches)

        for layer in self.mixerLayers:
            x = layer(x)

        x = self.avg_pool(x)
        outputs = self.fc(x)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs

    def patch_partition(self, inputs):
        B, C, H, W = inputs.shape
        x = inputs.reshape(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        # Permute: B, H//p, W//p, p, p, C
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B, self.seq_len, self.num_features)
        return x


class MixerLayer(nn.Moduel):
    def __init__(self, seqlen, hidden_dim):
        super(MixerLayer, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.mlpBlock1 = TokenMixingMLP(seqlen)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlpBlock2 = ChannelMixingMLP(hidden_dim)

    def forward(self, inputs):
        x = self.ln1(inputs)
        # (B, S, C) -> (B, C, S)
        x = x.transpose(1, 2).contiguous()
        x = self.mlpBlock1(x)
        # (B, C, S) -> (B, S, C)
        x = x.transpose(1, 2).contiguous()
        inter = inputs + x

        x = self.mlpBlock2(self.ln2(inter))
        x = inter + x
        return x


class TokenMixingMLP(nn.Module):
    def __init__(self, seqlen):
        super(TokenMixingMLP, self).__init__()
        self.fc1 = nn.Linear(seqlen, seqlen)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(seqlen, seqlen)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class ChannelMixingMLP(nn.Module):
    def __init__(self, channel):
        super(ChannelMixingMLP, self).__init__()
        self.fc1 = nn.Linear(channel, channel)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(channel, channel)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
