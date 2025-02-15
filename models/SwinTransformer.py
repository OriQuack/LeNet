import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# TODO: learning rate scheduler (warm-up 포함) 구현
# TODO: fine-tuning 모드에서 이미지 resolution이 다를 때 interpolation 구현
class SwinTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=96,
        num_classes=10,
        nlayer=1,
        layers_layout=[2, 2, 6, 2],
        hidden_dim=[96, 192, 384, 768],
        ff_dim=[96 * 4, 192 * 4, 384 * 4, 768 * 4],
        nhead=[3, 6, 12, 24],
        patch_size=8,
        dropout=0.1,
        fine_tune=False,
    ):
        super(SwinTransformer, self).__init__()
        self.device = torch.device(os.environ["TORCH_DEVICE"])
        self.layers_layout = layers_layout
        self.patch_size = patch_size
        self.seq_len = input_size**2 // patch_size**2
        self.num_features = patch_size**2 * input_channel
        self.fmap_size = input_size // patch_size

        input_channel = input_dim[1]
        input_size = input_dim[2]

        assert input_size % patch_size == 0

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net

        self.stages = nn.ModuleList()
        for i, nblock in enumerate(layers_layout):
            stage = nn.Sequential()
            # Linear embedding or patch merging
            if i == 0:
                linear_emb = nn.Embedding(self.seq_len, d_model)
                stage.append(linear_emb)
            else:
                in_dim = 4 * d_model * (2 ** (i - 1))
                fc = nn.Linear(in_dim, in_dim // 2)
                stage.append(fc)

            for block_set in range(nblock // 2):
                w_block = SwinTransformerBlock(
                    hidden_dim[i],
                    nhead[i],
                    ff_dim[i],
                    dropout,
                    nlayer,
                    self.seq_len,
                    self.fmap_size,
                    shifted=False,
                )
                sw_block = SwinTransformerBlock(
                    hidden_dim[i],
                    nhead[i],
                    ff_dim[i],
                    dropout,
                    nlayer,
                    self.seq_len,
                    self.fmap_size,
                    shifted=True,
                )
                stage.append(w_block)
                stage.append(sw_block)
            self.stages.append(stage)

    def forward(self, inputs, labels):
        patches = self.patch_partition(inputs)

        for i, stage in enumerate(self.stages):
            if i != 0:
                patches = self.patch_merging(patches)
            patches = stage(patches)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs

    def patch_partition(self, inputs):
        B, C, H, W = inputs.shape
        x = inputs.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        # Permute: B, H//p, W//p, p, p, C
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.view(B, self.seq_len, self.num_features)
        return x

    def patch_merging(self, inputs, win_size=7):
        B, S, C = inputs.shape
        x = inputs.view(
            B,
            self.fmap_size // (self.patch_size * 2),
            self.patch_size * 2,
            self.fmap_size // (self.patch_size * 2),
            self.patch_size * 2,
            C,
        )
        # Permute: B, F//2p, F//2p, 2p, 2p, C
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.view(B, -1, 4 * C)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        ff_dim,
        dropout,
        nlayer,
        seq_len,
        fmap_size,
        win_size=7,
        shifted=False,
    ):
        super(SwinTransformerBlock).__init__()
        self.device = torch.device(os.environ["TORCH_DEVICE"])
        self.fmap_size = fmap_size
        self.win_size = win_size
        self.shifted = shifted

        assert seq_len % (win_size**2) == 0
        self.nwindow = seq_len // (win_size**2)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nhead,
            ff_dim,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayer)

    def forward(self, inputs):
        win_len = self.fmap_size // self.win_size
        encoded_list = torch.tensor((), requires_grad=True, device=self.device)

        for i in range(self.nwindow):
            input_window = inputs[
                :, i * (self.win_size**2) : (i + 1) * (self.win_size**2), :
            ]
            if self.shifted:
                idx = torch.arange(self.win_size**2, device=self.device)
                left_mask = idx % self.win_size <= self.win_size // 2
                top_mask = idx <= self.win_size**2 // 2
                # If right-most and bottom-most window
                if (i + 1) % (win_len) == 0 and i >= self.nwindow - win_len:
                    mask_a = (left_mask & top_mask).to(torch.float).to(self.device)
                    mask_b = (~left_mask & top_mask).to(torch.float).to(self.device)
                    mask_c = (left_mask & ~top_mask).to(torch.float).to(self.device)
                    mask_d = (~left_mask & ~top_mask).to(torch.float).to(self.device)
                    a = self.encoder(inputs[input_window], mask=mask) * mask_a
                    b = self.encoder(inputs[input_window], mask=mask) * mask_b
                    c = self.encoder(inputs[input_window], mask=mask) * mask_c
                    d = self.encoder(inputs[input_window], mask=mask) * mask_d
                    x = a + b + c + d
                else:
                    # If right-most window
                    if (i + 1) % (win_len) == 0:
                        mask = left_mask.to(torch.float).to(self.device)
                    # If bottom-most window
                    elif i >= self.nwindow - win_len:
                        mask = top_mask.to(torch.float).to(self.device)
                    # Default
                    else:
                        mask = torch.ones(
                            self.win_size**2, dtype=torch.float, device=self.device
                        )
                    mask_a = mask
                    mask_b = ~mask
                    a = self.encoder(inputs[input_window], mask=mask_a) * mask_a
                    b = self.encoder(inputs[input_window], mask=mask_b) * mask_b
                    x = a + b

            x = self.encoder(inputs[input_window])
            encoded_list = torch.cat((encoded_list, x), 1)

    def window_partition(self, inputs):
        B, S, C = inputs.shape

        if self.shifted:
            # B, patch_H_idx, patch_W_idx, C
            x = inputs.view(B, self.fmap_size, self.fmap_size, C)
            # cyclic shift
            x = torch.roll(
                x, shifts=(-self.win_size // 2, -self.win_size // 2), dim=(1, 2)
            )

        # B, win_H_idx, patch_H_idx, win_W_idx, patch_W_idx, C
        x = inputs.view(
            B,
            self.fmap_size // self.win_size,
            self.win_size,
            self.fmap_size // self.win_size,
            self.win_size,
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5)
        # S index:
        # win1: 0 ~ win_size**2 - 1
        # win2: win_size**2 ~ 2 * win_size**2 - 1
        # ...
        x = x.view(B, S, C)
        return x
