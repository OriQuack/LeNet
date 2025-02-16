import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# TODO: Self-attention에 relative positional bias 구현
# TODO: 각종 augmentation & regularization 구현
class SwinTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=96,
        num_classes=1000,
        nlayer=1,
        layers_layout=[2, 2, 6, 2],
        hidden_dim=[96, 192, 384, 768],
        ff_dim=[96 * 4, 192 * 4, 384 * 4, 768 * 4],
        nhead=[3, 6, 12, 24],
        patch_size=4,
        win_size=7,
        dropout=0.0,
        fine_tune=False,
    ):
        super(SwinTransformer, self).__init__()
        self.device = torch.device(os.environ["TORCH_DEVICE"])
        self.layers_layout = layers_layout
        self.patch_size = patch_size

        input_channel = input_dim[1]
        input_size = input_dim[2]
        assert input_size % patch_size == 0

        self.seq_len = input_size**2 // patch_size**2
        self.num_features = patch_size**2 * input_channel
        self.fmap_size = input_size // patch_size

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net

        cur_seq_len = self.seq_len
        cur_fmap_size = self.fmap_size
        self.stages = nn.ModuleList()
        for i, nblock in enumerate(layers_layout):
            stage = nn.Sequential()

            # Linear embedding
            if i == 0:
                linear_emb = nn.Linear(patch_size**2 * input_channel, d_model)
                stage.append(linear_emb)
            # Patch merging
            else:
                # <Patch merging layer>
                in_dim = 4 * d_model * (2 ** (i - 1))
                fc = nn.Linear(in_dim, in_dim // 2)
                stage.append(fc)

                # Change due to merge
                cur_seq_len = cur_seq_len // 4
                cur_fmap_size = cur_fmap_size // 2

            for block_set in range(nblock // 2):
                w_block = SwinTransformerBlock(
                    hidden_dim[i],
                    nhead[i],
                    ff_dim[i],
                    dropout,
                    nlayer,
                    cur_seq_len,
                    cur_fmap_size,
                    win_size=win_size,
                    shifted=False,
                )
                sw_block = SwinTransformerBlock(
                    hidden_dim[i],
                    nhead[i],
                    ff_dim[i],
                    dropout,
                    nlayer,
                    cur_seq_len,
                    cur_fmap_size,
                    win_size=win_size,
                    shifted=True,
                )
                stage.append(w_block)
                stage.append(sw_block)
            self.stages.append(stage)

        self.fc = nn.Linear(d_model * (2 ** (len(layers_layout) - 1)), num_classes)

    def forward(self, inputs, labels):
        patches = self.patch_partition(inputs)

        for i, stage in enumerate(self.stages):
            if i != 0:
                patches = self.patch_merging(patches, stage=i + 1)
            patches = stage(patches)

        outputs = self.fc(patches)

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs

    def patch_partition(self, inputs):
        # Batch, Channel, Height, Width
        B, C, H, W = inputs.shape
        x = inputs.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        # Permute: B, H//p, W//p, p, p, Channel
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, self.seq_len, self.num_features)
        return x

    def patch_merging(self, inputs, stage):
        # Batch, seq_len: H//p * W//p, D: 2^(stage-2) * d_model
        B, S, D = inputs.shape
        # B, H//2p, 2, W//2p, 2, D
        x = inputs.view(
            B,
            self.fmap_size // 2,
            2,
            self.fmap_size // 2,
            2,
            D,
        )
        # Permute: B, H//2p, W//2p, 2, 2, D
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, -1, 4 * D)

        self.fmap_size = self.fmap_size // 2
        self.patch_size = self.patch_size * 2
        return x


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
        super(SwinTransformerBlock, self).__init__()
        self.device = torch.device(os.environ["TORCH_DEVICE"])
        self.fmap_size = fmap_size
        self.win_size = win_size
        self.shifted = shifted

        assert seq_len % (win_size**2) == 0 and fmap_size % win_size == 0
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
        encoded_windows = []

        if self.shifted:
            idx = torch.arange(self.win_size**2, device=self.device)
            left_mask = idx % self.win_size <= self.win_size // 2
            top_mask = idx <= (self.win_size**2) // 2

            atten_left_mask = left_mask.unsqueeze(1) & left_mask.unsqueeze(0)
            atten_top_mask = top_mask.unsqueeze(1) & top_mask.unsqueeze(0)

            left_mask = left_mask.unsqueeze(0).unsqueeze(2)
            top_mask = top_mask.unsqueeze(0).unsqueeze(2)

        for i in range(self.nwindow):
            start_idx = i * (self.win_size**2)
            end_idx = (i + 1) * (self.win_size**2)
            input_window = inputs[:, start_idx:end_idx, :]

            if self.shifted:
                x = self.apply_shifted_win_encoder(
                    input_window,
                    i,
                    win_len,
                    atten_left_mask,
                    atten_top_mask,
                    left_mask,
                    top_mask,
                )
            else:
                x = self.encoder(input_window)

            encoded_windows.append(x)

        return torch.cat(encoded_windows, dim=1)

    def apply_shifted_win_encoder(
        self,
        input_window,
        i,
        win_len,
        atten_left_mask,
        atten_top_mask,
        left_mask_exp,
        top_mask_exp,
    ):
        # If bottom-right window
        if (i + 1) % win_len == 0 and i >= self.nwindow - win_len:
            mask_a = torch.zeros_like(atten_left_mask)
            mask_b = torch.zeros_like(atten_left_mask)
            mask_c = torch.zeros_like(atten_left_mask)
            mask_d = torch.zeros_like(atten_left_mask)

            mask_a.masked_fill_(~(atten_left_mask & atten_top_mask), float("-inf"))
            mask_b.masked_fill_(~(~atten_left_mask & atten_top_mask), float("-inf"))
            mask_c.masked_fill_(~(atten_left_mask & ~atten_top_mask), float("-inf"))
            mask_d.masked_fill_(~(~atten_left_mask & ~atten_top_mask), float("-inf"))

            a = self.encoder(input_window, mask=mask_a) * (left_mask_exp & top_mask_exp)
            b = self.encoder(input_window, mask=mask_b) * (
                (~left_mask_exp) & top_mask_exp
            )
            c = self.encoder(input_window, mask=mask_c) * (
                left_mask_exp & (~top_mask_exp)
            )
            d = self.encoder(input_window, mask=mask_d) * (
                (~left_mask_exp) & (~top_mask_exp)
            )
            return a + b + c + d

        # If right-most window
        elif (i + 1) % win_len == 0:
            mask_a = torch.zeros_like(atten_left_mask)
            mask_b = torch.zeros_like(atten_left_mask)
            mask_a.masked_fill_(~atten_left_mask, float("-inf"))
            mask_b.masked_fill_(~(~atten_left_mask), float("-inf"))

            a = self.encoder(input_window, mask=mask_a) * left_mask_exp
            b = self.encoder(input_window, mask=mask_b) * (~left_mask_exp)
            return a + b

        # If bottom-most window
        elif i >= self.nwindow - win_len:
            mask_a = torch.zeros_like(atten_top_mask)
            mask_b = torch.zeros_like(atten_top_mask)
            mask_a.masked_fill_(~atten_top_mask, float("-inf"))
            mask_b.masked_fill_(~(~atten_top_mask), float("-inf"))

            a = self.encoder(input_window, mask=mask_a) * top_mask_exp
            b = self.encoder(input_window, mask=mask_b) * (~top_mask_exp)
            return a + b

        # Default
        else:
            return self.encoder(input_window)

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
        x = x.reshape(B, S, C)
        return x
