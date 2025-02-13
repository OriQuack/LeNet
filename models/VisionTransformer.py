import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=10,
        nlayer=12,
        hidden_dim=768,
        ff_dim=3072,
        nhead=12,
        patch_size=8,
        dropout=0.0,
        fine_tune=False,
    ):
        super(VisionTransformer, self).__init__()
        self.device = torch.device(os.environ["TORCH_DEVICE"])

        input_channel = input_dim[1]
        input_size = input_dim[2]

        assert input_size % patch_size == 0

        self.patch_size = patch_size
        self.seq_len = input_size**2 // patch_size**2
        self.num_features = patch_size**2 * input_channel

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Net
        self.class_token = nn.Embedding(1, hidden_dim)
        self.linear_proj = nn.Linear(self.num_features, hidden_dim)
        self.pos_emb = nn.Embedding(self.seq_len + 1, hidden_dim)

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

        self.layer_norm = nn.LayerNorm(hidden_dim)
        if fine_tune:
            self.class_head = nn.Linear(hidden_dim, num_classes)
            nn.init.zeros_(self.class_head.weight)
        else:
            self.class_head = nn.Sequential(
                nn.Linear(hidden_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, num_classes),
            )

    def forward(self, inputs, labels):
        patches = self.extract_patches(inputs)

        # Add class token
        c_token = self.class_token(
            torch.zeros(inputs.shape[0], 1, dtype=torch.int, device=self.device)
        )

        patches_emb = self.linear_proj(patches)
        patches_emb = torch.concat((c_token, patches_emb), dim=1)

        # Get position embedding
        pos_row = torch.arange(0, self.seq_len + 1, device=self.device)
        pos = pos_row.unsqueeze(0).repeat(patches.shape[0], 1)
        patches_emb = patches_emb + self.pos_emb(pos)

        x = self.encoder(patches_emb)

        # Get class token
        y = x[:, 0, :]
        outputs = self.class_head(self.layer_norm(y))

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs

    def extract_patches(self, inputs):
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
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, self.seq_len, self.num_features)
        return x
        # return rearrange(
        #     inputs, "b c (h p) (w p) -> b (h w) (p p c)", p=self.patch_size
        # )
