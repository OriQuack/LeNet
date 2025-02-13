import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VisionTransformer(nn.Module):
    def __init__(
        self, input_dim, nlayer=12, hidden_dim=768, ff_dim=3072, nhead=12, patch_size=16
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
        self.linear_proj = nn.Linear(self.num_features, hidden_dim)
        self.pos_emb = nn.Embedding(self.seq_len + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nhead,
            ff_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayer)

        self.layer_norm = nn.LayerNorm(self.num_features)
        self.fc = nn.Linear(self.num_features, 10)

    def forward(self, inputs, labels):
        print(inputs)
        patches = self.extract_patches(inputs)
        print(patches)

        # Add class patch
        c_emb = torch.zeros(
            patches.shape[0],
            1,
            patches.shape[2],
            device=self.device,
            requires_grad=True,
        )
        patches = torch.concat((c_emb, patches), dim=1)

        x = self.linear_proj(patches)

        # Get position embedding
        pos_row = torch.arange(0, self.seq_len + 1, device=self.device)
        pos = pos_row.unsqueeze(0).repeat(patches.shape[0], 1)
        x = x + self.pos_emb(pos)

        x = self.encoder(x)

        y = x[:, 0, :]
        outputs = self.fc(self.layer_norm(y))

        loss = self.loss_fn(outputs, labels)
        _, outputs = torch.max(outputs, dim=1)

        return loss, outputs

    def extract_patches(self, inputs):
        return rearrange(
            inputs, "b c (h p) (w p) -> b (h w) (p p c)", p=self.patch_size
        )
