import torch
import torch.nn as nn

class MACAFNetHybrid(nn.Module):
    def __init__(self, atlas_dims, embed_dim=64, dropout=0.4, n_heads=4, transformer_layers=1):
        """
        atlas_dims : list of input dimensions for each atlas
        embed_dim  : embedding dimension for linear layers
        n_heads    : number of attention heads
        transformer_layers : number of transformer encoder layers
        """
        super(MACAFNetHybrid, self).__init__()

        self.num_atlases = len(atlas_dims)
        self.use_transformer = transformer_layers > 0

        # Embedding layers for each atlas
        self.atlas_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for dim in atlas_dims
        ])

        # Transformer Encoder
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dropout=dropout,
                batch_first=True  # <-- Fix warning
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_layers
            )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)  # Binary classification
        )

    def forward(self, *inputs):
        """
        inputs: one tensor per atlas
        """
        # Embed each atlas
        embeds = [self.atlas_embeds[i](inputs[i]) for i in range(self.num_atlases)]

        # Stack along sequence dim for transformer: (batch, num_atlases, embed_dim)
        x = torch.stack(embeds, dim=1)

        # Apply transformer if enabled
        if self.use_transformer:
            x = self.transformer(x)

        # Pooling: average over atlas dimension
        x = x.mean(dim=1)

        out = self.classifier(x)
        return out
