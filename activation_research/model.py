import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model_in, d_model_out, nhead=8, ff_multiplier=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_model_in, d_model_out)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model_out,
            nhead=min(nhead, d_model_out // 64),  # clamp num heads to dimension
            dim_feedforward=d_model_out * ff_multiplier,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model_out)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.norm(x)

class ProgressiveCompressor(nn.Module):
    def __init__(self, input_dim=4096, final_dim=512, dropout=0.1, input_dropout=0.2):
        super().__init__()

        dims = []
        d = input_dim
        while d > final_dim:
            next_d = max(d // 2, final_dim)
            dims.append((d, next_d))
            d = next_d

        self.pos_encodings = PositionalEncoding(input_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_in, d_out, dropout=dropout)
            for (d_in, d_out) in dims
        ])
        self.final_proj = nn.Linear(dims[-1][1], final_dim)

        self.dropout = nn.Dropout(p=input_dropout)

    def forward(self, x):
        """
        x: (B, L, 4096)
        returns: (B, 512)
        """

        x = self.dropout(x)
    
        x = self.pos_encodings(x)
        for block in self.blocks:
            x = block(x)

        # Mean pooling over sequence dimension (L)
        x_pooled = x.mean(dim=1)  # (B, dim)
        return self.final_proj(x_pooled)  # (B, 512)

class LastLayerHaluClassifier(nn.Module):
    """
    Transformer-based classifier for hallucination detection using last layer activations.
    Input: (B, L, D) where L=sequence length, D=activation dim (e.g., 4096)
    Output: (B, 1) sigmoid probability of hallucination
    """
    def __init__(self, input_dim=4096, n_blocks=4, hidden_dim=512, nhead=8, ff_multiplier=4, dropout=0.1, max_len=63):
        super().__init__()
        self.pos_encodings = PositionalEncoding(input_dim, max_len=max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(input_dim if i == 0 else hidden_dim, hidden_dim, nhead=nhead, ff_multiplier=ff_multiplier, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (B, L, D)
        returns: (B, 1) sigmoid probability
        """
        x = self.pos_encodings(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x_pooled = x.mean(dim=1)  # mean pooling over sequence
        logits = self.classifier(x_pooled)
        return torch.sigmoid(logits)

class SimpleHaluClassifier(nn.Module):
    """
    Simple feed-forward classifier for hallucination detection using only the last token's activations.
    Input: (B, L, D) where L=sequence length, D=activation dim (e.g., 4096)
    Output: (B, 1) sigmoid probability of hallucination

    This is SAPLMA
    """
    def __init__(self, input_dim=4096, hidden_dims=[2048, 1024, 512], dropout=0.1):
        super().__init__()
        
        # Create feed-forward layers with decreasing dimensions
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Final classification layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, L, D)
        returns: (B, 1) sigmoid probability
        """
        # Take only the last token's activations
        last_token = x[:, -1, :]  # (B, D)
        
        # Pass through feed-forward layers
        logits = self.classifier(last_token)
        return torch.sigmoid(logits)
