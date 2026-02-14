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

class HallucinationClassifier(nn.Module):
    """
    Simple feed-forward classifier for hallucination detection using the last token of a selected layer.
    Input: (B, L, D) where L=sequence length, D=activation dim (e.g., 4096)
    Output: (B, 1) sigmoid probability of hallucination
    """
    def __init__(self, dim, layer_index=0):
        super().__init__()
        self.layer_index = layer_index
        self.net = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        """
        x: (B, L, D) or list of (B, L, D) tensors for multiple layers
        returns: (B, 1) sigmoid probability
        """
        # If x is a list/tuple of layer activations, select the specified layer
        if isinstance(x, (list, tuple)):
            x = x[self.layer_index]
        
        # Convert to fp32 if needed (activations might be saved as fp16)
        x = x.float()
        
        # Take only the last token's activations
        last_token = x[:, -1, :]  # (B, D)
        
        # Pass through feed-forward layers
        return torch.sigmoid(self.net(last_token))  # (B, 1) - removed squeeze()


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioner.

    Learns per-index scale (gamma) and shift (beta) vectors so that:
        h' = gamma(n) * h + beta(n)

    Initialised to identity (gamma=1, beta=0) so the model starts as
    if unconditioned and gradually learns layer-specific modulation.
    """

    def __init__(self, num_indices: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Embedding(num_indices, feature_dim)
        self.beta = nn.Embedding(num_indices, feature_dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """h: (B, D) or (B, L, D), idx: (B,) long → same shape as h."""
        g = self.gamma(idx)  # (B, D)
        b = self.beta(idx)   # (B, D)
        if h.dim() == 3:
            # Broadcast over sequence dimension: (B, D) → (B, 1, D)
            g = g.unsqueeze(1)
            b = b.unsqueeze(1)
        return g * h + b


class LayerAwareProgressiveCompressor(nn.Module):
    """Layer-aware variant of `ProgressiveCompressor`.

    Encodes activations into a fixed-size embedding and conditions the
    representation on `layer_idx` using one of several strategies:

    * ``"film_in"`` – FiLM applied to the **raw input** before the
      encoder.  Normalises cross-layer distribution shift so the
      encoder sees standardised activations.
    * ``"film_out"`` – FiLM applied to the **encoder output**.
      Post-hoc rescaling of the learned representation.
    * ``"film_both"`` – FiLM on both input and output (two independent
      FiLM modules).  Input-side handles distribution normalisation,
      output-side handles representation-space calibration.
    * ``"positional"`` – additive conditioning.  A learned layer
      embedding is added to the encoder output: ``z' = z + e(n)``.
      Lightweight but assumes activations are already roughly
      homogeneous across layers.
    * ``"concatenate"`` – the original strategy.  Layer embedding is
      projected, concatenated with ``z``, and fused through a linear
      layer.
    * ``"none"`` – no layer conditioning at all.  ``layer_idx`` is
      accepted but ignored; useful as a baseline.

    Forward accepts keyword arguments so evaluation utilities can pass
    `layer_idx=...` without positional coupling.

    Parameters
    ----------
    num_layers : int
        Number of distinct layer indices.
    input_dim : int
        Activation dimension (e.g. 4096 for Llama-7B).
    final_dim : int
        Output embedding dimension.
    layer_embed_dim : int
        Intermediate layer embedding size (used by ``"concatenate"`` and
        ``"positional"`` modes).
    conditioning : str
        One of ``"film_in"``, ``"film_out"``, ``"film_both"``,
        ``"positional"``, ``"concatenate"``, ``"none"``.
        Defaults to ``"positional"`` (FiLM off).
    normalize_output : bool
        If True, L2-normalize the output embeddings.  Essential when
        using a contrastive loss that computes dot-product similarity
        (e.g. ``SupConLoss``).  Default ``False``.
    dropout, input_dropout : float
        Dropout rates forwarded to the inner encoder.
    """

    VALID_CONDITIONING = {"film_in", "film_out", "film_both", "positional", "concatenate", "none"}

    def __init__(
        self,
        *,
        num_layers: int,
        input_dim: int = 4096,
        final_dim: int = 512,
        layer_embed_dim: int = 128,
        conditioning: str = "positional",
        normalize_output: bool = False,
        dropout: float = 0.1,
        input_dropout: float = 0.2,
    ):
        super().__init__()
        conditioning = conditioning.lower()
        if conditioning not in self.VALID_CONDITIONING:
            raise ValueError(
                f"conditioning must be one of {self.VALID_CONDITIONING}, got '{conditioning}'"
            )
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be > 0")

        self.conditioning = conditioning
        self.normalize_output = bool(normalize_output)

        self.encoder = ProgressiveCompressor(
            input_dim=int(input_dim),
            final_dim=int(final_dim),
            dropout=float(dropout),
            input_dropout=float(input_dropout),
        )

        # --- conditioning-specific modules ---
        if conditioning in ("film_in", "film_both"):
            # Input-side FiLM: operates on (B, L, input_dim)
            self.film_in = FiLMConditioner(int(num_layers), int(input_dim))

        if conditioning in ("film_out", "film_both"):
            # Output-side FiLM: operates on (B, final_dim)
            self.film_out = FiLMConditioner(int(num_layers), int(final_dim))

        if conditioning == "positional":
            self.layer_embedding = nn.Embedding(int(num_layers), int(final_dim))

        elif conditioning == "concatenate":
            self.layer_embedding = nn.Embedding(int(num_layers), int(layer_embed_dim))
            self.layer_proj = nn.Linear(int(layer_embed_dim), int(final_dim))
            self.fuse = nn.Linear(int(final_dim) * 2, int(final_dim))

        self.norm = nn.LayerNorm(int(final_dim))

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _coerce_layer_idx(self, layer_idx, batch_size: int, device: torch.device) -> torch.Tensor:
        """Normalise ``layer_idx`` to a (B,) long tensor on the right device."""
        if isinstance(layer_idx, int):
            return torch.full((batch_size,), layer_idx, dtype=torch.long, device=device)

        if isinstance(layer_idx, torch.Tensor):
            layer_idx = layer_idx.to(device=device, dtype=torch.long, non_blocking=True)
            if layer_idx.dim() == 0:
                return layer_idx.view(1).expand(batch_size)
            if layer_idx.dim() > 1:
                layer_idx = layer_idx.view(-1)
            if layer_idx.shape[0] != batch_size:
                raise ValueError(
                    f"layer_idx batch size {layer_idx.shape[0]} != x batch size {batch_size}"
                )
            return layer_idx

        raise TypeError(f"layer_idx must be int or Tensor, got {type(layer_idx)}")

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, *, layer_idx=None, **kwargs) -> torch.Tensor:
        _ = kwargs

        if layer_idx is None:
            z = self.encoder(x)
            if self.normalize_output:
                z = F.normalize(z, dim=-1)
            return z

        idx = self._coerce_layer_idx(layer_idx, x.shape[0], x.device)

        # --- input-side FiLM ---
        if self.conditioning in ("film_in", "film_both"):
            x = self.film_in(x, idx)  # (B, L, D) → (B, L, D)

        z = self.encoder(x)  # (B, final_dim)

        # --- output-side conditioning ---
        if self.conditioning in ("film_out", "film_both"):
            z = self.film_out(z, idx)  # (B, final_dim) → (B, final_dim)

        elif self.conditioning == "positional":
            e = self.layer_embedding(idx)  # (B, final_dim)
            z = z + e

        elif self.conditioning == "concatenate":
            e = self.layer_embedding(idx)
            e = self.layer_proj(e)
            z = self.fuse(torch.cat([z, e], dim=-1))

        # conditioning == "none": no-op, z passes through unchanged

        z = self.norm(z)

        if self.normalize_output:
            z = F.normalize(z, dim=-1)

        return z
