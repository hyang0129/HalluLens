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
    """Progressive dimensionality-reducing transformer encoder.

    Parameters
    ----------
    input_dim : int
        Activation dimension (e.g. 4096).
    final_dim : int
        Output embedding dimension (e.g. 512).
    dropout, input_dropout : float
        Dropout rates.
    normalize_input : bool
        If ``True``, apply ``LayerNorm(input_dim)`` to raw activations
        **before** dropout and positional encoding.  This is important
        when the encoder receives activations from different LLM layers
        whose magnitudes can vary significantly (e.g. layer 14 ≈ 50,
        layer 29 ≈ 200).  Without this, the positional encoding
        (amplitude ≈ 1) is negligible and the model must waste capacity
        learning implicit scale normalisation.  Default ``False`` to
        preserve backward compatibility.
    """

    def __init__(self, input_dim=4096, final_dim=512, dropout=0.1, input_dropout=0.2,
                 normalize_input: bool = False):
        super().__init__()

        self.normalize_input = bool(normalize_input)
        if self.normalize_input:
            self.input_norm = nn.LayerNorm(int(input_dim))

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
        x = x.float()

        if self.normalize_input:
            x = self.input_norm(x)

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
      embedding is added to the encoder output: ``z' = z + α·e(n)``
      where α is a **learnable scalar** initialised to a small value
      (default 0.1) so the layer signal starts near-zero and grows
      during training.  This prevents the embedding from dominating
      or being negligible relative to the encoder output.
    * ``"concatenate"`` – the original strategy.  Layer embedding is
      projected, concatenated with ``z``, and fused through a linear
      layer.
    * ``"none"`` – no layer conditioning at all.  ``layer_idx`` is
      accepted but ignored; useful as a baseline.
    * ``None`` (Python ``None``) – **pure encoder mode**.  Equivalent to
      using a plain ``ProgressiveCompressor`` directly: no layer-specific
      modules are created, ``layer_idx`` is accepted but ignored, and
      the output ``LayerNorm`` is skipped so the architecture is
      identical to the standard contrastive encoder.  Use this to
      confirm that the layer-aware training loop reproduces standard
      contrastive results.

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
    conditioning : str or None
        One of ``"film_in"``, ``"film_out"``, ``"film_both"``,
        ``"positional"``, ``"concatenate"``, ``"none"``, or Python
        ``None``.  Defaults to ``"positional"``.
    positional_init_alpha : float
        Initial value for the learnable scale on the positional layer
        embedding.  Only used when ``conditioning="positional"``.
        Default ``0.1``.
    requires_calibration : bool
        If True, the model declares that it needs input-based calibration
        before training.  The trainer will call ``calibrate()`` before the
        first epoch when this flag is set.  Default ``True`` — set to
        ``False`` to skip calibration (e.g. when resuming from a
        checkpoint or using ``conditioning=None``).
    normalize_output : bool
        If True, L2-normalize the output embeddings.  Essential when
        using a contrastive loss that computes dot-product similarity
        (e.g. ``SupConLoss``).  Default ``False``.
    normalize_input : bool
        If True, apply ``LayerNorm`` to raw activations before encoding.
        Recommended when the encoder receives activations from different
        LLM layers with varying magnitudes.  Default ``False``.
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
        conditioning: "str | None" = "positional",
        normalize_output: bool = False,
        normalize_input: bool = False,
        positional_init_alpha: float = 0.1,
        requires_calibration: bool = True,
        calibrate_max_batches: int = 50,
        calibrate_target_ratio: float = 0.01,
        dropout: float = 0.1,
        input_dropout: float = 0.2,
    ):
        super().__init__()
        if conditioning is None:
            conditioning = None  # keep as Python None – pure encoder mode
        else:
            conditioning = conditioning.lower()
            if conditioning not in self.VALID_CONDITIONING:
                raise ValueError(
                    f"conditioning must be one of {self.VALID_CONDITIONING} or None, got '{conditioning}'"
                )
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be > 0")

        self.conditioning = conditioning
        self.normalize_output = bool(normalize_output)

        # Calibration attributes — checked by the trainer.
        # conditioning=None means pure encoder mode, no calibration needed.
        self.requires_calibration = bool(requires_calibration) and conditioning is not None
        self.calibrate_max_batches = int(calibrate_max_batches)
        self.calibrate_target_ratio = float(calibrate_target_ratio)

        self.encoder = ProgressiveCompressor(
            input_dim=int(input_dim),
            final_dim=int(final_dim),
            dropout=float(dropout),
            input_dropout=float(input_dropout),
            normalize_input=bool(normalize_input),
        )

        # --- conditioning-specific modules ---
        if conditioning is not None and conditioning in ("film_in", "film_both"):
            # Input-side FiLM: operates on (B, L, input_dim)
            self.film_in = FiLMConditioner(int(num_layers), int(input_dim))

        if conditioning is not None and conditioning in ("film_out", "film_both"):
            # Output-side FiLM: operates on (B, final_dim)
            self.film_out = FiLMConditioner(int(num_layers), int(final_dim))

        if conditioning == "positional":
            self.layer_embedding = nn.Embedding(int(num_layers), int(final_dim))
            # Learnable scale so the layer signal starts small relative to
            # the encoder output and grows as the model needs it.
            self.positional_alpha = nn.Parameter(
                torch.tensor(float(positional_init_alpha))
            )

        elif conditioning == "concatenate":
            self.layer_embedding = nn.Embedding(int(num_layers), int(layer_embed_dim))
            self.layer_proj = nn.Linear(int(layer_embed_dim), int(final_dim))
            self.fuse = nn.Linear(int(final_dim) * 2, int(final_dim))

        # LayerNorm is only used when conditioning is active; skipping it
        # in pure-encoder mode (None) keeps the architecture identical to
        # a standalone ProgressiveCompressor.
        # When normalize_output is True the LayerNorm stabilises the
        # representation before L2-normalisation.  When normalize_output
        # is False, LayerNorm inflates the vector norm to ~sqrt(final_dim)
        # which causes SupConLoss logits to explode, so we skip it.
        if conditioning is not None and normalize_output:
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
    #  Calibration
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def calibrate(
        self,
        dataloader,
        *,
        max_batches: int = 50,
        target_ratio: float = 0.01,
    ) -> dict:
        """Calibrate layer embeddings to match input activation statistics.

        Runs a quick pass over training batches, measures the per-element
        mean and std of the raw input activations, and re-initialises the
        conditioning parameters so their magnitude is proportional to the
        actual data scale.

        The input statistics are the only meaningful reference before
        training — the encoder output is random noise at init.

        For ``"positional"`` conditioning this rescales the embedding
        weights so each row's norm is ``target_ratio * mean_input_norm``
        and resets ``positional_alpha`` to 1.0 (scale is baked in).

        For ``"film_in"`` / ``"film_both"`` this rescales
        the beta (shift) embeddings proportionally while keeping gamma
        (scale) near identity.

        For ``"concatenate"`` this rescales the layer embedding weights
        proportionally.

        Parameters
        ----------
        dataloader : DataLoader or iterator
            Training dataloader or an iterator over batches (e.g. an
            infinite-stream iterator).  Only the activation tensors are
            used.  When an iterator is passed, consumed batches advance
            the iterator state — this is fine for infinite streams.
        max_batches : int
            Number of batches to sample for statistics (default 50).
        target_ratio : float
            Desired ratio of embedding contribution to input activation
            norm.  Default ``0.01`` means the embedding starts at ~1% of
            the mean input activation magnitude — conservative since the
            encoder will compress/transform the inputs during training.

        Returns
        -------
        dict
            Calibration statistics: ``input_mean_norm``,
            ``input_elem_mean``, ``input_elem_std``,
            ``embedding_target_norm``.
        """
        if self.conditioning is None:
            return {}

        device = next(self.parameters()).device

        norms = []
        elem_vals = []
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            # Accept both raw tensors and dict batches
            if isinstance(batch, dict):
                x = batch.get("views_activations", batch.get("activations"))
                if x is None:
                    continue
                if x.dim() == 4:
                    batch_size, num_views, seq_len, hidden_dim = x.shape
                    x = x.reshape(batch_size * num_views, seq_len, hidden_dim)
            else:
                x = batch
            x = x.to(device).float()
            # x: (B, L, D) — compute per-sample norms over the feature dim
            # Flatten to (B*L, D) to get per-token norms
            flat = x.reshape(-1, x.shape[-1])
            norms.append(flat.norm(dim=-1).cpu())
            elem_vals.append(flat.cpu())

        if not norms:
            return {}

        all_norms = torch.cat(norms, dim=0)
        all_vals = torch.cat(elem_vals, dim=0)
        input_mean_norm = float(all_norms.mean())
        input_elem_mean = float(all_vals.mean())
        input_elem_std = float(all_vals.std())

        target_norm = input_mean_norm * target_ratio

        # --- Rescale embeddings ---
        if self.conditioning == "positional":
            emb = self.layer_embedding.weight  # (num_layers, final_dim)
            row_norms = emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.layer_embedding.weight.copy_(emb * (target_norm / row_norms))
            # Bake scale into weights; reset alpha to 1.0
            self.positional_alpha.fill_(1.0)

        elif self.conditioning in ("film_in", "film_both"):
            beta = self.film_in.beta.weight
            beta_norms = beta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.film_in.beta.weight.copy_(beta * (target_norm / beta_norms))

        if self.conditioning == "concatenate":
            emb = self.layer_embedding.weight
            row_norms = emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.layer_embedding.weight.copy_(emb * (target_norm / row_norms))

        stats = {
            "input_mean_norm": input_mean_norm,
            "input_elem_mean": input_elem_mean,
            "input_elem_std": input_elem_std,
            "embedding_target_norm": target_norm,
        }
        return stats

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, *, layer_idx=None, **kwargs) -> torch.Tensor:
        _ = kwargs

        # Pure-encoder mode (conditioning=None): behave exactly like
        # ProgressiveCompressor – ignore layer_idx entirely.
        if self.conditioning is None:
            z = self.encoder(x)
            if self.normalize_output:
                z = F.normalize(z, dim=-1)
            return z

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
            z = z + self.positional_alpha * e

        elif self.conditioning == "concatenate":
            e = self.layer_embedding(idx)
            e = self.layer_proj(e)
            z = self.fuse(torch.cat([z, e], dim=-1))

        # conditioning == "none": no-op, z passes through unchanged

        if hasattr(self, 'norm'):
            z = self.norm(z)

        if self.normalize_output:
            z = F.normalize(z, dim=-1)

        return z
