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

class LogprobReconProgressiveCompressor(nn.Module):
    """ProgressiveCompressor with auxiliary logprob reconstruction (Mechanism F).

    Adds a lightweight auxiliary decoder ``g: R^d -> R^recon_seq_len`` that,
    during training, reconstructs the per-token logprob sequence from the
    compressed embedding ``z``.  The training objective is:

        L = L_SupCon(z) + λ · L_recon(g(z), ℓ)

    where ``ℓ`` is the per-token logprob sequence and ``L_recon`` is MSE.

    The auxiliary decoder is discarded at inference.  ``forward()`` is
    identical to ``ProgressiveCompressor`` so the model can be used as a
    drop-in replacement downstream.

    Parameters
    ----------
    input_dim : int
        Activation dimension (e.g. 4096 for Llama-8B).
    final_dim : int
        Output embedding dimension.
    dropout, input_dropout : float
        Dropout rates forwarded to the inner encoder.
    normalize_input : bool
        If True, apply ``LayerNorm`` to raw activations before encoding.
    recon_seq_len : int
        Fixed output length for the reconstructed logprob sequence.  The
        actual per-token logprob target is resampled to this length via
        linear interpolation during ``recon_loss()``.  Default 64.
    recon_hidden_dim : int
        Hidden dimension of the two-layer MLP decoder.  Default 256.
    recon_lambda : float
        Weight ``λ`` for the reconstruction term in the combined loss.
        Used by the training function; can be overridden at call time.
    logprob_var_threshold : float
        If the batch-level variance of the logprob target falls below this
        value the reconstruction term is suppressed (returns 0) and a
        ``"suppressed": True`` diagnostic is emitted.  Prevents the
        decoder from wasting capacity when logprob is near-constant.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        final_dim: int = 512,
        dropout: float = 0.1,
        input_dropout: float = 0.2,
        normalize_input: bool = False,
        recon_seq_len: int = 64,
        recon_hidden_dim: int = 256,
        recon_lambda: float = 1.0,
        logprob_var_threshold: float = 1e-4,
    ):
        super().__init__()
        self.recon_seq_len = int(recon_seq_len)
        self.recon_lambda = float(recon_lambda)
        self.logprob_var_threshold = float(logprob_var_threshold)

        self.encoder = ProgressiveCompressor(
            input_dim=int(input_dim),
            final_dim=int(final_dim),
            dropout=float(dropout),
            input_dropout=float(input_dropout),
            normalize_input=bool(normalize_input),
        )

        # Auxiliary decoder: z (B, final_dim) → logprob_pred (B, recon_seq_len)
        self.decoder = nn.Sequential(
            nn.Linear(int(final_dim), int(recon_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(recon_hidden_dim), self.recon_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward — identical to ProgressiveCompressor.

        The auxiliary decoder is **not** called.  Use this path for
        inference and for evaluation utilities that expect a single tensor.

        x : (B, L, input_dim)
        returns : (B, final_dim)
        """
        return self.encoder(x)

    def forward_with_recon(self, x: torch.Tensor):
        """Forward pass with auxiliary reconstruction.

        Returns
        -------
        z : (B, final_dim)
        logprob_pred : (B, recon_seq_len)
        """
        z = self.encoder(x)
        logprob_pred = self.decoder(z)
        return z, logprob_pred

    def recon_loss(
        self,
        logprob_pred: torch.Tensor,
        logprob_target: torch.Tensor,
    ):
        """Compute MSE reconstruction loss with variance diagnostic.

        Parameters
        ----------
        logprob_pred : (B, recon_seq_len)
            Decoder output from ``forward_with_recon``.
        logprob_target : (B, L)
            Per-token logprob sequences for the batch.  May have a
            different length L than ``recon_seq_len``; will be resampled.

        Returns
        -------
        loss : scalar Tensor
            MSE loss, or zero when suppressed.
        diagnostics : dict
            ``logprob_var`` (float) and ``suppressed`` (bool).
        """
        target = logprob_target.float()

        # Variance diagnostic — suppress when logprob is near-constant.
        logprob_var = float(target.var())
        if logprob_var < self.logprob_var_threshold:
            zero = torch.zeros(1, device=logprob_pred.device).squeeze()
            return zero, {"logprob_var": logprob_var, "suppressed": True}

        # Resample target to fixed seq len via linear interpolation.
        # F.interpolate expects (B, C, L) → operate on (B, 1, L_orig).
        if target.shape[-1] != self.recon_seq_len:
            target = F.interpolate(
                target.unsqueeze(1),        # (B, 1, L)
                size=self.recon_seq_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)                    # (B, recon_seq_len)

        loss = F.mse_loss(logprob_pred, target)
        return loss, {"logprob_var": logprob_var, "suppressed": False}


class LogprobAttnReconProgressiveCompressor(nn.Module):
    """ProgressiveCompressor with combined logprob (Mechanism F) + attention
    summary (Mechanism K) auxiliary reconstruction heads.

    Spec: ``specs/issue_75_combined_logprob_attn_recon.md``.

    Training objective assembled by ``train_contrastive_logprob_attn_recon``::

        L = L_SupCon(z)
          + λ_lp   · L_recon_logprob(g_lp(z), ℓ)
          + Σ_d λ_attn · L_recon_attn(g_attn^d(z), A_d)

    where ``d`` iterates over the configured attention direction(s).

    Inference path (``forward(x)``) is identical to ``ProgressiveCompressor``:
    both auxiliary decoders are discarded. F-only and K-only fall out as
    ``recon_lambda=0`` / ``attn_recon_lambda=0`` special cases. Setting
    ``attn_direction="none"`` makes this class behave as a strict superset of
    ``LogprobReconProgressiveCompressor``.

    Parameters
    ----------
    input_dim, final_dim, dropout, input_dropout, normalize_input
        Forwarded to the inner ``ProgressiveCompressor``.
    recon_seq_len, recon_hidden_dim, recon_lambda, logprob_var_threshold
        F head — same semantics as ``LogprobReconProgressiveCompressor``.
    attn_direction : {"forward", "backward", "both", "none"}
        Which attention-reconstruction heads to instantiate. ``"none"``
        disables K entirely.
    attn_offset_k : int
        Layer offset between the rep's source layer ``ℓ`` and the prediction
        target layer ``ℓ ± k``. Consumed by the *dataset*, not the model —
        recorded here for config provenance only.
    attn_target : {"stats", "coarse", "full"}
        Decoder output type:

        - ``"stats"`` : 3 scalars per view (entropy, focal_frac, self_mass).
        - ``"coarse"``: 8×8 binned attention map per LLM layer per view;
          decoder flat output dim = ``attn_num_layers × 64``.
        - ``"full"``  : full r_max×r_max attention map per LLM layer per view;
          decoder flat output dim = ``attn_num_layers × attn_r_max²``.
    attn_num_stat_features : int
        Output dimension of the K decoder per direction when
        ``attn_target="stats"`` (default 3: entropy, focal_frac, self_mass).
    attn_num_layers : int
        Number of LLM attention layers whose maps the coarse/full decoders
        predict. Ignored when ``attn_target="stats"``. Should match the
        model's ``num_layers`` field in the capture config (e.g. 32 for
        Llama-3.1-8B). Default 32.
    attn_r_max : int
        Maximum response length used in the capture (r_max in config).
        Determines the full-target output width (r_max × r_max per layer).
        Ignored for ``"stats"`` and ``"coarse"``. Default 64.
    attn_recon_hidden_dim, attn_recon_lambda, attn_var_threshold
        K head hyperparameters. Lambda > 0 enables the head; set 0 to
        disable while keeping the decoder weights for inspection.
    attn_loss : {"mse", "kl"}
        Loss function for the K head. ``"mse"`` (default) is always valid.
        ``"kl"`` treats each attention row as a categorical distribution and
        computes KL(target || pred); falls back to MSE when
        ``attn_target="stats"`` (scalars are not probability distributions).
    """

    _VALID_DIRECTIONS = ("forward", "backward", "both", "none")
    _VALID_TARGETS = ("stats", "coarse", "full")
    _VALID_ATTN_LOSS = ("mse", "kl")
    # Internal ModuleDict keys cannot use "forward" (collides with nn.Module.forward).
    _DIRECTION_TO_KEY = {"forward": "fwd", "backward": "bwd"}

    def __init__(
        self,
        input_dim: int = 4096,
        final_dim: int = 512,
        dropout: float = 0.1,
        input_dropout: float = 0.2,
        normalize_input: bool = False,
        # F head
        recon_seq_len: int = 64,
        recon_hidden_dim: int = 256,
        recon_lambda: float = 1.0,
        logprob_var_threshold: float = 1e-4,
        # K head
        attn_direction: str = "backward",
        attn_offset_k: int = 4,
        attn_target: str = "stats",
        attn_num_stat_features: int = 3,
        attn_num_layers: int = 32,
        attn_r_max: int = 64,
        attn_recon_hidden_dim: int = 256,
        attn_recon_lambda: float = 1.0,
        attn_var_threshold: float = 1e-5,
        attn_loss: str = "mse",
    ):
        super().__init__()

        attn_direction = str(attn_direction).lower()
        if attn_direction not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"attn_direction must be one of {self._VALID_DIRECTIONS}, "
                f"got {attn_direction!r}"
            )
        if attn_target not in self._VALID_TARGETS:
            raise ValueError(
                f"attn_target must be one of {self._VALID_TARGETS}, "
                f"got {attn_target!r}"
            )
        attn_loss = str(attn_loss).lower()
        if attn_loss not in self._VALID_ATTN_LOSS:
            raise ValueError(
                f"attn_loss must be one of {self._VALID_ATTN_LOSS}, "
                f"got {attn_loss!r}"
            )

        self.recon_seq_len = int(recon_seq_len)
        self.recon_lambda = float(recon_lambda)
        self.logprob_var_threshold = float(logprob_var_threshold)

        self.attn_direction = attn_direction
        self.attn_offset_k = int(attn_offset_k)
        self.attn_target = attn_target
        self.attn_num_stat_features = int(attn_num_stat_features)
        self.attn_num_layers = int(attn_num_layers)
        self.attn_r_max = int(attn_r_max)
        self.attn_recon_lambda = float(attn_recon_lambda)
        self.attn_var_threshold = float(attn_var_threshold)
        self.attn_loss = attn_loss

        self.encoder = ProgressiveCompressor(
            input_dim=int(input_dim),
            final_dim=int(final_dim),
            dropout=float(dropout),
            input_dropout=float(input_dropout),
            normalize_input=bool(normalize_input),
        )

        # F decoder
        self.lp_decoder = nn.Sequential(
            nn.Linear(int(final_dim), int(recon_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(recon_hidden_dim), self.recon_seq_len),
        )

        # K decoders — one per active direction, none when attn_direction="none".
        # The public direction names are "forward" / "backward", but ModuleDict
        # cannot use "forward" as a key because it collides with nn.Module.forward.
        # Internal storage uses "fwd" / "bwd"; we expose helpers to translate.
        if attn_direction == "none":
            active_dirs: tuple[str, ...] = ()
        elif attn_direction == "both":
            active_dirs = ("forward", "backward")
        else:
            active_dirs = (attn_direction,)
        self._active_attn_dirs = active_dirs

        # Flat output dimension of each K decoder, determined by target type.
        # stats:  one stat-vector per view  → attn_num_stat_features scalars.
        # coarse: one 8×8 binned map per LLM attention layer per view.
        # full:   one r_max×r_max map per LLM attention layer per view.
        # The decoder always outputs a flat (B, out_dim) vector; the caller
        # reshapes to (B, attn_num_layers, 8, 8) or (B, attn_num_layers, r_max, r_max)
        # before loss computation.
        if attn_target == "stats":
            _attn_out_dim = self.attn_num_stat_features
        elif attn_target == "coarse":
            _attn_out_dim = self.attn_num_layers * 8 * 8
        else:  # "full"
            _attn_out_dim = self.attn_num_layers * self.attn_r_max * self.attn_r_max
        self._attn_out_dim = _attn_out_dim

        self.attn_decoders = nn.ModuleDict(
            {
                self._DIRECTION_TO_KEY[d]: nn.Sequential(
                    nn.Linear(int(final_dim), int(attn_recon_hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(attn_recon_hidden_dim), _attn_out_dim),
                )
                for d in active_dirs
            }
        )

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard inference forward — identical to ``ProgressiveCompressor``.

        Auxiliary decoders are not called. ``x: (B, L, input_dim)``,
        returns ``(B, final_dim)``.
        """
        return self.encoder(x)

    # ------------------------------------------------------------------ #
    def forward_with_recon(self, x: torch.Tensor):
        """Forward pass returning both auxiliary predictions.

        Returns
        -------
        z : (B, final_dim)
        lp_pred : (B, recon_seq_len)
        attn_pred_by_direction : dict[str, Tensor]
            Keys are subset of ``{"forward", "backward"}``; empty when
            ``attn_direction == "none"``. Shape per target type:

            - ``"stats"`` : ``(B, attn_num_stat_features)``
            - ``"coarse"``: ``(B, attn_num_layers * 64)``  (flat; reshape to ``(B, attn_num_layers, 8, 8)``)
            - ``"full"``  : ``(B, attn_num_layers * r_max^2)`` (flat; reshape similarly)
        """
        z = self.encoder(x)
        lp_pred = self.lp_decoder(z)
        attn_pred = {
            d: self.attn_decoders[self._DIRECTION_TO_KEY[d]](z)
            for d in self._active_attn_dirs
        }
        return z, lp_pred, attn_pred

    # ------------------------------------------------------------------ #
    def recon_loss_lp(
        self,
        logprob_pred: torch.Tensor,
        logprob_target: torch.Tensor,
    ):
        """MSE reconstruction loss for the logprob (F) head.

        Identical contract to ``LogprobReconProgressiveCompressor.recon_loss``:
        NaN-mask + variance-threshold suppression + linear-interpolation
        resample to ``recon_seq_len``.
        """
        target = logprob_target.float()
        logprob_var = float(target.var())
        if logprob_var < self.logprob_var_threshold:
            zero = torch.zeros(1, device=logprob_pred.device).squeeze()
            return zero, {"logprob_var": logprob_var, "suppressed": True}

        if target.shape[-1] != self.recon_seq_len:
            target = F.interpolate(
                target.unsqueeze(1),
                size=self.recon_seq_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        loss = F.mse_loss(logprob_pred, target)
        return loss, {"logprob_var": logprob_var, "suppressed": False}

    # ------------------------------------------------------------------ #
    def recon_loss_attn(
        self,
        attn_pred: torch.Tensor,
        attn_target: torch.Tensor,
    ):
        """Reconstruction loss for one attention direction.

        Handles all three target types (stats / coarse / full) and both
        loss modes (MSE / KL), selected by ``self.attn_loss``.

        ``attn_pred`` and ``attn_target`` are both flat ``(N, D)`` tensors
        where N = batch × views (flattened by the trainer) and D is the
        total number of scalars per view:

        - ``"stats"`` : D = attn_num_stat_features
        - ``"coarse"``: D = attn_num_layers × 64  (8×8 per layer, all layers)
        - ``"full"``  : D = attn_num_layers × r_max² (r_max×r_max per layer)

        NaN cells in ``attn_target`` (out-of-range layers or response positions
        beyond r_eff) are excluded from the loss — valid_count divides the
        total, so longer responses do not dominate.

        MSE path: element-wise squared error over valid cells / valid_count.

        KL path (``attn_loss='kl'``, intended for coarse/full only): treats
        each row of the reshaped ``(..., map_size)`` attention map as a
        categorical distribution. The decoder output is softmax-normalised
        per row; the target is re-normalised over valid cells per row.
        Rows that are fully NaN are skipped. KL is per-row, averaged over
        valid rows. MSE is used as fallback for ``"stats"`` regardless of
        ``attn_loss`` because stats scalars are not probability distributions.
        """
        target = attn_target.float()
        pred = attn_pred.float()
        nan_mask = torch.isnan(target)

        if nan_mask.all():
            zero = torch.zeros(1, device=pred.device).squeeze()
            return zero, {
                "attn_var": float("nan"),
                "suppressed": True,
                "reason": "all-nan",
            }

        valid_target = target[~nan_mask]
        attn_var = float(valid_target.var())
        if attn_var < self.attn_var_threshold:
            zero = torch.zeros(1, device=pred.device).squeeze()
            return zero, {
                "attn_var": attn_var,
                "suppressed": True,
                "reason": "low-variance",
            }

        use_kl = self.attn_loss == "kl" and self.attn_target in ("coarse", "full")

        if use_kl:
            # KL path: reshape to (N, num_layers, map_size) then operate row-wise.
            # For coarse: map_size = 64 (8×8 per layer).
            # For full:   map_size = r_max² per layer.
            # The "row" for KL is the map_size axis (keys within one attention layer).
            N = pred.shape[0]
            D = pred.shape[1]
            if self.attn_target == "coarse":
                map_size = 64  # 8×8
            else:
                map_size = self.attn_r_max * self.attn_r_max
            num_layers = D // map_size

            pred_3d = pred.reshape(N, num_layers, map_size)          # (N, L, S)
            target_3d = target.reshape(N, num_layers, map_size)      # (N, L, S)
            nan_3d = nan_mask.reshape(N, num_layers, map_size)       # (N, L, S)

            # Row-valid: a row is usable when it has at least one non-NaN cell.
            row_valid = ~nan_3d.all(dim=-1)  # (N, L) bool

            if not row_valid.any():
                zero = torch.zeros(1, device=pred.device).squeeze()
                return zero, {
                    "attn_var": attn_var,
                    "suppressed": True,
                    "reason": "all-nan-rows",
                }

            # Softmax-normalise decoder output per row (unconstrained logits → dist).
            pred_dist = F.softmax(pred_3d, dim=-1)  # (N, L, S)

            # Re-normalise target over valid cells per row; skip NaN cells.
            target_filled = target_3d.masked_fill(nan_3d, 0.0)
            row_sums = target_filled.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            target_dist = target_filled / row_sums  # (N, L, S); NaN cells → 0

            # KL(target || pred) per row, only for row_valid rows.
            # kl_div expects log-probabilities for pred; use log_softmax for stability.
            log_pred = F.log_softmax(pred_3d, dim=-1)  # (N, L, S)
            kl_per_cell = target_dist * (target_dist.clamp(min=1e-9).log() - log_pred)
            kl_per_cell = kl_per_cell.masked_fill(nan_3d, 0.0)
            kl_per_row = kl_per_cell.sum(dim=-1)  # (N, L)

            valid_row_count = row_valid.sum().clamp(min=1)
            loss = kl_per_row[row_valid].sum() / valid_row_count
            return loss, {
                "attn_var": attn_var,
                "suppressed": False,
                "valid_count": int(valid_row_count.item()),
            }

        # MSE path (default; also used unconditionally for "stats").
        # Replace NaNs in target with the corresponding prediction so the
        # squared diff is zero (no gradient contribution).
        target_filled = torch.where(nan_mask, pred.detach(), target)
        sq_err = (pred - target_filled).pow(2)
        valid_count = (~nan_mask).sum().clamp(min=1)
        loss = sq_err.sum() / valid_count
        return loss, {
            "attn_var": attn_var,
            "suppressed": False,
            "valid_count": int(valid_count.item()),
        }


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
        x = x.float()
        last_token = x[:, -1, :]  # (B, D)
        logits = self.classifier(last_token)
        return torch.sigmoid(logits)

class SaplmaWithReconHead(nn.Module):
    """SAPLMA classifier with auxiliary logprob reconstruction head.

    Identical inference path to ``SimpleHaluClassifier``: a feed-forward MLP
    on the last token's activation produces a sigmoid hallucination
    probability.  During training, a separate decoder reconstructs the
    per-token logprob sequence from the penultimate (dim ``hidden_dims[-1]``)
    representation ``z``.  Training objective:

        L = L_BCE(sigmoid_logit, y) + λ · L_recon(g(z), ℓ)

    The auxiliary decoder is discarded at inference; ``forward()`` is a
    drop-in replacement for ``SimpleHaluClassifier.forward()``.

    Ablation rationale: isolates the contribution of the contrastive
    objective in ``LogprobReconProgressiveCompressor`` by giving SAPLMA
    the same auxiliary recon target.  See issue #67.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims=(2048, 1024, 512),
        dropout: float = 0.1,
        recon_seq_len: int = 64,
        recon_hidden_dim: int = 256,
        recon_lambda: float = 1.0,
        logprob_var_threshold: float = 1e-4,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        self.recon_seq_len = int(recon_seq_len)
        self.recon_lambda = float(recon_lambda)
        self.logprob_var_threshold = float(logprob_var_threshold)

        body_layers = []
        prev_dim = int(input_dim)
        for h in hidden_dims:
            body_layers.extend([
                nn.Linear(prev_dim, int(h)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            ])
            prev_dim = int(h)
        self.body = nn.Sequential(*body_layers)
        self.head = nn.Linear(prev_dim, 1)

        self.decoder = nn.Sequential(
            nn.Linear(prev_dim, int(recon_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(recon_hidden_dim), self.recon_seq_len),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        last_token = x[:, -1, :]
        return self.body(last_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path — identical to ``SimpleHaluClassifier.forward``.

        x : (B, L, D)
        returns : (B, 1) sigmoid probability of hallucination
        """
        z = self._encode(x)
        return torch.sigmoid(self.head(z))

    def forward_with_recon(self, x: torch.Tensor):
        """Training forward — exposes z and the recon prediction.

        Returns
        -------
        sigmoid_logit : (B, 1)
        z : (B, hidden_dims[-1])
        logprob_pred : (B, recon_seq_len)
        """
        z = self._encode(x)
        sigmoid_logit = torch.sigmoid(self.head(z))
        logprob_pred = self.decoder(z)
        return sigmoid_logit, z, logprob_pred

    def recon_loss(
        self,
        logprob_pred: torch.Tensor,
        logprob_target: torch.Tensor,
    ):
        """MSE reconstruction loss with variance diagnostic.

        Same semantics as
        ``LogprobReconProgressiveCompressor.recon_loss``: suppresses
        the loss when batch logprob variance falls below
        ``logprob_var_threshold`` and resamples the target to
        ``recon_seq_len`` via linear interpolation.
        """
        target = logprob_target.float()

        logprob_var = float(target.var())
        if logprob_var < self.logprob_var_threshold:
            zero = torch.zeros(1, device=logprob_pred.device).squeeze()
            return zero, {"logprob_var": logprob_var, "suppressed": True}

        if target.shape[-1] != self.recon_seq_len:
            target = F.interpolate(
                target.unsqueeze(1),
                size=self.recon_seq_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        loss = F.mse_loss(logprob_pred, target)
        return loss, {"logprob_var": logprob_var, "suppressed": False}


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


class LinearProbe(nn.Module):
    """True linear probe for hallucination detection.

    Single linear layer on pooled activations — the simplest possible
    baseline.  Used for per-layer probing sweeps: train one probe per
    layer, report AUROC per layer.

    Parameters
    ----------
    input_dim : int
        Activation dimension (e.g. 4096).
    pooling : str
        ``"mean"`` pools over the sequence dimension, ``"last"`` takes
        the last token only.
    """

    def __init__(self, input_dim: int = 4096, pooling: str = "mean"):
        super().__init__()
        pooling = pooling.lower().strip()
        if pooling not in ("mean", "last"):
            raise ValueError(f"pooling must be 'mean' or 'last', got '{pooling}'")
        self.pooling = pooling
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        x: (B, L, D)
        returns: (B, 1) sigmoid probability
        """
        x = x.float()
        if self.pooling == "mean":
            pooled = x.mean(dim=1)      # (B, D)
        else:
            pooled = x[:, -1, :]         # (B, D)
        return torch.sigmoid(self.linear(pooled))


class MultiLayerLinearProbe(nn.Module):
    """Multi-layer linear probe for hallucination detection.

    Pools each layer independently over the sequence dimension, concatenates
    across layers, then applies a single linear classifier.  Controls for
    multi-layer information access vs. single-layer linear probe.

    Parameters
    ----------
    input_dim : int
        Per-layer activation dimension (e.g. 4096).
    num_layers : int
        Number of layers whose activations are concatenated.
    pooling : str
        ``"mean"`` pools over the sequence dimension, ``"last"`` takes
        the last token only.
    """

    def __init__(self, input_dim: int = 4096, num_layers: int = 16, pooling: str = "mean"):
        super().__init__()
        pooling = pooling.lower().strip()
        if pooling not in ("mean", "last"):
            raise ValueError(f"pooling must be 'mean' or 'last', got '{pooling}'")
        self.pooling = pooling
        self.num_layers = num_layers
        # LayerNorm prevents sigmoid saturation when activation norms vary across
        # model families (e.g. Qwen3-8B has ~10x larger residual norms than Llama3-8B).
        self.input_norm = nn.LayerNorm(num_layers * input_dim)
        self.linear = nn.Linear(num_layers * input_dim, 1)

    def forward(self, x):
        """
        x: (B, num_layers, T, D)
        returns: (B, 1) sigmoid probability
        """
        x = x.float()
        if self.pooling == "mean":
            pooled = x.mean(dim=2)       # (B, num_layers, D)
        else:
            pooled = x[:, :, -1, :]      # (B, num_layers, D)
        flat = pooled.reshape(pooled.size(0), -1)  # (B, num_layers * D)
        flat = self.input_norm(flat)
        return torch.sigmoid(self.linear(flat))


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


class SimCLRCotrainedModel(nn.Module):
    """ProgressiveCompressor with a binary hallucination classification head.

    Designed for joint SimCLR + BCE training via SimCLRCotrainedTrainer.
    forward() returns only the embedding (used during OOD evaluation).
    forward_with_head() returns (embedding, hallucination_probability) for
    training, with a gradient gate controlling how much the BCE loss influences
    the encoder.

    The gradient gate (bce_grad_gate, 0.0–1.0) works as follows:
      - gate=0.0: BCE loss trains the head weights only; zero gradient reaches
                  the encoder. Equivalent to training contrastive first, then
                  fitting the head on frozen embeddings.
      - gate=1.0: full gradient from BCE flows through the encoder (original
                  co-training behaviour).
      - gate in (0,1): interpolated — the encoder sees a fraction of the BCE
                       gradient proportional to the gate value.

    Mechanism: z_for_head = gate * z + (1 - gate) * z.detach()
    In the forward pass this is identical to z. In the backward pass the
    gradient reaching the encoder from the head is scaled by `gate`.
    The head always receives its own full gradient regardless of gate value.
    """

    def __init__(self, input_dim: int = 4096, final_dim: int = 512,
                 input_dropout: float = 0.3):
        super().__init__()
        self.encoder = ProgressiveCompressor(
            input_dim=input_dim,
            final_dim=final_dim,
            input_dropout=input_dropout,
            normalize_input=True,
        )
        self.head = nn.Sequential(
            nn.Linear(final_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding only (B, final_dim). Used during evaluation."""
        return self.encoder(x)

    def forward_with_head(self, x: torch.Tensor, bce_grad_gate: float = 1.0):
        """Return (embedding, hallucination_prob) for joint training.

        Args:
            x: (B, L, input_dim) activations.
            bce_grad_gate: float in [0, 1]. Controls how much gradient from the
                BCE loss flows back through the encoder. 0 = head-only update,
                1 = full co-training. Default 1.0 (backward-compatible).
        """
        z = self.encoder(x)
        # Gradient gate: scales BCE gradient reaching the encoder.
        # Forward value is unchanged (z_for_head == z numerically).
        gate = float(bce_grad_gate)
        z_for_head = gate * z + (1.0 - gate) * z.detach()
        pred = self.head(z_for_head)
        return z, pred
