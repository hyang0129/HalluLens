"""
ACT-ViT implementation — Vision Transformer over full (layers × tokens × hidden) activation tensors.
Paper: "Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT,"
Bar-Shalom, Frasca, Galron, Ziser, Maron, NeurIPS 2025 (arXiv:2510.00296).

Treats the full activation tensor of a generation (L × N × D) as an "image"
and classifies it with a Vision Transformer. Max-pools in the (L, N) spatial
plane to a fixed resolution, projects hidden dim via a per-LLM LinearAdapter,
tiles into patches, then runs a standard ViT encoder on the patch sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ACTViTConfig:
    """Hyperparameters for ACT-ViT (paper defaults).

    Parameters
    ----------
    input_dim : int
        Hidden dim D of the source LLM (4096 for Llama-3.1-8B and Qwen3-8B).
    L_p : int
        Spatial height (layers) after adaptive max-pool.
    N_p : int
        Spatial width (tokens) after adaptive max-pool.
    patch_h : int
        Patch height in the (L_p, N_p) grid.
    patch_w : int
        Patch width in the (L_p, N_p) grid.
    d_adapter : int
        Output dimension D' of the per-LLM LinearAdapter.
    d_model : int
        ViT encoder hidden dimension.
    num_heads : int
        Number of attention heads in each Transformer block.
    depth : int
        Number of Transformer encoder blocks.
    mlp_ratio : float
        Hidden-dim expansion factor in each MLP block (mlp_hidden = mlp_ratio * d_model).
    dropout : float
        Dropout probability applied in the Transformer encoder.
    """

    input_dim: int = 4096
    L_p: int = 8
    N_p: int = 100
    patch_h: int = 2
    patch_w: int = 10
    d_adapter: int = 256
    d_model: int = 256
    num_heads: int = 8
    depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.L_p % self.patch_h != 0:
            raise ValueError(
                f"L_p ({self.L_p}) must be divisible by patch_h ({self.patch_h})"
            )
        if self.N_p % self.patch_w != 0:
            raise ValueError(
                f"N_p ({self.N_p}) must be divisible by patch_w ({self.patch_w})"
            )

    @property
    def n_patches(self) -> int:
        return (self.L_p // self.patch_h) * (self.N_p // self.patch_w)

    @property
    def patch_dim(self) -> int:
        """Flat patch dimension before the patch embedding linear: patch_h * patch_w * d_adapter."""
        return self.patch_h * self.patch_w * self.d_adapter


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Standard two-layer MLP block used inside each Transformer layer."""

    def __init__(self, d_model: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (LayerNorm before Attn/MLP)."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        # MLP with residual
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ACTViT(nn.Module):
    """Vision Transformer over full (L × N × D) activation tensors.

    Architecture
    ------------
    1. Adaptive max-pool: (B, L, N, D) → (B, L_p, N_p, D)
    2. LinearAdapter: per-LLM projection D → D' (d_adapter)
    3. PatchEmbed2D: tile (L_p, N_p) grid with (patch_h, patch_w) patches
                     → (B, n_patches, patch_h*patch_w*D') → Linear → (B, n_patches, d_model)
    4. CLS token prepended; learned positional encoding added
    5. Transformer encoder (depth blocks)
    6. CLS output → Linear → scalar logit

    Parameters
    ----------
    cfg : ACTViTConfig
        Model hyperparameters.
    """

    def __init__(self, cfg: ACTViTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. Per-LLM linear adapter D → D'
        self.adapter = nn.Linear(cfg.input_dim, cfg.d_adapter, bias=False)

        # 2. Patch embedding: flatten (patch_h, patch_w, D') → d_model
        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model, bias=True)

        # 3. CLS token + positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.n_patches + 1, cfg.d_model))

        # 4. Transformer encoder
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

        # 5. Classification head
        self.head = nn.Linear(cfg.d_model, 1, bias=True)

        # Weight init
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        response_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, L, N, D)
            Activation tensor, float16 or float32. Will be cast to float32.
        response_len : Tensor (B,), optional
            Per-sample response lengths. Accepted but not used; max-pooling
            handles variable-length sequences natively.

        Returns
        -------
        Tensor (B, 1)
            Raw (un-sigmoid'd) logits for the hallucination class.
        """
        cfg = self.cfg
        x = x.float()  # ensure float32 for stable computation
        B, L, N, D = x.shape

        # --- Step 1: Adaptive max-pool (L, N) → (L_p, N_p) per D-slice ---
        # Reshape to (B*D, 1, L, N) for F.adaptive_max_pool2d
        x = x.permute(0, 3, 1, 2)                          # (B, D, L, N)
        x = x.reshape(B * D, 1, L, N)                      # (B*D, 1, L, N)
        x = F.adaptive_max_pool2d(x, (cfg.L_p, cfg.N_p))   # (B*D, 1, L_p, N_p)
        x = x.squeeze(1)                                    # (B*D, L_p, N_p)
        x = x.reshape(B, D, cfg.L_p, cfg.N_p)              # (B, D, L_p, N_p)
        x = x.permute(0, 2, 3, 1)                          # (B, L_p, N_p, D)

        # --- Step 2: LinearAdapter D → D' ---
        x = self.adapter(x)                                 # (B, L_p, N_p, D')

        # --- Step 3: Patch embedding ---
        # Tile into patches of (patch_h, patch_w)
        pH, pW = cfg.patch_h, cfg.patch_w
        nH = cfg.L_p // pH   # number of patches along layer axis
        nW = cfg.N_p // pW   # number of patches along token axis
        d_prime = cfg.d_adapter

        # Rearrange (B, L_p, N_p, D') → (B, nH, pH, nW, pW, D')
        x = x.reshape(B, nH, pH, nW, pW, d_prime)
        # Gather patch dims: (B, nH, nW, pH, pW, D')
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # Flatten each patch: (B, nH*nW, pH*pW*D')
        x = x.reshape(B, nH * nW, pH * pW * d_prime)
        # Project to d_model
        x = self.patch_embed(x)                             # (B, n_patches, d_model)

        # --- Step 4: CLS token + positional encoding ---
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                      # (B, n_patches+1, d_model)
        x = x + self.pos_embed                              # broadcast over B

        # --- Step 5: Transformer encoder ---
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # --- Step 6: CLS output → logit ---
        cls_out = x[:, 0]                                   # (B, d_model)
        logit = self.head(cls_out)                          # (B, 1)
        return logit


# ---------------------------------------------------------------------------
# Contrastive ACT-ViT (issue #134)
# ---------------------------------------------------------------------------


class ContrastiveACTViT(nn.Module):
    """ACT-ViT encoder trained with SupCon + logprob reconstruction (issue #134).

    Reuses the :class:`ACTViT` joint (layers × tokens) encoder but, instead of
    act_vit's BCE classifier head, exposes a contrastive embedding ``z`` (for
    SupCon + KNN/Mahalanobis OOD scoring) plus the auxiliary logprob-recon
    decoder used by the rest of the contrastive method.

    To slot into the existing 4-D contrastive trainer/evaluator without forking
    them, each grid view is **flattened to a sequence** ``(L*N, D)`` upstream
    (in the dataset); this module reshapes it back to ``(L, N, D)`` before the
    ACT-ViT encoder. ``n_layers`` and ``n_tokens`` are therefore fixed per run
    (read from the capture and injected by the routine).

    Interface mirrors :class:`LogprobReconProgressiveCompressor`:
    ``forward(x) -> z`` (eval), ``forward_with_recon(x) -> (z, logprob_pred)``
    (training), ``recon_loss(pred, target) -> (loss, diag)``. Both forward
    methods accept and ignore ``layer_idx`` for trainer/eval call-site
    compatibility.

    Parameters
    ----------
    n_layers, n_tokens : int
        The L and N of the flattened grid (so ``L*N`` = input sequence length).
    final_dim : int
        Contrastive embedding dimension (the SupCon/KNN vector).
    recon_seq_len, recon_hidden_dim, recon_lambda, logprob_var_threshold :
        Logprob-recon aux head config (same semantics as the progressive model).
    Remaining kwargs are the :class:`ACTViTConfig` fields (defaults = paper /
    ~5.5M-param base).
    """

    def __init__(
        self,
        n_layers: int,
        n_tokens: int,
        input_dim: int = 4096,
        final_dim: int = 256,
        recon_seq_len: int = 64,
        recon_hidden_dim: int = 256,
        recon_lambda: float = 1.0,
        logprob_var_threshold: float = 1e-4,
        *,
        L_p: int = 8,
        N_p: int = 100,
        patch_h: int = 2,
        patch_w: int = 10,
        d_adapter: int = 256,
        d_model: int = 256,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_tokens = int(n_tokens)
        self.input_dim = int(input_dim)
        self.recon_seq_len = int(recon_seq_len)
        self.recon_lambda = float(recon_lambda)
        self.logprob_var_threshold = float(logprob_var_threshold)
        # When True, the returned embedding is L2-normalized (unit sphere), so
        # SupCon operates on cosine similarity (textbook geometry) and the KNN
        # eval scores the same normalized vectors. Default False = legacy raw
        # dot-product behavior, preserved for existing cav0-3 configs.
        self.normalize_output = bool(normalize_output)

        cfg = ACTViTConfig(
            input_dim=int(input_dim), L_p=int(L_p), N_p=int(N_p),
            patch_h=int(patch_h), patch_w=int(patch_w), d_adapter=int(d_adapter),
            d_model=int(d_model), num_heads=int(num_heads), depth=int(depth),
            mlp_ratio=float(mlp_ratio), dropout=float(dropout),
        )
        self.encoder = ACTViT(cfg)
        # Drop the BCE classifier: ACTViT.forward now returns the CLS embedding.
        self.encoder.head = nn.Identity()

        # SupCon projection head: CLS (d_model) → contrastive embedding.
        self.proj = nn.Linear(int(d_model), int(final_dim))

        # Auxiliary logprob decoder: z (final_dim) → (recon_seq_len,).
        self.decoder = nn.Sequential(
            nn.Linear(int(final_dim), int(recon_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(recon_hidden_dim), self.recon_seq_len),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L*N, D) flattened grid → z: (B, final_dim)."""
        B = x.shape[0]
        x = x.reshape(B, self.n_layers, self.n_tokens, self.input_dim)
        cls = self.encoder(x)              # (B, d_model) — head is Identity
        z = self.proj(cls)                 # (B, final_dim)
        if self.normalize_output:
            z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor, layer_idx=None) -> torch.Tensor:
        return self._encode(x)

    def forward_with_recon(self, x: torch.Tensor, layer_idx=None):
        z = self._encode(x)
        return z, self.decoder(z)

    def recon_loss(self, logprob_pred: torch.Tensor, logprob_target: torch.Tensor):
        """MSE logprob reconstruction with a variance gate (see progressive model)."""
        target = logprob_target.float()
        logprob_var = float(target.var())
        if logprob_var < self.logprob_var_threshold:
            zero = torch.zeros(1, device=logprob_pred.device).squeeze()
            return zero, {"logprob_var": logprob_var, "suppressed": True}
        if target.shape[-1] != self.recon_seq_len:
            target = F.interpolate(
                target.unsqueeze(1), size=self.recon_seq_len,
                mode="linear", align_corners=False,
            ).squeeze(1)
        loss = F.mse_loss(logprob_pred, target)
        return loss, {"logprob_var": logprob_var, "suppressed": False}
