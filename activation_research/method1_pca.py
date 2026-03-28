"""Method 1: Cross-Layer Attention on Aligned Projections.

Architecture:
  1. PCAProjector  — fits PCA on pooled multi-layer training activations (frozen)
  2. TokenPoolingGate — learned gate blending mean and last-token pooling (1 param)
  3. CrossLayerPCATransformer — bidirectional transformer over K PCA-projected layers
  4. PairwiseCosineMLP — ablation: 120 pairwise cosines → MLP
  5. ConcatLinear — ablation: concat K layers → linear
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TokenPoolingGate
# ---------------------------------------------------------------------------

class TokenPoolingGate(nn.Module):
    """Learned scalar gate blending mean and last-token pooling.

    pool(x) = sigmoid(g_raw) * x.mean(dim=-2) + (1 - sigmoid(g_raw)) * x[..., -1, :]

    g_raw is initialized to 0.0 → sigmoid(0) = 0.5 (equal mix).
    """

    def __init__(self):
        super().__init__()
        self.g_raw = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence dimension.

        Args:
            x: (..., T, H) — any leading batch dims, token dim second-to-last
        Returns:
            (..., H)
        """
        g = torch.sigmoid(self.g_raw)
        return g * x.mean(dim=-2) + (1.0 - g) * x[..., -1, :]

    @property
    def gate_value(self) -> float:
        """Current gate value in [0, 1]. 1 = pure mean, 0 = pure last."""
        with torch.no_grad():
            return float(torch.sigmoid(self.g_raw).item())


# ---------------------------------------------------------------------------
# PCAProjector
# ---------------------------------------------------------------------------

class PCAProjector:
    """Fits PCA on pooled multi-layer training activations (no nn.Module).

    Usage::

        proj = PCAProjector(n_components=128)
        proj.fit(X_train)        # X_train: (N*K, input_dim) float32 ndarray
        H = proj.transform(X)   # X: (N, input_dim) → (N, n_components)
        proj.save("pca.npz")
        proj2 = PCAProjector.load("pca.npz")
    """

    def __init__(self, n_components: int = 128):
        self.n_components = int(n_components)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray]  = None  # (n_components, input_dim)
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "PCAProjector":
        """Fit PCA on training data.

        Args:
            X: (N, input_dim) float32 ndarray — typically pooled activations
               across all training samples × all layers.
        """
        from sklearn.decomposition import PCA as SklearnPCA

        pca = SklearnPCA(n_components=self.n_components, random_state=42)
        pca.fit(X.astype(np.float32))
        self.mean_ = pca.mean_.astype(np.float32)                          # (input_dim,)
        self.components_ = pca.components_.astype(np.float32)              # (n_components, input_dim)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_.astype(np.float32)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into PCA space.

        Args:
            X: (N, input_dim) float32 ndarray
        Returns:
            (N, n_components) float32 ndarray
        """
        self._check_fitted()
        X = X.astype(np.float32)
        return (X - self.mean_) @ self.components_.T  # (N, n_components)

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable projection for use in training loop.

        Args:
            x: (..., input_dim) float32 tensor
        Returns:
            (..., n_components) float32 tensor
        """
        self._check_fitted()
        device = x.device
        mean = torch.from_numpy(self.mean_).to(device=device, dtype=torch.float32)
        P = torch.from_numpy(self.components_).to(device=device, dtype=torch.float32)
        return (x.float() - mean) @ P.T  # (..., n_components)

    def save(self, path: str) -> None:
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            mean=self.mean_,
            components=self.components_,
            explained_variance_ratio=self.explained_variance_ratio_,
            n_components=np.array([self.n_components]),
        )

    @classmethod
    def load(cls, path: str) -> "PCAProjector":
        data = np.load(path)
        proj = cls(n_components=int(data["n_components"][0]))
        proj.mean_ = data["mean"].astype(np.float32)
        proj.components_ = data["components"].astype(np.float32)
        proj.explained_variance_ratio_ = data["explained_variance_ratio"].astype(np.float32)
        proj._is_fitted = True
        return proj

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("PCAProjector has not been fitted. Call .fit() first.")

    @property
    def cumulative_variance(self) -> Optional[np.ndarray]:
        if self.explained_variance_ratio_ is None:
            return None
        return np.cumsum(self.explained_variance_ratio_)


# ---------------------------------------------------------------------------
# CrossLayerPCATransformer
# ---------------------------------------------------------------------------

class CrossLayerPCATransformer(nn.Module):
    """Main Method 1 model.

    Expects inputs as PCA-projected, token-pooled layer activations.

    Args:
        pca_dim: Dimensionality of PCA-projected activations (e.g. 128).
        num_layers: Total number of LLM layers this model was built for (K=16).
        layer_ids: Absolute LLM layer IDs, e.g. [14, 15, ..., 29].
                   Used to index the positional encoding table.
        nhead: Number of transformer attention heads.
        ffn_dim: Feedforward network dimension inside the transformer.
        dropout: Dropout rate.
        num_transformer_layers: Depth of the bidirectional transformer (default 1).
    """

    def __init__(
        self,
        pca_dim: int = 128,
        num_layers: int = 16,
        layer_ids: Optional[List[int]] = None,
        nhead: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.pca_dim = int(pca_dim)
        self.num_layers = int(num_layers)

        if layer_ids is None:
            layer_ids = list(range(14, 14 + num_layers))
        self.register_buffer(
            "layer_ids",
            torch.tensor(layer_ids, dtype=torch.long),
        )
        self._min_layer_id = int(min(layer_ids))
        self._max_layer_id = int(max(layer_ids))

        # Positional encoding table indexed by absolute layer ID (shifted to 0-based)
        pe_table_size = self._max_layer_id - self._min_layer_id + 1
        self.layer_pe = nn.Embedding(pe_table_size, pca_dim)
        nn.init.normal_(self.layer_pe.weight, std=0.01)

        self.input_norm = nn.LayerNorm(pca_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pca_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    def forward(
        self,
        H: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            H: (B, K, pca_dim) — PCA-projected, token-pooled activations
               for K layers (K can be < num_layers during training subset aug)
            layer_indices: List of K absolute layer IDs for the K input layers.
                           If None, uses self.layer_ids (all num_layers layers).
        Returns:
            z: (B, pca_dim)
        """
        H = H.float()

        if layer_indices is None:
            pe_indices = self.layer_ids - self._min_layer_id   # (K,)
        else:
            pe_indices = (
                torch.tensor(layer_indices, dtype=torch.long, device=H.device)
                - self._min_layer_id
            )

        # LayerNorm across feature dim
        H = self.input_norm(H)                          # (B, K, pca_dim)

        # Add absolute layer positional encoding
        pe = self.layer_pe(pe_indices)                  # (K, pca_dim)
        H = H + pe.unsqueeze(0)                         # (B, K, pca_dim)

        # Bidirectional transformer (no causal mask → full attention)
        Z = self.transformer(H)                         # (B, K, pca_dim)

        # Mean pool over layer dimension
        z = Z.mean(dim=1)                               # (B, pca_dim)
        return z


# ---------------------------------------------------------------------------
# PairwiseCosineMLP  (Ablation 1)
# ---------------------------------------------------------------------------

class PairwiseCosineMLP(nn.Module):
    """Ablation 1: 120 pairwise cosine similarities → 2-layer MLP → pca_dim.

    For K=16 layers: C(16,2) = 120 pairs.

    Args:
        pca_dim: Dimensionality of PCA-projected layer activations.
        num_layers: Number of input layers K (default 16).
        hidden_dim: Hidden dimension of the MLP (default 128).
        out_dim: Output dimensionality (default = pca_dim).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        pca_dim: int = 128,
        num_layers: int = 16,
        hidden_dim: int = 128,
        out_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pca_dim = int(pca_dim)
        self.num_layers = int(num_layers)
        n_pairs = num_layers * (num_layers - 1) // 2
        out_dim = int(out_dim) if out_dim is not None else int(pca_dim)

        self.mlp = nn.Sequential(
            nn.Linear(n_pairs, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosines and pass through MLP.

        Args:
            H: (B, K, pca_dim)
        Returns:
            z: (B, out_dim)
        """
        H = F.normalize(H.float(), dim=-1)  # (B, K, pca_dim) unit vectors
        B, K, _ = H.shape

        cosines = []
        for i in range(K):
            for j in range(i + 1, K):
                c = (H[:, i, :] * H[:, j, :]).sum(dim=-1)  # (B,)
                cosines.append(c)

        cos_features = torch.stack(cosines, dim=1)  # (B, n_pairs)
        return self.mlp(cos_features)               # (B, out_dim)


# ---------------------------------------------------------------------------
# ConcatLinear  (Ablation 2)
# ---------------------------------------------------------------------------

class ConcatLinear(nn.Module):
    """Ablation 2: Concat K PCA-projected layers → Linear → pca_dim.

    Concatenates all K=16 layers (16 × pca_dim = 2048 dim) and projects
    down to pca_dim with a single linear layer.

    Args:
        pca_dim: Dimensionality of PCA-projected layer activations.
        num_layers: Number of input layers K (default 16).
        out_dim: Output dimensionality (default = pca_dim).
    """

    def __init__(
        self,
        pca_dim: int = 128,
        num_layers: int = 16,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.pca_dim = int(pca_dim)
        self.num_layers = int(num_layers)
        out_dim = int(out_dim) if out_dim is not None else int(pca_dim)
        self.linear = nn.Linear(num_layers * pca_dim, out_dim)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Concatenate layers and project.

        Args:
            H: (B, K, pca_dim)
        Returns:
            z: (B, out_dim)
        """
        B = H.shape[0]
        flat = H.float().reshape(B, -1)  # (B, K * pca_dim)
        return self.linear(flat)          # (B, out_dim)
