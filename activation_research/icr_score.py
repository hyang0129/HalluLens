"""
ICR Score primitive: token-mean Jensen-Shannon Divergence between the top-p
attention distribution and the residual-projection distribution at each response
query token.

Formula follows icr_score.py:217-267 in the upstream XavierZhang2002/ICR_Probe
repository. The non-obvious choice is the z-score-then-softmax normalization
inside js_divergence (notes §10): raw attention values (already a softmax output
from the model) are re-standardized before JSD to equalize temperature across
samples with very different attention sharpness. This deviates from the paper's
clean Eq. 7 but matches the released code exactly for apples-to-apples comparison.
"""

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    # Subtract max for numerical stability; safe for all-zero input.
    e = np.exp(x - x.max())
    return e / e.sum()


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / max(x.std(), 1e-8)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # 1e-12 inside logs prevents log(0) when either distribution has near-zero mass
    # after softmax; clip is unnecessary since both distributions are positive.
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def _js_divergence(a: np.ndarray, w: np.ndarray) -> float:
    """JSD on standardized softmax of two raw score vectors (upstream §10)."""
    a_norm = _softmax(_zscore(a))
    w_norm = _softmax(_zscore(w))
    m = 0.5 * (a_norm + w_norm)
    return 0.5 * _kl_divergence(a_norm, m) + 0.5 * _kl_divergence(w_norm, m)


def compute_icr_score(
    response_attn: np.ndarray,   # (R, R) float32 — single-block, single-sample, response-to-response
    h_block_input: np.ndarray,   # (R, H) float32 — h^{l-1} at response positions
    delta_h: np.ndarray,         # (R, H) float32 — h^l - h^{l-1} at response positions
    response_len: int,            # actual response length (<= R)
    top_p: float = 0.1,          # per notes §3: top_p overrides top_k in upstream
) -> float:
    """Return token-mean ICR Score for one (sample, layer).

    Per icr_score.py:217-267 in the upstream repo, with the standardize-then-softmax
    JSD variant from notes §10. Top-k key positions are the top-p fraction of the
    attention row; projection direction is the l2-normalized h_block_input at those
    positions; projected quantity is delta_h at the query position.
    """
    if top_p <= 0 or top_p > 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    # Cast at boundary — attention may arrive as fp16 from the memmap store.
    response_attn = np.asarray(response_attn, dtype=np.float32)
    h_block_input = np.asarray(h_block_input, dtype=np.float32)
    delta_h = np.asarray(delta_h, dtype=np.float32)

    R = response_attn.shape[0]
    if response_attn.shape != (R, R):
        raise ValueError(
            f"response_attn must be square (R, R), got {response_attn.shape}"
        )
    if h_block_input.shape[0] != R:
        raise ValueError(
            f"h_block_input first dim must match R={R}, got {h_block_input.shape}"
        )
    if delta_h.shape != h_block_input.shape:
        raise ValueError(
            f"delta_h shape {delta_h.shape} must match h_block_input shape {h_block_input.shape}"
        )
    if response_len > R:
        raise ValueError(
            f"response_len={response_len} exceeds R={R}"
        )

    if response_len == 0:
        return 0.0

    # Restrict to valid (response-length) positions upfront; rows beyond
    # response_len are zero-padded and must not contribute to top-k selection.
    attn = response_attn[:response_len, :response_len]  # (response_len, response_len)
    h_in = h_block_input[:response_len]                 # (response_len, H)
    dh = delta_h[:response_len]                         # (response_len, H)

    k = max(1, int(top_p * response_len))

    per_token_jsds = []
    for q in range(response_len):
        attn_row = attn[q]  # (response_len,)

        # Top-k key indices by attention weight (highest first).
        # np.argpartition is O(response_len) but we need sorted order for
        # determinism; argsort of the full row is fine at R_max<=64.
        idx = np.argsort(attn_row)[-k:]  # indices of k largest values

        a_topk = attn_row[idx]  # (k,)

        # Projection lengths: (Δh_q · h_j) / ||h_j|| for each top-k context j.
        # Numerator: (k,) — one dot product per top-k key.
        # Denominator: (k,) — L2 norm of each key's block-input hidden state.
        numerator = dh[q] @ h_in[idx].T              # (k,)
        denominator = np.linalg.norm(h_in[idx], axis=1) + 1e-8  # (k,)
        w_topk = numerator / denominator              # (k,)

        jsd = _js_divergence(a_topk, w_topk)
        per_token_jsds.append(jsd)

    return float(np.mean(per_token_jsds))
