import torch
from typing import Optional, Tuple


def compute_qkv(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X:
            Input tensor of shape (seq_len, d_model)
            or (batch_size, seq_len, d_model)

        W_q, W_k, W_v:
            Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V:
            Each has the same shape as X.
    """

    if X.ndim not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, but got shape {X.shape}")

    d_model = X.shape[-1]

    if W_q.shape != (d_model, d_model):
        raise ValueError(f"W_q must have shape ({d_model}, {d_model}), but got {W_q.shape}")

    if W_k.shape != (d_model, d_model):
        raise ValueError(f"W_k must have shape ({d_model}, {d_model}), but got {W_k.shape}")

    if W_v.shape != (d_model, d_model):
        raise ValueError(f"W_v must have shape ({d_model}, {d_model}), but got {W_v.shape}")

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V


def prepare_attention_mask(
    mask: Optional[torch.Tensor],
    scores: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Prepare attention mask so it can broadcast with attention scores.
    """
    if mask is None:
        return None

    mask = mask.to(device=scores.device, dtype=torch.bool)

    if scores.ndim == 2:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return mask

    if scores.ndim == 4:
        batch_size = scores.shape[0]
        query_len = scores.shape[-2]
        key_len = scores.shape[-1]

        if mask.ndim == 2 and mask.shape == (query_len, key_len):
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 2 and mask.shape == (batch_size, key_len):
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim == 4:
            mask = mask
        else:
            raise ValueError(f"Unsupported mask shape {mask.shape} for scores shape {scores.shape}")

        return mask

    raise ValueError(f"Unsupported scores shape {scores.shape}")


def stable_softmax(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = -1,
) -> torch.Tensor:
    """
    Numerically stable softmax with optional mask.
    """
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    scores_max = torch.max(scores, dim=dim, keepdim=True).values
    scores_max = torch.where(torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max))
    scores_shifted = scores - scores_max
    exp_scores = torch.exp(scores_shifted)

    if mask is not None:
        exp_scores = exp_scores.masked_fill(~mask, 0.0)

    exp_sum = torch.sum(exp_scores, dim=dim, keepdim=True)
    exp_sum = exp_sum.clamp_min(torch.finfo(exp_scores.dtype).tiny)
    probabilities = exp_scores / exp_sum
    return probabilities


def self_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q:
            Query tensor of shape (seq_len, d_k)
            or (batch_size, n_heads, seq_len, head_dim)

        K:
            Key tensor with same shape as Q.

        V:
            Value tensor with same shape as Q.

        mask:
            Optional attention mask.

    Returns:
        Attention output with the same shape as V.
    """
    if Q.ndim != K.ndim or Q.ndim != V.ndim:
        raise ValueError(f"Q, K, V must have same rank, but got {Q.ndim}, {K.ndim}, {V.ndim}")

    if Q.shape[-1] != K.shape[-1]:
        raise ValueError(f"Q and K must have same last dimension, got {Q.shape[-1]} and {K.shape[-1]}")

    if K.shape[-2] != V.shape[-2]:
        raise ValueError(f"K and V must have same sequence length, got {K.shape[-2]} and {V.shape[-2]}")

    head_dim = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / (head_dim ** 0.5)
    prepared_mask = prepare_attention_mask(mask, scores)
    attention_weights = stable_softmax(scores, mask=prepared_mask, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output


def multi_head_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n_heads: int,
    W_o: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V:
            Tensors of shape (seq_len, d_model)
            or (batch_size, seq_len, d_model)

        n_heads:
            Number of attention heads.

        W_o:
            Optional output projection of shape (d_model, d_model).

        mask:
            Optional attention mask.

    Returns:
        Tensor with the same shape as Q.
    """
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(f"Q, K, V must have same shape, got {Q.shape}, {K.shape}, {V.shape}")

    if Q.ndim not in (2, 3):
        raise ValueError(f"Q, K, V must be 2D or 3D, but got {Q.shape}")

    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, but got {n_heads}")

    input_was_2d = Q.ndim == 2

    if input_was_2d:
        Q = Q.unsqueeze(0)
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)

    batch_size, seq_len, d_model = Q.shape

    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

    head_dim = d_model // n_heads
    Q = Q.reshape(batch_size, seq_len, n_heads, head_dim)
    K = K.reshape(batch_size, seq_len, n_heads, head_dim)
    V = V.reshape(batch_size, seq_len, n_heads, head_dim)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    head_output = self_attention(Q, K, V, mask=mask)
    head_output = head_output.transpose(1, 2).contiguous()
    output = head_output.reshape(batch_size, seq_len, d_model)

    if W_o is not None:
        if W_o.shape != (d_model, d_model):
            raise ValueError(f"W_o must have shape ({d_model}, {d_model}), but got {W_o.shape}")
        output = output @ W_o

    if input_was_2d:
        output = output.squeeze(0)

    return output


if __name__ == "__main__":
    X = torch.arange(1, 3 * 16 + 1, dtype=torch.float32).reshape(3, 16)

    W_q = torch.eye(16)
    W_k = torch.eye(16)
    W_v = torch.eye(16)
    W_o = torch.eye(16)

    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    result = multi_head_attention(Q, K, V, n_heads=2, W_o=W_o)
    print(result.shape)

    torch.manual_seed(0)
    Q = torch.randn(5, 8)
    K = torch.randn(5, 8)
    V = torch.randn(5, 8)
    result = self_attention(Q, K, V)
    print(result.shape)

    seq_len = 5
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    Q = torch.randn(seq_len, 8)
    K = torch.randn(seq_len, 8)
    V = torch.randn(seq_len, 8)
    result = self_attention(Q, K, V, mask=causal_mask)
    print(result.shape)

    batch_size = 2
    seq_len = 5
    d_model = 8
    n_heads = 2
    X = torch.randn(batch_size, seq_len, d_model)
    W_q = torch.eye(d_model)
    W_k = torch.eye(d_model)
    W_v = torch.eye(d_model)
    W_o = torch.eye(d_model)
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    result = multi_head_attention(Q, K, V, n_heads=n_heads, W_o=W_o)
    print(result.shape)

    padding_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    result = multi_head_attention(Q, K, V, n_heads=n_heads, W_o=W_o, mask=padding_mask)
    print(result.shape)
