import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    d_model: int
    n_q_heads: int
    n_kv_heads: Optional[int] = None
    dropout: float = 0.0
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    ffn_hidden_dim: Optional[int] = None
    bias: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_q_heads
        if self.n_q_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("n_q_heads and n_kv_heads must be positive")
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        if self.d_model % self.n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = 4 * self.d_model


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps) * self.weight


def compute_qkv(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Educational helper that projects an already-batched input with explicit weights.
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

    return X @ W_q, X @ W_k, X @ W_v


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"x must be 4D, but got shape {x.shape}")
    if n_rep == 1:
        return x
    batch_size, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    dim_positions = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (dim_positions / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"x must be 4D, but got shape {x.shape}")
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


def prepare_attention_mask(
    mask: Optional[torch.Tensor],
    scores: torch.Tensor,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None

    mask = mask.to(device=scores.device, dtype=torch.bool)

    if scores.ndim == 2:
        query_len, key_len = scores.shape
        if mask.ndim == 1 and mask.shape == (key_len,):
            return mask.unsqueeze(0)
        if mask.ndim == 2 and mask.shape == (query_len, key_len):
            return mask
        raise ValueError(f"Unsupported mask shape {mask.shape} for scores shape {scores.shape}")

    if scores.ndim == 4:
        batch_size, _, query_len, key_len = scores.shape
        if mask.ndim == 2 and mask.shape == (query_len, key_len):
            return mask.unsqueeze(0).unsqueeze(0)
        if mask.ndim == 2 and mask.shape == (batch_size, key_len):
            return mask.unsqueeze(1).unsqueeze(2)
        if mask.ndim == 3 and mask.shape == (batch_size, query_len, key_len):
            return mask.unsqueeze(1)
        if mask.ndim == 4 and mask.shape[-2:] == (query_len, key_len):
            return mask
        raise ValueError(f"Unsupported mask shape {mask.shape} for scores shape {scores.shape}")

    raise ValueError(f"Unsupported scores shape {scores.shape}")


def combine_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    combined = None
    for mask in masks:
        if mask is None:
            continue
        combined = mask if combined is None else (combined & mask)
    return combined


def make_causal_mask(
    query_len: int,
    key_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = past_len + torch.arange(query_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def stable_softmax(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = -1,
) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    scores_max = torch.max(scores, dim=dim, keepdim=True).values
    scores_max = torch.where(torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max))
    exp_scores = torch.exp(scores - scores_max)
    if mask is not None:
        exp_scores = exp_scores.masked_fill(~mask, 0.0)
    exp_sum = exp_scores.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(exp_scores.dtype).tiny)
    return exp_scores / exp_sum


def self_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Core scaled dot-product attention on already projected tensors.

    Supported shapes:
        Q, K, V: (seq_len, head_dim)
        Q, K, V: (batch_size, n_heads, seq_len, head_dim)
    """
    if Q.ndim != K.ndim or Q.ndim != V.ndim:
        raise ValueError(f"Q, K, V must have same rank, got {Q.ndim}, {K.ndim}, {V.ndim}")
    if Q.ndim not in (2, 4):
        raise ValueError(f"Q, K, V must be 2D or 4D, but got {Q.shape}")
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError(f"Q and K must have same last dim, got {Q.shape[-1]} and {K.shape[-1]}")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError(f"K and V must have same sequence length, got {K.shape[-2]} and {V.shape[-2]}")

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    prepared_mask = prepare_attention_mask(mask, scores)
    attention_weights = stable_softmax(scores, mask=prepared_mask, dim=-1)
    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p, training=training)
    return torch.matmul(attention_weights, V)


def multi_head_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n_q_heads: int,
    n_kv_heads: Optional[int] = None,
    W_o: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Educational wrapper around the core attention math with optional GQA and output projection.
    """
    if Q.ndim not in (2, 3) or K.ndim not in (2, 3) or V.ndim not in (2, 3):
        raise ValueError("Q, K, V must be 2D or 3D")
    if Q.ndim != K.ndim or Q.ndim != V.ndim:
        raise ValueError("Q, K, V must have matching ranks")

    if n_kv_heads is None:
        n_kv_heads = n_q_heads
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")

    input_was_2d = Q.ndim == 2
    if input_was_2d:
        Q = Q.unsqueeze(0)
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)

    if Q.shape[:2] != K.shape[:2] or Q.shape[:2] != V.shape[:2]:
        raise ValueError("Q, K, V must share batch size and sequence length")

    batch_size, seq_len, q_dim = Q.shape
    _, _, k_dim = K.shape
    _, _, v_dim = V.shape

    if q_dim % n_q_heads != 0:
        raise ValueError("Query dim must be divisible by n_q_heads")
    head_dim = q_dim // n_q_heads
    if k_dim != n_kv_heads * head_dim or v_dim != n_kv_heads * head_dim:
        raise ValueError("K and V dims must equal n_kv_heads * head_dim")

    q = Q.reshape(batch_size, seq_len, n_q_heads, head_dim).transpose(1, 2)
    k = K.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    v = V.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    if n_q_heads != n_kv_heads:
        n_rep = n_q_heads // n_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

    output = self_attention(q, k, v, mask=mask, dropout_p=dropout_p, training=training)
    output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, q_dim)

    if W_o is not None:
        if W_o.shape != (q_dim, q_dim):
            raise ValueError(f"W_o must have shape ({q_dim}, {q_dim}), but got {W_o.shape}")
        output = output @ W_o

    if input_was_2d:
        output = output.squeeze(0)
    return output


class GroupedQueryAttention(nn.Module):
    """
    Decoder-grade self-attention with GQA, RoPE, dropout, masks, and KV cache.
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_q_heads = config.n_q_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = self.d_model // self.n_q_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads

        kv_dim = self.n_kv_heads * self.head_dim
        q_dim = self.n_q_heads * self.head_dim

        self.q_proj = nn.Linear(self.d_model, q_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, kv_dim, bias=config.bias)
        self.o_proj = nn.Linear(q_dim, self.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

        cos, sin = precompute_rope_frequencies(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _get_rope_slice(self, start: int, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = start + length
        if end > self.rope_cos.shape[0]:
            raise ValueError(f"Requested sequence length {end} exceeds max_seq_len {self.rope_cos.shape[0]}")
        return self.rope_cos[start:end], self.rope_sin[start:end]

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if x.ndim not in (2, 3):
            raise ValueError(f"x must be 2D or 3D, but got shape {x.shape}")

        input_was_2d = x.ndim == 2
        if input_was_2d:
            x = x.unsqueeze(0)

        batch_size, query_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, but got {d_model}")

        q = self.q_proj(x).view(batch_size, query_len, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, query_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, query_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k.shape[:2] != (batch_size, self.n_kv_heads):
                raise ValueError("KV cache batch/head shape mismatch")
            if past_v.shape[:2] != (batch_size, self.n_kv_heads):
                raise ValueError("KV cache batch/head shape mismatch")
            past_len = past_k.shape[2]
        cos, sin = self._get_rope_slice(past_len, query_len)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        k_for_attention = repeat_kv(k, self.n_rep)
        v_for_attention = repeat_kv(v, self.n_rep)
        scores = torch.matmul(q, k_for_attention.transpose(-2, -1))

        key_len = k_for_attention.shape[2]
        causal_mask = None
        if is_causal:
            causal_mask = make_causal_mask(query_len, key_len, past_len, x.device)

        combined_mask = combine_masks(
            prepare_attention_mask(causal_mask, scores) if causal_mask is not None else None,
            prepare_attention_mask(padding_mask, scores) if padding_mask is not None else None,
            prepare_attention_mask(attention_mask, scores) if attention_mask is not None else None,
        )

        attention_weights = stable_softmax(scores / math.sqrt(self.head_dim), mask=combined_mask, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        output = torch.matmul(attention_weights, v_for_attention)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        output = self.out_dropout(self.o_proj(output))

        if input_was_2d:
            output = output.squeeze(0)
        return output, new_cache


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm decoder block:
        x -> RMSNorm -> Attention -> Residual
        x -> RMSNorm -> FFN -> Residual
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFeedForward(
            d_model=config.d_model,
            hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_input = self.attn_norm(x)
        attn_output, new_cache = self.attn(
            attn_input,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
            is_causal=is_causal,
        )
        x = x + attn_output
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


def run_smoke_tests() -> None:
    torch.manual_seed(0)

    # 1. Core attention works on unbatched tensors.
    q = torch.randn(5, 8)
    k = torch.randn(5, 8)
    v = torch.randn(5, 8)
    out = self_attention(q, k, v)
    assert out.shape == (5, 8)

    # 2. Core attention works on batched multi-head tensors.
    q4 = torch.randn(2, 4, 5, 8)
    k4 = torch.randn(2, 4, 5, 8)
    v4 = torch.randn(2, 4, 5, 8)
    out4 = self_attention(q4, k4, v4)
    assert out4.shape == (2, 4, 5, 8)

    # 3. Educational wrapper supports output projection and GQA.
    x = torch.randn(2, 5, 16)
    q_proj = x
    k_proj = x[..., :8]
    v_proj = x[..., :8]
    w_o = torch.eye(16)
    mha_out = multi_head_attention(
        q_proj,
        k_proj,
        v_proj,
        n_q_heads=4,
        n_kv_heads=2,
        W_o=w_o,
    )
    assert mha_out.shape == (2, 5, 16)

    # 4. Mask broadcasting handles padding masks cleanly.
    padding_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    masked_out = multi_head_attention(
        q_proj,
        k_proj,
        v_proj,
        n_q_heads=4,
        n_kv_heads=2,
        mask=padding_mask,
    )
    assert masked_out.shape == (2, 5, 16)
    assert torch.isfinite(masked_out).all()

    # 5. Decoder-grade attention exposes q/k/v/o projection, RoPE, GQA, and caching.
    config = AttentionConfig(
        d_model=16,
        n_q_heads=4,
        n_kv_heads=2,
        dropout=0.0,
        max_seq_len=32,
        ffn_hidden_dim=64,
    )
    attn = GroupedQueryAttention(config)
    seq_input = torch.randn(2, 5, 16)
    attn_out, cache = attn(seq_input, padding_mask=padding_mask, use_cache=True)
    assert attn_out.shape == (2, 5, 16)
    assert cache is not None
    cached_k, cached_v = cache
    assert cached_k.shape == (2, 2, 5, 4)
    assert cached_v.shape == (2, 2, 5, 4)

    # 6. Incremental decoding grows the cache and matches full causal decoding.
    decode_input = torch.randn(1, 4, 16)
    full_out, _ = attn(decode_input, use_cache=False, is_causal=True)
    first_half, cache = attn(decode_input[:, :2], use_cache=True, is_causal=True)
    second_half, cache = attn(
        decode_input[:, 2:],
        kv_cache=cache,
        use_cache=True,
        is_causal=True,
    )
    assert cache is not None
    assert cache[0].shape[2] == 4
    incremental_last = second_half[:, -1]
    full_last = full_out[:, -1]
    assert torch.allclose(incremental_last, full_last, atol=1e-5)
    assert first_half.shape == (1, 2, 16)

    # 7. Pre-norm Transformer block preserves shape and returns cache.
    block = TransformerBlock(config)
    block_out, block_cache = block(seq_input, padding_mask=padding_mask, use_cache=True)
    assert block_out.shape == (2, 5, 16)
    assert block_cache is not None

    print("All industrial attention smoke tests passed.")


if __name__ == "__main__":
    run_smoke_tests()