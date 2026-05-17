import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Model configuration
# ============================================================

@dataclass
class TransformerConfig:
    # Vocabulary size.
    vocab_size: int = 32000

    # Hidden size of each token vector.
    d_model: int = 512

    # Number of Transformer blocks.
    n_layers: int = 6

    # Number of query heads.
    n_q_heads: int = 8

    # Number of key/value heads.
    # If this equals n_q_heads, this is standard Multi-Head Attention.
    # If this is smaller than n_q_heads, this is Grouped Query Attention.
    # If this is 1, this is Multi-Query Attention.
    n_kv_heads: int = 2

    # Maximum sequence length supported by RoPE.
    max_seq_len: int = 2048

    # Hidden size inside the feed-forward network.
    ffn_hidden_dim: int = 2048

    # Dropout probability.
    dropout: float = 0.0

    # Small value used inside RMSNorm.
    norm_eps: float = 1e-5

    # RoPE base value.
    rope_theta: float = 10000.0

    # Whether to share token embedding weights with the final LM head.
    tie_embeddings: bool = True


# ============================================================
# 2. RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """
    RMSNorm normalizes each token vector.

    Input shape:
        (batch_size, seq_len, d_model)

    Output shape:
        (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        # Save small value for numerical safety.
        self.eps = eps

        # Learnable scale parameter.
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean square over the feature dimension.
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # Compute inverse root mean square.
        inv_rms = torch.rsqrt(mean_square + self.eps)

        # Normalize x.
        x_norm = x * inv_rms

        # Apply learnable scale.
        output = x_norm * self.weight

        # Return normalized output.
        return output


# ============================================================
# 3. RoPE positional encoding
# ============================================================

def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cosine and sine values.

    Returns:
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
    """

    # RoPE needs pairs of dimensions.
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    # Create dimension indices: 0, 2, 4, ...
    dim_indices = torch.arange(0, head_dim, 2).float()

    # Compute inverse frequencies.
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))

    # Create position indices: 0, 1, 2, ...
    positions = torch.arange(max_seq_len).float()

    # Compute position-frequency matrix.
    freqs = torch.outer(positions, inv_freq)

    # Precompute cosine values.
    cos = torch.cos(freqs)

    # Precompute sine values.
    sin = torch.sin(freqs)

    # Return cosine and sine tables.
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to Q or K.

    Args:
        x:
            Shape (batch_size, n_heads, seq_len, head_dim)

        cos:
            Shape (seq_len, head_dim // 2)

        sin:
            Shape (seq_len, head_dim // 2)

    Returns:
        Tensor with the same shape as x.
    """

    # Split even dimensions.
    x_even = x[..., 0::2]

    # Split odd dimensions.
    x_odd = x[..., 1::2]

    # Add batch and head dimensions to cos.
    cos = cos.unsqueeze(0).unsqueeze(0)

    # Add batch and head dimensions to sin.
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Rotate even dimensions.
    rotated_even = x_even * cos - x_odd * sin

    # Rotate odd dimensions.
    rotated_odd = x_even * sin + x_odd * cos

    # Put even and odd dimensions back next to each other.
    output = torch.stack((rotated_even, rotated_odd), dim=-1)

    # Flatten the last two dimensions back into head_dim.
    output = output.flatten(start_dim=-2)

    # Return RoPE output.
    return output


# ============================================================
# 4. repeat_kv for GQA / MQA
# ============================================================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat K or V heads so their head count matches Q heads.

    Args:
        x:
            Shape (batch_size, n_kv_heads, seq_len, head_dim)

        n_rep:
            Number of times each KV head is shared.

    Returns:
        Shape (batch_size, n_kv_heads * n_rep, seq_len, head_dim)
    """

    # If no repeat is needed, return x directly.
    if n_rep == 1:
        return x

    # Read shape.
    batch_size, n_kv_heads, seq_len, head_dim = x.shape

    # Add a repeat dimension after n_kv_heads.
    x = x[:, :, None, :, :]

    # Expand KV heads without immediately copying data.
    x = x.expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim)

    # Merge n_kv_heads and n_rep into the final head count.
    x = x.reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)

    # Return repeated tensor.
    return x


# ============================================================
# 5. Masks and stable softmax
# ============================================================

def make_causal_mask(
    query_len: int,
    key_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create causal mask.

    True means allowed.
    False means blocked.

    Shape:
        (query_len, key_len)
    """

    # Query positions start after the cached tokens.
    query_positions = past_len + torch.arange(query_len, device=device)

    # Key positions include cached tokens and current tokens.
    key_positions = torch.arange(key_len, device=device)

    # A query token can attend to key tokens at or before its position.
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    # Return causal mask.
    return mask


def build_attention_mask(
    scores: torch.Tensor,
    causal_mask: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Build final attention mask.

    Args:
        scores:
            Shape (batch_size, n_heads, query_len, key_len)

        causal_mask:
            Shape (query_len, key_len)

        padding_mask:
            Optional shape (batch_size, key_len)
            True means real token.
            False means padding token.

    Returns:
        Shape broadcastable to scores:
            (batch_size, 1, query_len, key_len)
    """

    # Read score shape.
    batch_size, _, query_len, key_len = scores.shape

    # Expand causal mask across batch and heads.
    final_mask = causal_mask.view(1, 1, query_len, key_len)

    # If no padding mask is provided, return causal mask only.
    if padding_mask is None:
        return final_mask

    # Check padding mask shape.
    if padding_mask.shape != (batch_size, key_len):
        raise ValueError(
            f"padding_mask must have shape ({batch_size}, {key_len}), "
            f"but got {padding_mask.shape}"
        )

    # Convert padding mask to bool on the correct device.
    padding_mask = padding_mask.to(device=scores.device, dtype=torch.bool)

    # Expand padding mask across heads and query positions.
    padding_mask = padding_mask.view(batch_size, 1, 1, key_len)

    # Combine causal mask and padding mask.
    final_mask = final_mask & padding_mask

    # Return final mask.
    return final_mask


def stable_masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Stable softmax with mask.

    True in mask means keep.
    False in mask means block.
    """

    # Use float32 for safer softmax math.
    original_dtype = scores.dtype

    # Convert scores to float32.
    scores = scores.float()

    # Put -inf on blocked positions.
    scores = scores.masked_fill(~mask, float("-inf"))

    # Find max score in each row.
    scores_max = torch.max(scores, dim=dim, keepdim=True).values

    # If a full row is masked, max becomes -inf; replace it with 0.
    scores_max = torch.where(torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max))

    # Subtract max for numerical safety.
    scores = scores - scores_max

    # Exponentiate scores.
    exp_scores = torch.exp(scores)

    # Force blocked positions to exactly 0.
    exp_scores = exp_scores.masked_fill(~mask, 0.0)

    # Sum probabilities.
    exp_sum = torch.sum(exp_scores, dim=dim, keepdim=True)

    # Avoid divide-by-zero.
    exp_sum = exp_sum.clamp_min(torch.finfo(exp_scores.dtype).tiny)

    # Normalize.
    probs = exp_scores / exp_sum

    # Convert back to original dtype.
    probs = probs.to(dtype=original_dtype)

    # Return probabilities.
    return probs


# ============================================================
# 6. Grouped Query Attention
# ============================================================

class GroupedQueryAttention(nn.Module):
    """
    Decoder self-attention with GQA support.

    Input:
        x: (batch_size, seq_len, d_model)

    Output:
        output: (batch_size, seq_len, d_model)
        new_cache: optional K/V cache
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # Save config.
        self.config = config

        # Model hidden size.
        self.d_model = config.d_model

        # Number of query heads.
        self.n_q_heads = config.n_q_heads

        # Number of key/value heads.
        self.n_kv_heads = config.n_kv_heads

        # Check GQA grouping.
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")

        # Feature size per attention head.
        if self.d_model % self.n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")

        # Compute head dimension.
        self.head_dim = self.d_model // self.n_q_heads

        # Number of times each KV head is shared.
        self.n_rep = self.n_q_heads // self.n_kv_heads

        # Query projection.
        self.q_proj = nn.Linear(
            self.d_model,
            self.n_q_heads * self.head_dim,
            bias=False,
        )

        # Key projection.
        self.k_proj = nn.Linear(
            self.d_model,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )

        # Value projection.
        self.v_proj = nn.Linear(
            self.d_model,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )

        # Output projection.
        self.o_proj = nn.Linear(
            self.n_q_heads * self.head_dim,
            self.d_model,
            bias=False,
        )

        # Dropout applied to attention weights.
        self.attn_dropout = nn.Dropout(config.dropout)

        # Dropout applied after output projection.
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # Read input shape.
        batch_size, seq_len, _ = x.shape

        # Project x to Q.
        Q = self.q_proj(x)

        # Project x to K.
        K = self.k_proj(x)

        # Project x to V.
        V = self.v_proj(x)

        # Reshape Q to separate query heads.
        Q = Q.view(batch_size, seq_len, self.n_q_heads, self.head_dim)

        # Reshape K to separate KV heads.
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Reshape V to separate KV heads.
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Move heads before sequence.
        Q = Q.transpose(1, 2)

        # Move heads before sequence.
        K = K.transpose(1, 2)

        # Move heads before sequence.
        V = V.transpose(1, 2)

        # Apply RoPE to Q.
        Q = apply_rope(Q, cos, sin)

        # Apply RoPE to K.
        K = apply_rope(K, cos, sin)

        # Start with no past tokens.
        past_len = 0

        # If KV cache exists, append new K/V to old K/V.
        if kv_cache is not None:

            # Read cached K and V.
            past_K, past_V = kv_cache

            # Read number of cached tokens.
            past_len = past_K.shape[2]

            # Add new K after cached K.
            K = torch.cat([past_K, K], dim=2)

            # Add new V after cached V.
            V = torch.cat([past_V, V], dim=2)

        # Save new cache if requested.
        new_cache = (K, V) if use_cache else None

        # Repeat K heads to match Q heads.
        K_for_attention = repeat_kv(K, self.n_rep)

        # Repeat V heads to match Q heads.
        V_for_attention = repeat_kv(V, self.n_rep)

        # Compute raw attention scores.
        scores = torch.matmul(Q, K_for_attention.transpose(-2, -1))

        # Scale scores using per-head dimension.
        scores = scores / math.sqrt(self.head_dim)

        # Read total key length.
        key_len = K_for_attention.shape[2]

        # Create causal mask.
        causal_mask = make_causal_mask(
            query_len=seq_len,
            key_len=key_len,
            past_len=past_len,
            device=x.device,
        )

        # Build final mask using causal mask and optional padding mask.
        attention_mask = build_attention_mask(
            scores=scores,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
        )

        # Convert scores to attention weights.
        attention_weights = stable_masked_softmax(
            scores=scores,
            mask=attention_mask,
            dim=-1,
        )

        # Apply attention dropout.
        attention_weights = self.attn_dropout(attention_weights)

        # Mix value vectors.
        output = torch.matmul(attention_weights, V_for_attention)

        # Move sequence before heads.
        output = output.transpose(1, 2)

        # Make memory layout safe.
        output = output.contiguous()

        # Merge all query heads back together.
        output = output.view(batch_size, seq_len, self.n_q_heads * self.head_dim)

        # Apply output projection.
        output = self.o_proj(output)

        # Apply output dropout.
        output = self.out_dropout(output)

        # Return output and optional cache.
        return output, new_cache


# ============================================================
# 7. SwiGLU Feed-Forward Network
# ============================================================

class SwiGLUFeedForward(nn.Module):
    """
    Feed-forward network used in many modern LLMs.

    Input:
        (batch_size, seq_len, d_model)

    Output:
        (batch_size, seq_len, d_model)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # First projection.
        self.w1 = nn.Linear(
            config.d_model,
            config.ffn_hidden_dim,
            bias=False,
        )

        # Gate projection.
        self.w3 = nn.Linear(
            config.d_model,
            config.ffn_hidden_dim,
            bias=False,
        )

        # Output projection.
        self.w2 = nn.Linear(
            config.ffn_hidden_dim,
            config.d_model,
            bias=False,
        )

        # Dropout after FFN output.
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute first branch.
        branch = self.w1(x)

        # Apply SiLU activation.
        branch = F.silu(branch)

        # Compute gate branch.
        gate = self.w3(x)

        # Multiply activated branch by gate.
        hidden = branch * gate

        # Project back to d_model.
        output = self.w2(hidden)

        # Apply dropout.
        output = self.dropout(output)

        # Return output.
        return output


# ============================================================
# 8. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """
    One Transformer decoder block.

    Structure:
        x -> RMSNorm -> Attention -> Residual Add
        x -> RMSNorm -> FFN       -> Residual Add
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # RMSNorm before attention.
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Grouped Query Attention.
        self.attn = GroupedQueryAttention(config)

        # RMSNorm before feed-forward network.
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Feed-forward network.
        self.ffn = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # Normalize input before attention.
        attn_input = self.attn_norm(x)

        # Run attention.
        attn_output, new_cache = self.attn(
            x=attn_input,
            cos=cos,
            sin=sin,
            padding_mask=padding_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        # Add attention output back to residual stream.
        x = x + attn_output

        # Normalize input before FFN.
        ffn_input = self.ffn_norm(x)

        # Run FFN.
        ffn_output = self.ffn(ffn_input)

        # Add FFN output back to residual stream.
        x = x + ffn_output

        # Return block output and optional cache.
        return x, new_cache


# ============================================================
# 9. Full Decoder-Only Transformer
# ============================================================

class DecoderOnlyTransformer(nn.Module):
    """
    Full decoder-only Transformer model.

    Inputs:
        input_ids: (batch_size, seq_len)

    Outputs:
        logits: (batch_size, seq_len, vocab_size)
        loss: optional language modeling loss
        new_caches: optional KV caches
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # Save config.
        self.config = config

        # Token embedding table.
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        # Dropout after token embedding.
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Transformer blocks.
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final RMSNorm.
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Final LM head.
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Optionally share token embedding weight with LM head.
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Compute head dimension.
        head_dim = config.d_model // config.n_q_heads

        # Precompute RoPE values.
        cos, sin = precompute_rope_frequencies(
            head_dim=head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # Save RoPE cosine table.
        self.register_buffer("rope_cos", cos, persistent=False)

        # Save RoPE sine table.
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    ]:

        # If input is shape (seq_len,), add batch dimension.
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        # Read input shape.
        batch_size, seq_len = input_ids.shape

        # Start with no cached tokens.
        past_len = 0

        # If cache exists, read cache length from first layer.
        if kv_caches is not None:
            past_len = kv_caches[0][0].shape[2]

        # Check sequence length against RoPE table.
        if past_len + seq_len > self.config.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")

        # Select RoPE values for current token positions.
        cos = self.rope_cos[past_len: past_len + seq_len].to(device=input_ids.device)

        # Select RoPE values for current token positions.
        sin = self.rope_sin[past_len: past_len + seq_len].to(device=input_ids.device)

        # Convert token IDs to token embeddings.
        x = self.token_embedding(input_ids)

        # Apply embedding dropout.
        x = self.embedding_dropout(x)

        # Prepare new cache list if requested.
        new_caches = [] if use_cache else None

        # If no cache is provided, create empty placeholder list.
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        # Pass through each Transformer block.
        for layer, layer_cache in zip(self.layers, kv_caches):

            # Run one block.
            x, new_layer_cache = layer(
                x=x,
                cos=cos,
                sin=sin,
                padding_mask=padding_mask,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )

            # Save new cache if requested.
            if use_cache:
                new_caches.append(new_layer_cache)

        # Apply final norm.
        x = self.final_norm(x)

        # Compute vocabulary logits.
        logits = self.lm_head(x)

        # Start with no loss.
        loss = None

        # If targets are provided, compute language modeling loss.
        if targets is not None:

            # If targets are shape (seq_len,), add batch dimension.
            if targets.ndim == 1:
                targets = targets.unsqueeze(0)

            # Flatten logits and targets for cross entropy.
            logits_flat = logits.reshape(-1, logits.shape[-1])

            # Flatten targets.
            targets_flat = targets.reshape(-1)

            # Compute next-token prediction loss.
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=-100,
            )

        # Return logits, loss, and optional caches.
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens using KV cache.

        Args:
            input_ids:
                Shape (batch_size, seq_len)

            max_new_tokens:
                Number of new tokens to generate.

            temperature:
                If 0, use greedy decoding.
                If > 0, sample from probability distribution.

            top_k:
                If provided, keep only top-k logits before sampling.

        Returns:
            Generated token IDs.
        """

        # Put model in eval mode.
        self.eval()

        # If input is shape (seq_len,), add batch dimension.
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        # Move input to model device.
        device = next(self.parameters()).device

        # Move token IDs to device.
        input_ids = input_ids.to(device)

        # Generated sequence starts with prompt.
        generated = input_ids

        # Start with empty KV cache.
        kv_caches = None

        # Loop for each new token.
        for step in range(max_new_tokens):

            # First step uses the full prompt.
            if kv_caches is None:
                model_input = generated

            # Later steps use only the newest token.
            else:
                model_input = generated[:, -1:]

            # Run model with cache.
            logits, _, kv_caches = self.forward(
                input_ids=model_input,
                kv_caches=kv_caches,
                use_cache=True,
            )

            # Get logits for the last token.
            next_logits = logits[:, -1, :]

            # Greedy decoding.
            if temperature == 0.0:

                # Pick the highest-logit token.
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            # Sampling decoding.
            else:

                # Apply temperature.
                next_logits = next_logits / temperature

                # If top_k is provided, keep only top_k logits.
                if top_k is not None:

                    # Get top-k logits.
                    top_values, _ = torch.topk(next_logits, k=top_k, dim=-1)

                    # Find the smallest value inside top-k.
                    min_top_value = top_values[:, -1].unsqueeze(-1)

                    # Block logits below top-k threshold.
                    next_logits = torch.where(
                        next_logits < min_top_value,
                        torch.full_like(next_logits, float("-inf")),
                        next_logits,
                    )

                # Convert logits to probabilities.
                probs = torch.softmax(next_logits, dim=-1)

                # Sample one token.
                next_token = torch.multinomial(probs, num_samples=1)

            # Append generated token.
            generated = torch.cat([generated, next_token], dim=1)

        # Return generated sequence.
        return generated


# ============================================================
# 10. Test cases
# ============================================================

def test_forward_shape() -> None:
    # Create small config.
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        ffn_hidden_dim=64,
        dropout=0.0,
    )

    # Create model.
    model = DecoderOnlyTransformer(config)

    # Create fake token IDs.
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    # Run model.
    logits, loss, caches = model(input_ids)

    # Check output shape.
    print("forward logits shape:", logits.shape)

    # Expected: (2, 8, 100)
    assert logits.shape == (2, 8, config.vocab_size)

    # Loss should be None because no targets were given.
    assert loss is None

    # Cache should be None because use_cache=False.
    assert caches is None


def test_training_loss() -> None:
    # Create small config.
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        ffn_hidden_dim=64,
        dropout=0.0,
    )

    # Create model.
    model = DecoderOnlyTransformer(config)

    # Create fake input and target token IDs.
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    # Run model with targets.
    logits, loss, _ = model(input_ids, targets=targets)

    # Print loss.
    print("training loss:", float(loss))

    # Check shape.
    assert logits.shape == (2, 8, config.vocab_size)

    # Check loss exists.
    assert loss is not None


def test_padding_mask() -> None:
    # Create small config.
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        ffn_hidden_dim=64,
        dropout=0.0,
    )

    # Create model.
    model = DecoderOnlyTransformer(config)

    # Create fake token IDs.
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    # True means real token.
    # False means padding token.
    padding_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )

    # Run model with padding mask.
    logits, _, _ = model(
        input_ids=input_ids,
        padding_mask=padding_mask,
    )

    # Print shape.
    print("padding mask logits shape:", logits.shape)

    # Check shape.
    assert logits.shape == (2, 6, config.vocab_size)


def test_kv_cache_matches_full_forward() -> None:
    # Use fixed seed.
    torch.manual_seed(0)

    # Create small config.
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        ffn_hidden_dim=64,
        dropout=0.0,
    )

    # Create model.
    model = DecoderOnlyTransformer(config)

    # Put model in eval mode.
    model.eval()

    # Create fake token IDs.
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    # Run full forward once.
    full_logits, _, _ = model(input_ids)

    # Start empty cache.
    kv_caches = None

    # Store step-by-step logits.
    step_logits_list = []

    # Decode one token at a time.
    for t in range(input_ids.shape[1]):

        # Select one token.
        token = input_ids[:, t:t + 1]

        # Run model with cache.
        step_logits, _, kv_caches = model(
            input_ids=token,
            kv_caches=kv_caches,
            use_cache=True,
        )

        # Save step logits.
        step_logits_list.append(step_logits)

    # Concatenate step logits.
    cached_logits = torch.cat(step_logits_list, dim=1)

    # Compare full forward and cached forward.
    max_diff = (full_logits - cached_logits).abs().max().item()

    # Print difference.
    print("KV cache max difference:", max_diff)

    # They should be very close.
    assert max_diff < 1e-4


def test_generation() -> None:
    # Use fixed seed.
    torch.manual_seed(0)

    # Create small config.
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        ffn_hidden_dim=64,
        dropout=0.0,
    )

    # Create model.
    model = DecoderOnlyTransformer(config)

    # Create prompt.
    input_ids = torch.tensor([[1, 2, 3]])

    # Generate tokens.
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        temperature=0.0,
    )

    # Print generated shape and tokens.
    print("generated shape:", generated.shape)
    print("generated tokens:", generated)

    # Expected length: 3 prompt tokens + 5 new tokens.
    assert generated.shape == (1, 8)


def test_mha_gqa_mqa_modes() -> None:
    # MHA: n_q_heads == n_kv_heads
    config_mha = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_layers=1,
        n_q_heads=4,
        n_kv_heads=4,
        max_seq_len=64,
        ffn_hidden_dim=64,
        dropout=0.0,
    )

    # GQA: n_q_heads > n_kv_heads
    config_gqa = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_layers=1,
        n_q_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        ffn_hidden_dim=64,
        dropout=0.0,
    )

    # MQA: n_kv_heads == 1
    config_mqa = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_layers=1,
        n_q_heads=4,
        n_kv_heads=1,
        max_seq_len=64,
        ffn_hidden_dim=64,
        dropout=0.0,
    )

    # Create token IDs.
    input_ids = torch.randint(0, 100, (2, 5))

    # Test MHA.
    model_mha = DecoderOnlyTransformer(config_mha)
    logits_mha, _, _ = model_mha(input_ids)
    print("MHA logits:", logits_mha.shape)

    # Test GQA.
    model_gqa = DecoderOnlyTransformer(config_gqa)
    logits_gqa, _, _ = model_gqa(input_ids)
    print("GQA logits:", logits_gqa.shape)

    # Test MQA.
    model_mqa = DecoderOnlyTransformer(config_mqa)
    logits_mqa, _, _ = model_mqa(input_ids)
    print("MQA logits:", logits_mqa.shape)

    # Check all shapes.
    assert logits_mha.shape == (2, 5, 100)
    assert logits_gqa.shape == (2, 5, 100)
    assert logits_mqa.shape == (2, 5, 100)


if __name__ == "__main__":
    test_forward_shape()
    test_training_loss()
    test_padding_mask()
    test_kv_cache_matches_full_forward()
    test_generation()
    test_mha_gqa_mqa_modes()

    print("All tests passed.")