"""
RMSNorm (Root Mean Square Normalization) - Educational Implementation

RMSNorm is a simplified version of LayerNorm, used in modern LLMs like:
- LLaMA
- Qwen
- Mistral
- GPT-NeoX

It's faster than LayerNorm because it skips the mean-centering step!
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Math Formula:
    =============
    
    Given input x = [x₁, x₂, ..., xₙ]:
    
    1. Compute RMS (Root Mean Square):
       RMS(x) = √(1/n × Σᵢ xᵢ²)
       
    2. Normalize:
       x_norm = x / RMS(x)
       
    3. Scale (learnable):
       output = x_norm × γ  (where γ is learnable weight)
    
    Or in one formula:
       output = (x / √(mean(x²) + ε)) × γ
    
    Key Differences from LayerNorm:
    ==============================
    - LayerNorm: centers (subtracts mean) then normalizes by std
    - RMSNorm: only normalizes by RMS (no mean subtraction)
    - RMSNorm is ~10-15% faster!
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: The dimension of the input (typically hidden_size)
            eps: Small constant to prevent division by zero
        """
        super().__init__()
        self.eps = eps
        self.dim = dim
        
        # Learnable scale parameter (γ), initialized to 1
        # Shape: [dim]
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.
        
        Args:
            x: Input tensor of shape [..., dim]
               Typically [batch, seq_len, dim]
        
        Returns:
            Normalized tensor of same shape
        """
        # Step 1: Compute x² for each element
        x_squared = x.pow(2)
        
        # Step 2: Compute mean of x² along last dimension
        # keepdim=True to maintain shape for broadcasting
        mean_x_squared = x_squared.mean(dim=-1, keepdim=True)
        
        # Step 3: Compute RMS = √(mean(x²) + ε)
        rms = torch.sqrt(mean_x_squared + self.eps)
        
        # Step 4: Normalize: x / RMS
        x_normalized = x / rms
        
        # Step 5: Scale by learnable weight γ
        output = x_normalized * self.weight
        
        return output
    
    def forward_compact(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compact version (same as above, just one line).
        This is how production code often looks.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (for comparison)
    
    Math Formula:
    =============
    
    Given input x = [x₁, x₂, ..., xₙ]:
    
    1. Compute mean:
       μ = 1/n × Σᵢ xᵢ
       
    2. Compute variance:
       σ² = 1/n × Σᵢ (xᵢ - μ)²
       
    3. Normalize:
       x_norm = (x - μ) / √(σ² + ε)
       
    4. Scale and shift (learnable):
       output = x_norm × γ + β
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # γ (scale)
        self.bias = nn.Parameter(torch.zeros(dim))    # β (shift)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Compute mean
        mean = x.mean(dim=-1, keepdim=True)
        
        # Step 2: Compute variance
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        
        # Step 3: Normalize (center and scale)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        
        # Step 4: Scale and shift
        return x_normalized * self.weight + self.bias


# ========================================
# CLASS DEMO
# ========================================
if __name__ == "__main__":
    print("=" * 60)
    print("RMSNorm Educational Demo")
    print("=" * 60)
    
    # ----------------------------------------
    # Demo 1: Basic RMSNorm Operation
    # ----------------------------------------
    print("\n=== Demo 1: Step-by-step RMSNorm ===")
    
    dim = 4
    rms_norm = RMSNorm(dim=dim)
    
    # Create a simple input
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # Shape: [1, 4]
    print(f"Input x: {x.tolist()}")
    
    # Manual calculation step by step:
    print("\n--- Manual Calculation ---")
    
    # Step 1: x²
    x_squared = x ** 2
    print(f"Step 1 - x²: {x_squared.tolist()}")
    # [1, 4, 9, 16]
    
    # Step 2: mean(x²)
    mean_x_squared = x_squared.mean()
    print(f"Step 2 - mean(x²): {mean_x_squared.item():.4f}")
    # (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    
    # Step 3: RMS = √(mean(x²))
    rms = torch.sqrt(mean_x_squared)
    print(f"Step 3 - RMS = √mean(x²): {rms.item():.4f}")
    # √7.5 ≈ 2.7386
    
    # Step 4: Normalize
    x_normalized = x / rms
    print(f"Step 4 - x / RMS: {x_normalized.tolist()}")
    # [1/2.74, 2/2.74, 3/2.74, 4/2.74] ≈ [0.365, 0.730, 1.095, 1.461]
    
    # Step 5: Scale by weight (initially all 1s, so no change)
    print(f"Step 5 - weights: {rms_norm.weight.tolist()}")
    
    # Using our module
    output = rms_norm(x)
    print(f"\nRMSNorm output: {output.tolist()}")
    
    # ----------------------------------------
    # Demo 2: Compare RMSNorm vs LayerNorm
    # ----------------------------------------
    print("\n=== Demo 2: RMSNorm vs LayerNorm ===")
    
    layer_norm = LayerNorm(dim=dim)
    
    # Same input
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    
    rms_output = rms_norm(x)
    ln_output = layer_norm(x)
    
    print(f"Input:           {x.tolist()}")
    print(f"RMSNorm output:  {rms_output.tolist()}")
    print(f"LayerNorm output:{ln_output.tolist()}")
    
    print("\n--- Key Difference ---")
    print("LayerNorm subtracts mean (centers around 0)")
    print(f"  LayerNorm output mean: {ln_output.mean().item():.6f} (≈ 0)")
    print("RMSNorm does NOT center")
    print(f"  RMSNorm output mean:   {rms_output.mean().item():.6f} (≠ 0)")
    
    # ----------------------------------------
    # Demo 3: Why Normalization?
    # ----------------------------------------
    print("\n=== Demo 3: Why Normalization Helps ===")
    
    # Large values can cause numerical issues
    x_large = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
    x_small = torch.tensor([[0.001, 0.002, 0.003, 0.004]])
    
    print("Without normalization:")
    print(f"  Large input:  {x_large.tolist()}")
    print(f"  Small input:  {x_small.tolist()}")
    print(f"  Ratio: {(x_large / x_small).mean().item():.0f}x difference!")
    
    print("\nWith RMSNorm (values become similar scale):")
    print(f"  Large normalized: {rms_norm(x_large).tolist()}")
    print(f"  Small normalized: {rms_norm(x_small).tolist()}")
    print("  → Both now have similar magnitudes!")
    
    # ----------------------------------------
    # Demo 4: Batch Processing
    # ----------------------------------------
    print("\n=== Demo 4: Batch Processing (like in Transformer) ===")
    
    # Typical transformer input: [batch, seq_len, hidden_dim]
    batch_size = 2
    seq_len = 3
    hidden_dim = 8
    
    rms_norm_big = RMSNorm(dim=hidden_dim)
    
    # Random input simulating transformer hidden states
    torch.manual_seed(42)
    x_batch = torch.randn(batch_size, seq_len, hidden_dim)
    
    print(f"Input shape: {x_batch.shape}")
    print(f"Input (batch 0, position 0): {x_batch[0, 0, :].tolist()[:4]}...")
    
    output_batch = rms_norm_big(x_batch)
    print(f"Output shape: {output_batch.shape}")
    print(f"Output (batch 0, position 0): {output_batch[0, 0, :].tolist()[:4]}...")
    
    # Verify RMS ≈ 1 after normalization (before weight scaling)
    # Note: with weight=1, the RMS of output should be ≈ 1
    rms_of_output = torch.sqrt((output_batch ** 2).mean(dim=-1))
    print(f"\nRMS of each position (should be ≈ 1.0):")
    print(f"  Batch 0: {rms_of_output[0].tolist()}")
    
    # ----------------------------------------
    # Demo 5: Speed Comparison
    # ----------------------------------------
    print("\n=== Demo 5: Speed Comparison ===")
    
    import time
    
    dim = 4096  # Typical LLM hidden size
    batch, seq = 32, 512
    
    rms = RMSNorm(dim)
    ln = LayerNorm(dim)
    x = torch.randn(batch, seq, dim)
    
    # Warmup
    for _ in range(10):
        _ = rms(x)
        _ = ln(x)
    
    # Time RMSNorm
    start = time.time()
    for _ in range(100):
        _ = rms(x)
    rms_time = time.time() - start
    
    # Time LayerNorm
    start = time.time()
    for _ in range(100):
        _ = ln(x)
    ln_time = time.time() - start
    
    print(f"Input shape: [{batch}, {seq}, {dim}]")
    print(f"RMSNorm time (100 runs): {rms_time:.4f}s")
    print(f"LayerNorm time (100 runs): {ln_time:.4f}s")
    print(f"RMSNorm is {ln_time/rms_time:.2f}x faster!")
    
    # ----------------------------------------
    # Summary
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    RMSNorm Formula:
    ────────────────
    output = (x / √(mean(x²) + ε)) × γ
    
    vs LayerNorm Formula:
    ────────────────────
    output = ((x - μ) / √(σ² + ε)) × γ + β
    
    Key Differences:
    ────────────────
    ┌──────────────┬─────────────┬─────────────┐
    │ Feature      │ RMSNorm     │ LayerNorm   │
    ├──────────────┼─────────────┼─────────────┤
    │ Center (μ)   │ ❌ No       │ ✅ Yes      │
    │ Scale (γ)    │ ✅ Yes      │ ✅ Yes      │
    │ Shift (β)    │ ❌ No       │ ✅ Yes      │
    │ Parameters   │ dim         │ 2 × dim     │
    │ Speed        │ Faster      │ Slower      │
    └──────────────┴─────────────┴─────────────┘
    
    Why Modern LLMs Use RMSNorm:
    ───────────────────────────
    1. Simpler (fewer operations)
    2. Faster (~10-15% speedup)
    3. Works just as well in practice!
    """)

