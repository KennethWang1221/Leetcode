import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim: int, base=10000.0):
        super().__init__()
        self.dim = dim
        # We need pairs of dimensions, so dim must be even
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        # 1. Prepare the frequencies (theta)
        # We only need dim/2 frequencies because we process pairs
        # Formula: theta_i = base^(-2i/d)
        exponent = torch.arange(0, dim, 2).float() / dim
        self.theta = 1.0 / (base ** exponent)
        print("self.theta:", self.theta)

    def get_rotation_matrix(self, seq_len):
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len).float()
        
        # Calculate angles: position * theta
        # Outer product to get matrix of angles [seq_len, dim/2]
        # angles = torch.outer(positions, self.theta)
        positions_col = positions.unsqueeze(1)  # [5] → [5, 1]
        theta_row = self.theta.unsqueeze(0)     # [2] → [1, 2]
        angles = positions_col * theta_row      # [5, 1] × [1, 2] → [5, 2]
        # 2. We need to apply this to pairs (x, y) -> (x', y')
        # The rotation formulas are:
        # x' = x * cos(angle) - y * sin(angle)
        # y' = x * sin(angle) + y * cos(angle)
        
        # We repeat angles to match the input shape [seq_len, dim]
        # values: [theta_0, theta_0, theta_1, theta_1, ...]
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        
        return torch.cos(angles), torch.sin(angles)

    def _rotate(self, x, cos, sin):
        """Apply rotation to a single tensor using the RoPE formula.
        
        Formula: x' = x * cos + rotate_pairs(x) * sin
        Where rotate_pairs transforms [x0, x1, x2, x3] into [-x1, x0, -x3, x2]
        """
        # Split x into pairs (evens and odds) to apply rotation
        x_half_1 = x[..., 0::2]  # Evens: [x0, x2, ...] - the "x" coordinates
        x_half_2 = x[..., 1::2]  # Odds:  [x1, x3, ...] - the "y" coordinates
        
        # Construct the "rotated" version for the calculation
        # [-x1, x0, -x3, x2, ...] - this enables vectorized rotation formula
        x_rotated_pairs = torch.stack([-x_half_2, x_half_1], dim=-1).flatten(-2)
        
        # Apply formula: x * cos + x_rotated_pairs * sin
        return (x * cos) + (x_rotated_pairs * sin)

    def forward(self, q, k=None):
        """Apply RoPE to query tensor, and optionally key tensor.
        
        Args:
            q: Query tensor of shape [Batch, Seq_Len, Dim]
            k: Key tensor of shape [Batch, Seq_Len, Dim] (optional)
            
        Returns:
            If k is None: rotated q tensor
            If k is provided: tuple of (rotated_q, rotated_k)
        """
        # q shape: [Batch, Seq_Len, Dim]
        batch, seq_len, dim = q.shape
        
        # Get cos and sin for rotation
        cos, sin = self.get_rotation_matrix(seq_len)
        
        # Add batch dimension to cos/sin for broadcasting
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        
        # Rotate query
        q_rotated = self._rotate(q, cos, sin)
        
        # If key is provided, rotate it too (like Qwen3's apply_rotary_pos_emb)
        if k is None:
            return q_rotated
        else:
            k_rotated = self._rotate(k, cos, sin)
            return q_rotated, k_rotated

# --- CLASS DEMO ---
if __name__ == "__main__":
    dim = 8  # 8D vector (4 pairs) → exponent = [0.0, 0.25, 0.5, 0.75]
    rope = RoPE(dim=dim)
    
    # Print the exponent and theta values
    print("=== RoPE Configuration ===")
    exponent = torch.arange(0, dim, 2).float() / dim
    print(f"exponent = {exponent.tolist()}")
    print(f"theta    = {rope.theta.tolist()}")
    
    # ========================================
    # Demo 1: Using q and k together (like Qwen3)
    # ========================================
    print("\n=== Demo 1: Rotate Q and K together ===")
    
    # Create query and key tensors (in real transformer, these come from projections)
    q = torch.randn(1, 5, dim)  # batch=1, seq_len=5, dim=8
    k = torch.randn(1, 5, dim)
    
    # Method 1: Rotate both at once (new feature, like Qwen3)
    q_rotated, k_rotated = rope(q, k)
    print(f"Input q shape: {q.shape}, Output q_rotated shape: {q_rotated.shape}")
    print(f"Input k shape: {k.shape}, Output k_rotated shape: {k_rotated.shape}")
    
    # Method 2: Rotate separately (backwards compatible)
    q_rotated_v2 = rope(q)  # Only pass q
    k_rotated_v2 = rope(k)  # Only pass k
    
    # Verify both methods give same result
    assert torch.allclose(q_rotated, q_rotated_v2), "q rotation mismatch!"
    assert torch.allclose(k_rotated, k_rotated_v2), "k rotation mismatch!"
    print("Both methods (together vs separate) give identical results!")
    
    # ========================================
    # Demo 2: Relative Position Property (with all 1s)
    # ========================================
    print("\n=== Demo 2: Relative Position Property (all 1s) ===")
    
    # Scenario A: q at pos 0, k at pos 5 (Distance = 5)
    full_seq = torch.ones(1, 10, dim)
    rotated_seq = rope(full_seq)
    
    q_pos0 = rotated_seq[:, 0, :]
    k_pos5 = rotated_seq[:, 5, :]
    score_A = torch.dot(q_pos0.squeeze(), k_pos5.squeeze())
    
    # Scenario B: q at pos 10, k at pos 15 (Distance = 5)
    full_seq_long = torch.ones(1, 20, dim)
    rotated_seq_long = rope(full_seq_long)
    
    q_pos10 = rotated_seq_long[:, 10, :]
    k_pos15 = rotated_seq_long[:, 15, :]
    score_B = torch.dot(q_pos10.squeeze(), k_pos15.squeeze())
    
    print(f"Score at positions (0, 5):   {score_A.item():.4f}")
    print(f"Score at positions (10, 15): {score_B.item():.4f}")
    
    if abs(score_A - score_B) < 1e-4:
        print("SUCCESS: The scores are identical! (same input, same distance)")
    else:
        print("FAIL: Something went wrong.")
    
    # ========================================
    # Demo 3: Relative Position with RANDOM vectors
    # ========================================
    print("\n=== Demo 3: Relative Position Property (random vectors) ===")
    
    # Create a random query vector and a random key vector
    torch.manual_seed(42)  # For reproducibility
    q_vector = torch.randn(1, 1, dim)  # One random query vector
    k_vector = torch.randn(1, 1, dim)  # One random key vector
    
    print(f"q_vector: {q_vector.squeeze().tolist()[:4]}... (showing first 4)")
    print(f"k_vector: {k_vector.squeeze().tolist()[:4]}... (showing first 4)")
    
    # Scenario A: q at position 0, k at position 5 (Distance = 5)
    # We place the SAME q_vector at pos 0 and SAME k_vector at pos 5
    seq_A = torch.zeros(1, 10, dim)
    seq_A[:, 0, :] = q_vector
    seq_A[:, 5, :] = k_vector
    rotated_A = rope(seq_A)
    score_A_rand = torch.dot(rotated_A[:, 0, :].squeeze(), rotated_A[:, 5, :].squeeze())
    
    # Scenario B: q at position 10, k at position 15 (Distance = 5)
    # We place the SAME q_vector at pos 10 and SAME k_vector at pos 15
    seq_B = torch.zeros(1, 20, dim)
    seq_B[:, 10, :] = q_vector
    seq_B[:, 15, :] = k_vector
    rotated_B = rope(seq_B)
    score_B_rand = torch.dot(rotated_B[:, 10, :].squeeze(), rotated_B[:, 15, :].squeeze())
    
    print(f"Score at positions (0, 5):   {score_A_rand.item():.4f}")
    print(f"Score at positions (10, 15): {score_B_rand.item():.4f}")
    
    if abs(score_A_rand - score_B_rand) < 1e-4:
        print("SUCCESS: Random vectors, same distance → same score!")
    else:
        print("FAIL: Something went wrong.")
    
    # ========================================
    # Demo 4: Different distances → Different scores
    # ========================================
    print("\n=== Demo 4: Different distances → Different scores ===")
    
    # Same vectors, but DIFFERENT distances
    seq_C = torch.zeros(1, 10, dim)
    seq_C[:, 0, :] = q_vector
    seq_C[:, 3, :] = k_vector  # Distance = 3 (not 5!)
    rotated_C = rope(seq_C)
    score_dist3 = torch.dot(rotated_C[:, 0, :].squeeze(), rotated_C[:, 3, :].squeeze())
    
    print(f"Score at distance 5: {score_A_rand.item():.4f}")
    print(f"Score at distance 3: {score_dist3.item():.4f}")
    print(f"Difference: {abs(score_A_rand - score_dist3).item():.4f}")
    
    if abs(score_A_rand - score_dist3) > 0.01:
        print("SUCCESS: Different distances → Different scores (as expected!)")