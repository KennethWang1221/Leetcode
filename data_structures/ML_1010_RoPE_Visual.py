import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
def visualize_rope_frequencies():
    # Setup
    dim = 64            # Total dimensions
    seq_len = 100       # How many positions to look at
    base = 10000.0      # The standard RoPE base
    
    # 1. Calculate Frequencies (Thetas)
    # Formula: theta_i = base^(-2i/d)
    # We take indices 0, 2, 4... up to dim
    indices = torch.arange(0, dim, 2).float()
    theta = 1.0 / (base ** (indices / dim))
    
    # 2. Calculate Position Embeddings for specific dimensions
    # We will track 3 specific "pairs" of dimensions to show the speed difference
    # Pair 0 (Indices 0,1): The "Seconds hand" (Fastest)
    # Pair 16 (Indices 32,33): The "Minutes hand" (Medium)
    # Pair 31 (Indices 62,63): The "Hour hand" (Slowest)
    
    positions = torch.arange(seq_len).float()
    
    # Calculate the angle for each position: position * theta
    # We only care about the Cosine component for visualization
    # y = cos(position * freq)
    
    y_fast   = torch.cos(positions * theta[0]).numpy()
    y_medium = torch.cos(positions * theta[16]).numpy()
    y_slow   = torch.cos(positions * theta[31]).numpy()
    
        # 3. Plottingimport torch


    plt.figure(figsize=(12, 6))
    
    plt.plot(y_fast, label=f'Dimension 0 (High Freq / "Seconds")', alpha=0.8, linewidth=2)
    plt.plot(y_medium, label=f'Dimension 32 (Med Freq / "Minutes")', alpha=0.8, linewidth=2)
    plt.plot(y_slow, label=f'Dimension 62 (Low Freq / "Hours")', alpha=0.8, linewidth=2)
    
    plt.title("RoPE: Why we need different speeds", fontsize=16)
    plt.xlabel("Position (Token Index)", fontsize=12)
    plt.ylabel("Value (Cosine Projection)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Highlight the concept
    plt.annotate('Completes many cycles\n(Good for local detail)', 
                 xy=(10, 1), xytext=(10, 1.5),
                 arrowprops=dict(facecolor='blue', shrink=0.05))
                 
    plt.annotate('Barely moves\n(Good for long distance)', 
                 xy=(90, 0.9), xytext=(60, 0.5),
                 arrowprops=dict(facecolor='green', shrink=0.05))

    # Save the plot
    plt.tight_layout()
    plt.savefig('rope_frequencies.png')
    print("Plot saved as 'rope_frequencies.png'")

if __name__ == "__main__":
    visualize_rope_frequencies()