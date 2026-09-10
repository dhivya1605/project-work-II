"""
test_kan.py
-----------
Standalone test script to execute FastKANLayer and NumericalKANEmbedding independently.
"""

import torch
from kan_layer import FastKANLayer, NumericalKANEmbedding

def test_kan():
    print("========================================")
    print(" 1. Testing Single FastKANLayer")
    print("========================================")
    kan_single = FastKANLayer(in_features=4, out_features=16, num_grids=8)
    x_dummy = torch.randn(5, 4)  # 5 batch samples, 4 input features
    out_single = kan_single(x_dummy)

    print(f"Input Shape:        {list(x_dummy.shape)}")
    print(f"Output Shape:       {list(out_single.shape)}")
    print(f"Sample Output Row:  {out_single[0, :4].detach().numpy()} ...")

    print("\n========================================")
    print(" 2. Testing NumericalKANEmbedding")
    print("========================================")
    # Embeds 8 numeric agronomic features independently into 32-dim vectors
    kan_embedding = NumericalKANEmbedding(num_numeric_features=8, embed_dim=32, num_grids=8)
    x_agronomic = torch.randn(5, 8)  # 5 batch samples, 8 agronomic features
    out_embedding = kan_embedding(x_agronomic)

    print(f"Input Shape:        {list(x_agronomic.shape)}")
    print(f"Output Shape:       {list(out_embedding.shape)}")
    print("SUCCESS: KAN layers executed cleanly!")

if __name__ == "__main__":
    test_kan()
