"""
test_tabkanet.py
----------------
Standalone test script to execute TabKANet independently.
"""

import torch
from tabkanet import TabKANet

def test_tabkanet():
    print("========================================")
    print(" Testing Standalone TabKANet Architecture")
    print("========================================")
    
    # Instantiate TabKANet directly
    model = TabKANet(
        num_features=8,           # 8 agronomic features
        num_classes=49,           # 49 target crop categories
        embed_dim=32,             # KAN token dimension
        num_grids=8,              # RBF grid points per KAN edge
        n_heads=4,                # Transformer attention heads
        n_transformer_layers=2,   # Transformer encoder depth
        ff_dim=64,                # Feedforward hidden dimension
        dropout=0.1
    )
    model.eval()

    # Batch of 4 samples with 8 features each
    x_batch = torch.randn(4, 8)

    with torch.no_grad():
        logits = model(x_batch)
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)

    print(f"Input Features Shape:     {list(x_batch.shape)}")
    print(f"Output Logits Shape:      {list(logits.shape)}")
    print(f"Predicted Crop Classes:   {predictions.tolist()}")
    print(f"Probabilities Sum Check:  {probs.sum(dim=1).numpy()}")
    print("SUCCESS: TabKANet forward pass executed cleanly!")

if __name__ == "__main__":
    test_tabkanet()
