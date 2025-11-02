#!/usr/bin/env python3
"""
Debug script to test cosine similarity loss and gradient flow.
"""
import torch
import torch.nn.functional as F
import numpy as np

def compute_loss(pred, target, loss_fn='cosine'):
    """Test version of compute_loss function."""
    if loss_fn == 'cosine':
        cosine_sim = F.cosine_similarity(pred, target, dim=1)
        loss = (1 - cosine_sim).mean()
        return loss, cosine_sim.mean().item()
    elif loss_fn == 'mse':
        loss = F.mse_loss(pred, target)
        return loss, None

def test_loss_computation():
    """Test if loss computation and gradients are working correctly."""
    print("="*60)
    print("Testing Cosine Similarity Loss")
    print("="*60)

    # Create dummy data
    batch_size = 4
    embedding_dim = 1536

    print(f"\nBatch size: {batch_size}")
    print(f"Embedding dim: {embedding_dim}")

    # Test 1: Random embeddings (should have loss ~1.0)
    print("\n" + "-"*60)
    print("Test 1: Random embeddings (expected loss ~1.0)")
    print("-"*60)

    pred = torch.randn(batch_size, embedding_dim, requires_grad=True)
    target = torch.randn(batch_size, embedding_dim)

    print(f"Pred norm: {pred.norm(dim=1).mean().item():.4f}")
    print(f"Target norm: {target.norm(dim=1).mean().item():.4f}")

    loss, cosine_sim = compute_loss(pred, target, loss_fn='cosine')
    print(f"Loss: {loss.item():.6f}")
    print(f"Cosine Sim: {cosine_sim:.6f}")

    # Check gradients
    loss.backward()
    if pred.grad is not None:
        grad_norm = pred.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.6f}")
        print(f"Gradient mean: {pred.grad.mean().item():.6f}")
        print(f"Gradient std: {pred.grad.std().item():.6f}")
        print(f"Gradient max: {pred.grad.max().item():.6f}")
        print(f"Gradient min: {pred.grad.min().item():.6f}")

        if grad_norm < 1e-8:
            print("⚠️  WARNING: Gradient is too small (vanishing gradient)")
        elif grad_norm > 1e3:
            print("⚠️  WARNING: Gradient is too large (exploding gradient)")
        else:
            print("✓ Gradient norm is reasonable")
    else:
        print("❌ ERROR: No gradient computed!")

    # Test 2: Identical embeddings (should have loss ~0.0)
    print("\n" + "-"*60)
    print("Test 2: Identical embeddings (expected loss ~0.0)")
    print("-"*60)

    pred = torch.randn(batch_size, embedding_dim, requires_grad=True)
    target = pred.detach().clone()

    loss, cosine_sim = compute_loss(pred, target, loss_fn='cosine')
    print(f"Loss: {loss.item():.6f}")
    print(f"Cosine Sim: {cosine_sim:.6f}")

    if abs(loss.item()) < 1e-6:
        print("✓ Loss is near zero as expected")
    else:
        print(f"❌ ERROR: Loss should be near 0, got {loss.item():.6f}")

    # Test 3: Opposite embeddings (should have loss ~2.0)
    print("\n" + "-"*60)
    print("Test 3: Opposite embeddings (expected loss ~2.0)")
    print("-"*60)

    pred = torch.randn(batch_size, embedding_dim, requires_grad=True)
    target = -pred.detach().clone()

    loss, cosine_sim = compute_loss(pred, target, loss_fn='cosine')
    print(f"Loss: {loss.item():.6f}")
    print(f"Cosine Sim: {cosine_sim:.6f}")

    if abs(loss.item() - 2.0) < 0.1:
        print("✓ Loss is near 2.0 as expected")
    else:
        print(f"⚠️  Loss should be near 2.0, got {loss.item():.6f}")

    # Test 4: Simulated training step (with FIXED input)
    print("\n" + "-"*60)
    print("Test 4: Simulated training (100 steps, fixed input)")
    print("-"*60)

    # Create a simple linear model
    model = torch.nn.Linear(embedding_dim, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed input and target
    x = torch.randn(batch_size, embedding_dim)
    target = torch.randn(batch_size, embedding_dim)

    print(f"Initial model output norm: {model(x).norm(dim=1).mean().item():.4f}")
    print(f"Target norm: {target.norm(dim=1).mean().item():.4f}")

    losses = []
    cosine_sims = []
    for step in range(100):
        # Forward
        pred = model(x)
        loss, cosine_sim = compute_loss(pred, target, loss_fn='cosine')

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradient norm
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        optimizer.step()

        losses.append(loss.item())
        cosine_sims.append(cosine_sim)

        if step == 0:
            print(f"Step {step}: Loss={loss.item():.6f}, Cosine={cosine_sim:.6f}, Grad Norm={total_grad_norm:.6f}")
        elif step in [9, 49, 99]:
            print(f"Step {step}: Loss={loss.item():.6f}, Cosine={cosine_sim:.6f}, Grad Norm={total_grad_norm:.6f}")

    if losses[-1] < losses[0] * 0.9:
        print("✓ Loss decreased significantly during training")
        print(f"   Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    else:
        print("❌ ERROR: Loss did not decrease sufficiently!")
        print(f"   Initial loss: {losses[0]:.6f}")
        print(f"   Final loss: {losses[-1]:.6f}")
        print(f"   Change: {(losses[-1] - losses[0]):.6f}")

    # Test 5: Check for NaN or Inf
    print("\n" + "-"*60)
    print("Test 5: Check for NaN/Inf in extreme cases")
    print("-"*60)

    # Very small embeddings
    pred = torch.randn(batch_size, embedding_dim) * 1e-6
    pred.requires_grad = True
    target = torch.randn(batch_size, embedding_dim) * 1e-6

    loss, cosine_sim = compute_loss(pred, target, loss_fn='cosine')
    print(f"Small embeddings - Loss: {loss.item():.6f}, Cosine: {cosine_sim:.6f}")

    if torch.isnan(loss) or torch.isinf(loss):
        print("❌ ERROR: Loss is NaN or Inf with small embeddings!")
    else:
        print("✓ No NaN/Inf with small embeddings")

    # Very large embeddings
    pred = torch.randn(batch_size, embedding_dim) * 1e6
    pred.requires_grad = True
    target = torch.randn(batch_size, embedding_dim) * 1e6

    loss, cosine_sim = compute_loss(pred, target, loss_fn='cosine')
    print(f"Large embeddings - Loss: {loss.item():.6f}, Cosine: {cosine_sim:.6f}")

    if torch.isnan(loss) or torch.isinf(loss):
        print("❌ ERROR: Loss is NaN or Inf with large embeddings!")
    else:
        print("✓ No NaN/Inf with large embeddings")

    # Test 6: Zero embeddings (edge case)
    print("\n" + "-"*60)
    print("Test 6: Zero embeddings (edge case)")
    print("-"*60)

    pred = torch.zeros(batch_size, embedding_dim, requires_grad=True)
    target = torch.randn(batch_size, embedding_dim)

    try:
        loss, cosine_sim = compute_loss(pred, target, loss_fn='cosine')
        print(f"Zero pred - Loss: {loss.item():.6f}, Cosine: {cosine_sim:.6f}")

        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ ERROR: Loss is NaN or Inf with zero embeddings!")
        else:
            print("⚠️  Zero embeddings produce valid loss (cosine_similarity returns NaN for zero vectors)")
    except Exception as e:
        print(f"❌ ERROR with zero embeddings: {e}")

    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    test_loss_computation()
