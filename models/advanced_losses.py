"""
Advanced Loss Functions for Pocket-Ligand Embedding Alignment

This module implements state-of-the-art loss functions for cross-modal
embedding alignment, inspired by CLIP, SimCLR, and contrastive learning literature.

Key implementations:
1. InfoNCE Loss - Contrastive learning (CLIP-style)
2. Cosine Similarity Loss - Direction alignment
3. Angular Loss - Direct angle optimization
4. Combined Loss - Multi-objective optimization
5. Supervised Contrastive Loss - With ligand similarity

Author: RNA-3E-FFI Project
Date: 2025-11-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (Contrastive Learning).

    Used in CLIP for image-text alignment. Here adapted for pocket-ligand alignment.

    The loss maximizes agreement between positive pairs while minimizing
    agreement with negative pairs (all other samples in the batch).

    Mathematical formulation:
        sim[i,j] = cosine_similarity(pocket[i], ligand[j]) / temperature
        loss = -log(exp(sim[i,i]) / Σ_j exp(sim[i,j]))

    Args:
        temperature: Temperature parameter for scaling logits.
                    Lower values make the loss harder (sharper distributions).
                    Default: 0.07 (from CLIP)
        learnable_temperature: Whether temperature should be a learnable parameter.

    References:
        - CLIP: https://arxiv.org/abs/2103.00020
        - InfoNCE: https://arxiv.org/abs/1807.03748
    """

    def __init__(self, temperature=0.07, learnable_temperature=False):
        super().__init__()

        if learnable_temperature:
            # Use log(temperature) to ensure temperature > 0
            self.log_temperature = nn.Parameter(
                torch.tensor(np.log(temperature), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'log_temperature',
                torch.tensor(np.log(temperature), dtype=torch.float32)
            )

    @property
    def temperature(self):
        """Current temperature value."""
        return self.log_temperature.exp()

    def forward(self, pocket_embeddings, ligand_embeddings, return_metrics=False):
        """
        Compute InfoNCE loss.

        Args:
            pocket_embeddings: Tensor of shape [batch_size, embedding_dim]
            ligand_embeddings: Tensor of shape [batch_size, embedding_dim]
            return_metrics: If True, return additional metrics for logging

        Returns:
            loss: Scalar tensor
            metrics (optional): Dict with accuracy and temperature
        """
        batch_size = pocket_embeddings.shape[0]

        # Normalize embeddings
        pocket_embeddings = F.normalize(pocket_embeddings, dim=1)
        ligand_embeddings = F.normalize(ligand_embeddings, dim=1)

        # Compute similarity matrix [batch_size, batch_size]
        logits = torch.matmul(pocket_embeddings, ligand_embeddings.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits.device, dtype=torch.long)

        # Compute cross-entropy loss in both directions (symmetric)
        loss_pocket_to_ligand = F.cross_entropy(logits, labels)
        loss_ligand_to_pocket = F.cross_entropy(logits.T, labels)

        # Average of both directions
        loss = (loss_pocket_to_ligand + loss_ligand_to_pocket) / 2

        if return_metrics:
            # Compute accuracy: how often is the correct ligand ranked first?
            with torch.no_grad():
                pred_labels = logits.argmax(dim=1)
                accuracy = (pred_labels == labels).float().mean()

            metrics = {
                'infonce_loss': loss.item(),
                'infonce_accuracy': accuracy.item(),
                'temperature': self.temperature.item(),
                'loss_p2l': loss_pocket_to_ligand.item(),
                'loss_l2p': loss_ligand_to_pocket.item()
            }
            return loss, metrics

        return loss


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss.

    Optimizes the cosine similarity between predicted and target embeddings.
    This is scale-invariant and focuses on direction alignment.

    loss = 1 - mean(cosine_similarity(pred, target))

    Args:
        reduction: 'mean' or 'sum' or 'none'
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, return_metrics=False):
        """
        Compute cosine similarity loss.

        Args:
            pred: Predicted embeddings [batch_size, embedding_dim]
            target: Target embeddings [batch_size, embedding_dim]
            return_metrics: If True, return additional metrics

        Returns:
            loss: Scalar tensor
            metrics (optional): Dict with average cosine similarity
        """
        # Compute cosine similarity for each pair
        cosine_sim = F.cosine_similarity(pred, target, dim=1)

        # Loss: 1 - cosine_similarity (range [0, 2])
        loss = 1 - cosine_sim

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        if return_metrics:
            with torch.no_grad():
                avg_cosine_sim = cosine_sim.mean()

            metrics = {
                'cosine_loss': loss.item(),
                'avg_cosine_similarity': avg_cosine_sim.item()
            }
            return loss, metrics

        return loss


class AngularLoss(nn.Module):
    """
    Angular Loss.

    Directly optimizes the angle between vectors.

    loss = mean(arccos(cosine_similarity(pred, target)))

    This can have better gradient properties than cosine loss near perfect alignment.

    Args:
        reduction: 'mean' or 'sum' or 'none'
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, return_metrics=False):
        """
        Compute angular loss.

        Args:
            pred: Predicted embeddings [batch_size, embedding_dim]
            target: Target embeddings [batch_size, embedding_dim]
            return_metrics: If True, return additional metrics

        Returns:
            loss: Scalar tensor (in radians)
            metrics (optional): Dict with average angle
        """
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(pred, target, dim=1)

        # Clamp to valid range for arccos: [-1, 1]
        # Add small epsilon for numerical stability
        cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute angle in radians [0, π]
        angle = torch.acos(cosine_sim)

        if self.reduction == 'mean':
            loss = angle.mean()
        elif self.reduction == 'sum':
            loss = angle.sum()
        else:
            loss = angle

        if return_metrics:
            with torch.no_grad():
                avg_angle_rad = angle.mean()
                avg_angle_deg = avg_angle_rad * (180.0 / np.pi)

            metrics = {
                'angular_loss': loss.item(),
                'avg_angle_rad': avg_angle_rad.item(),
                'avg_angle_deg': avg_angle_deg.item()
            }
            return loss, metrics

        return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss for Multi-Objective Optimization.

    Combines multiple loss functions with learnable or fixed weights:
        total_loss = α * InfoNCE + β * Cosine + γ * MSE

    This leverages the strengths of different objectives:
    - InfoNCE: Discriminative learning, positive/negative separation
    - Cosine: Direction alignment
    - MSE: Distance preservation

    Args:
        alpha: Weight for InfoNCE loss
        beta: Weight for Cosine loss
        gamma: Weight for MSE loss
        temperature: Temperature for InfoNCE
        learnable_weights: If True, loss weights become learnable parameters
    """

    def __init__(self,
                 alpha=0.5,
                 beta=0.3,
                 gamma=0.2,
                 temperature=0.07,
                 learnable_weights=False):
        super().__init__()

        # Initialize individual losses
        self.infonce = InfoNCELoss(temperature=temperature)
        self.cosine = CosineSimilarityLoss()

        # Loss weights
        if learnable_weights:
            # Use softplus to ensure weights > 0
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
            self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
            self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))

        self.learnable_weights = learnable_weights

    def get_weights(self):
        """Get current loss weights (after softplus if learnable)."""
        if self.learnable_weights:
            return {
                'alpha': F.softplus(self.alpha).item(),
                'beta': F.softplus(self.beta).item(),
                'gamma': F.softplus(self.gamma).item()
            }
        else:
            return {
                'alpha': self.alpha.item(),
                'beta': self.beta.item(),
                'gamma': self.gamma.item()
            }

    def forward(self, pred, target, return_metrics=False):
        """
        Compute combined loss.

        Args:
            pred: Predicted embeddings [batch_size, embedding_dim]
            target: Target embeddings [batch_size, embedding_dim]
            return_metrics: If True, return detailed metrics

        Returns:
            loss: Scalar tensor
            metrics (optional): Dict with individual losses and weights
        """
        # Get weights (apply softplus if learnable)
        if self.learnable_weights:
            alpha = F.softplus(self.alpha)
            beta = F.softplus(self.beta)
            gamma = F.softplus(self.gamma)
        else:
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma

        # Compute individual losses
        infonce_loss = self.infonce(pred, target)
        cosine_loss = self.cosine(pred, target)
        mse_loss = F.mse_loss(pred, target)

        # Combined loss
        total_loss = alpha * infonce_loss + beta * cosine_loss + gamma * mse_loss

        if return_metrics:
            metrics = {
                'total_loss': total_loss.item(),
                'infonce_loss': infonce_loss.item(),
                'cosine_loss': cosine_loss.item(),
                'mse_loss': mse_loss.item(),
                'weight_alpha': alpha.item(),
                'weight_beta': beta.item(),
                'weight_gamma': gamma.item(),
                'temperature': self.infonce.temperature.item()
            }

            # Add cosine similarity
            with torch.no_grad():
                avg_cos_sim = F.cosine_similarity(pred, target, dim=1).mean()
                metrics['avg_cosine_similarity'] = avg_cos_sim.item()

            return total_loss, metrics

        return total_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss with Ligand Similarity.

    Extends InfoNCE to handle cases where multiple ligands in the batch
    are similar to each other. Instead of treating all non-matching pairs
    as negatives, we use ligand similarity to create soft labels.

    Args:
        temperature: Temperature parameter
        similarity_threshold: Ligands with similarity > threshold are treated as positive
    """

    def __init__(self, temperature=0.07, similarity_threshold=0.8):
        super().__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold

    def forward(self, pocket_embeddings, ligand_embeddings,
                ligand_similarity_matrix=None, return_metrics=False):
        """
        Compute supervised contrastive loss.

        Args:
            pocket_embeddings: [batch_size, embedding_dim]
            ligand_embeddings: [batch_size, embedding_dim]
            ligand_similarity_matrix: [batch_size, batch_size] optional
                                     Pairwise ligand similarities
            return_metrics: If True, return additional metrics

        Returns:
            loss: Scalar tensor
            metrics (optional): Dict with statistics
        """
        batch_size = pocket_embeddings.shape[0]

        # Normalize embeddings
        pocket_embeddings = F.normalize(pocket_embeddings, dim=1)
        ligand_embeddings = F.normalize(ligand_embeddings, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(pocket_embeddings, ligand_embeddings.T) / self.temperature

        if ligand_similarity_matrix is None:
            # Fall back to standard InfoNCE
            labels = torch.arange(batch_size, device=logits.device, dtype=torch.long)
            loss = F.cross_entropy(logits, labels)
        else:
            # Use soft labels based on ligand similarity
            # Mask out self-similarity
            mask = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
            ligand_similarity_matrix = ligand_similarity_matrix.clone()
            ligand_similarity_matrix[mask] = 0.0

            # Create soft labels: similarity > threshold are positive
            positive_mask = ligand_similarity_matrix > self.similarity_threshold

            # Compute loss for each pocket
            losses = []
            for i in range(batch_size):
                # Positive pairs for pocket i
                positive_indices = torch.where(positive_mask[i])[0]

                if len(positive_indices) == 0:
                    # No additional positives, use standard cross-entropy
                    label = torch.tensor([i], device=logits.device, dtype=torch.long)
                    loss_i = F.cross_entropy(logits[i:i+1], label)
                else:
                    # Multiple positives: use soft labels
                    # Create target distribution
                    target = torch.zeros(batch_size, device=logits.device)
                    target[i] = 1.0  # Main positive
                    target[positive_indices] = ligand_similarity_matrix[i, positive_indices]
                    target = target / target.sum()  # Normalize

                    # KL divergence loss
                    log_probs = F.log_softmax(logits[i], dim=0)
                    loss_i = -(target * log_probs).sum()

                losses.append(loss_i)

            loss = torch.stack(losses).mean()

        if return_metrics:
            with torch.no_grad():
                pred_labels = logits.argmax(dim=1)
                true_labels = torch.arange(batch_size, device=logits.device)
                accuracy = (pred_labels == true_labels).float().mean()

                num_positives = (ligand_similarity_matrix > self.similarity_threshold).sum() if ligand_similarity_matrix is not None else 0

            metrics = {
                'supcon_loss': loss.item(),
                'supcon_accuracy': accuracy.item(),
                'num_additional_positives': num_positives.item() if isinstance(num_positives, torch.Tensor) else num_positives
            }
            return loss, metrics

        return loss


class MultiStageScheduler:
    """
    Multi-Stage Loss Scheduler.

    Automatically switches between different loss functions during training.

    Stage 1 (Warmup): MSE for stable initialization
    Stage 2 (Contrastive): InfoNCE + Cosine for discriminative learning
    Stage 3 (Fine-tuning): Combined loss for refinement

    Args:
        stage1_epochs: Number of epochs for stage 1 (MSE warmup)
        stage2_epochs: Number of epochs for stage 2 (contrastive learning)
        temperature: Temperature for InfoNCE
    """

    def __init__(self, stage1_epochs=50, stage2_epochs=100, temperature=0.07):
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_start = stage1_epochs + stage2_epochs

        # Create losses
        self.mse_loss = nn.MSELoss()
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.cosine_loss = CosineSimilarityLoss()
        self.combined_loss = CombinedLoss(
            alpha=0.5, beta=0.3, gamma=0.2, temperature=temperature
        )

        self.current_stage = 1

    def get_loss(self, epoch):
        """Get the appropriate loss function for the current epoch."""
        if epoch < self.stage1_epochs:
            self.current_stage = 1
            return self.mse_loss, "MSE (Warmup)"
        elif epoch < self.stage3_start:
            self.current_stage = 2
            return self._stage2_loss, "InfoNCE + Cosine (Contrastive)"
        else:
            self.current_stage = 3
            return self.combined_loss, "Combined (Fine-tuning)"

    def _stage2_loss(self, pred, target, return_metrics=False):
        """Stage 2: InfoNCE + 0.3 * Cosine"""
        infonce = self.infonce_loss(pred, target)
        cosine = self.cosine_loss(pred, target)
        loss = infonce + 0.3 * cosine

        if return_metrics:
            return loss, {
                'infonce_loss': infonce.item(),
                'cosine_loss': cosine.item(),
                'total_loss': loss.item()
            }
        return loss

    def get_recommended_lr(self):
        """Get recommended learning rate for current stage."""
        if self.current_stage == 1:
            return 1e-3
        elif self.current_stage == 2:
            return 5e-4
        else:
            return 1e-4


def compute_retrieval_metrics(pocket_embeddings, ligand_embeddings, top_k=[1, 5, 10]):
    """
    Compute retrieval metrics (Top-K accuracy, MRR).

    For each pocket, retrieve the most similar ligands and check if the
    correct ligand is in the top-K.

    Args:
        pocket_embeddings: [N, D]
        ligand_embeddings: [N, D]
        top_k: List of k values to compute accuracy for

    Returns:
        metrics: Dict with top-k accuracies and MRR
    """
    N = pocket_embeddings.shape[0]

    # Normalize
    pocket_embeddings = F.normalize(pocket_embeddings, dim=1)
    ligand_embeddings = F.normalize(ligand_embeddings, dim=1)

    # Compute similarity matrix [N, N]
    similarity = torch.matmul(pocket_embeddings, ligand_embeddings.T)

    # For each pocket, rank ligands by similarity
    ranks = similarity.argsort(dim=1, descending=True)

    # Ground truth: diagonal (pocket i matches ligand i)
    correct_indices = torch.arange(N, device=ranks.device).unsqueeze(1)

    # Compute metrics
    metrics = {}

    # Top-K accuracy
    for k in top_k:
        top_k_matches = ranks[:, :k]
        hits = (top_k_matches == correct_indices).any(dim=1).float()
        metrics[f'top{k}_accuracy'] = hits.mean().item()

    # Mean Reciprocal Rank (MRR)
    # Find position of correct ligand
    correct_positions = (ranks == correct_indices).nonzero(as_tuple=True)[1]
    reciprocal_ranks = 1.0 / (correct_positions.float() + 1)
    metrics['mrr'] = reciprocal_ranks.mean().item()

    # Average cosine similarity (for correct pairs)
    correct_similarities = similarity.diag()
    metrics['avg_correct_similarity'] = correct_similarities.mean().item()

    return metrics


# Example usage
if __name__ == "__main__":
    print("Testing Advanced Loss Functions...")

    # Create dummy data
    batch_size = 32
    embedding_dim = 256

    pocket_emb = torch.randn(batch_size, embedding_dim)
    ligand_emb = torch.randn(batch_size, embedding_dim)

    # Test InfoNCE
    print("\n1. InfoNCE Loss:")
    infonce = InfoNCELoss(temperature=0.07)
    loss, metrics = infonce(pocket_emb, ligand_emb, return_metrics=True)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Accuracy: {metrics['infonce_accuracy']*100:.2f}%")
    print(f"   Temperature: {metrics['temperature']:.4f}")

    # Test Cosine
    print("\n2. Cosine Similarity Loss:")
    cosine = CosineSimilarityLoss()
    loss, metrics = cosine(pocket_emb, ligand_emb, return_metrics=True)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Avg Cosine Sim: {metrics['avg_cosine_similarity']:.4f}")

    # Test Angular
    print("\n3. Angular Loss:")
    angular = AngularLoss()
    loss, metrics = angular(pocket_emb, ligand_emb, return_metrics=True)
    print(f"   Loss: {loss.item():.4f} rad = {metrics['avg_angle_deg']:.2f}°")

    # Test Combined
    print("\n4. Combined Loss:")
    combined = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)
    loss, metrics = combined(pocket_emb, ligand_emb, return_metrics=True)
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   InfoNCE: {metrics['infonce_loss']:.4f}")
    print(f"   Cosine: {metrics['cosine_loss']:.4f}")
    print(f"   MSE: {metrics['mse_loss']:.4f}")

    # Test retrieval metrics
    print("\n5. Retrieval Metrics:")
    # Make pocket_emb similar to ligand_emb for testing
    pocket_emb_similar = ligand_emb + 0.1 * torch.randn_like(ligand_emb)
    metrics = compute_retrieval_metrics(pocket_emb_similar, ligand_emb)
    print(f"   Top-1 Accuracy: {metrics['top1_accuracy']*100:.2f}%")
    print(f"   Top-5 Accuracy: {metrics['top5_accuracy']*100:.2f}%")
    print(f"   MRR: {metrics['mrr']:.4f}")

    print("\n✅ All tests completed!")
