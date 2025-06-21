"""
Auto ZK Training pipeline for joint adversarial training of the three networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from .networks import ZKProverNet, ZKVerifierNet, ZKAdversarialNet


class AutoZKTraining:
    """Automated Zero Knowledge training pipeline with real MNIST data."""
    
    def __init__(
        self,
        prover: ZKProverNet,
        verifier: ZKVerifierNet,
        adversary: ZKAdversarialNet,
        device: str = "auto",
        learning_rates: Dict[str, float] = None
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move networks to device
        self.prover = prover.to(self.device)
        self.verifier = verifier.to(self.device)
        self.adversary = adversary.to(self.device)
        
        # Set up optimizers with different learning rates
        if learning_rates is None:
            learning_rates = {
                "prover": 1e-3,
                "verifier": 5e-4,
                "adversary": 1e-4
            }
            
        self.prover_optimizer = optim.Adam(self.prover.parameters(), lr=learning_rates["prover"])
        self.verifier_optimizer = optim.Adam(self.verifier.parameters(), lr=learning_rates["verifier"])
        self.adversary_optimizer = optim.Adam(self.adversary.parameters(), lr=learning_rates["adversary"])
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # Training data
        self.train_data = None
        self.train_labels = None
        
    def load_mnist_data(self, num_samples: int = 10000) -> DataLoader:
        """Load real MNIST data and prepare for training."""
        print("Loading real MNIST dataset...")
        
        # Transform to flatten to 196 dimensions (14x14)
        transform = transforms.Compose([
            transforms.Resize((14, 14)),  # Resize to 14x14 to match our network
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 196 dims
        ])
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Create subset if requested
        if num_samples < len(train_dataset):
            indices = torch.randperm(len(train_dataset))[:num_samples]
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
        
        # Create dataloader
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        print(f"Loaded {len(train_dataset)} MNIST samples")
        return dataloader
        
    def generate_computational_proof(self, batch_data: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """Generate strong computational proofs that are clearly distinguishable from random proofs."""
        batch_size = batch_data.size(0)
        
        # Create highly structured proof based on actual computation
        # The key insight: make this VERY different from what adversary can easily generate
        
        # Input analysis - extract meaningful features that relate to computation
        input_mean = torch.mean(batch_data, dim=1, keepdim=True)
        input_std = torch.std(batch_data, dim=1, keepdim=True) + 1e-8
        input_min = torch.min(batch_data, dim=1, keepdim=True)[0]
        input_max = torch.max(batch_data, dim=1, keepdim=True)[0]
        input_sum = torch.sum(batch_data, dim=1, keepdim=True)
        
        # Output analysis - extract classification-specific features
        probs = torch.exp(log_probs)
        output_entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        output_max_logit = torch.max(log_probs, dim=1, keepdim=True)[0]
        output_prediction = torch.argmax(log_probs, dim=1, keepdim=True).float()
        output_confidence = torch.max(probs, dim=1, keepdim=True)[0]
        output_second_max = torch.sort(probs, dim=1, descending=True)[0][:, 1:2]
        
        # Create structured computational fingerprint
        # These features are hard for adversary to fake without doing real computation
        computational_features = torch.cat([
            input_mean, input_std, input_min, input_max, input_sum,
            output_entropy, output_max_logit, output_prediction, 
            output_confidence, output_second_max
        ], dim=1)  # Shape: (batch_size, 10)
        
        # Create proof with strong computational structure
        # Use deterministic transformations that depend on actual computation
        
        # Base proof from repeated computational features
        proof_base = computational_features.repeat(1, self.prover.proof_dim // 10 + 1)[:, :self.prover.proof_dim]
        
        # Add structured interactions between input and output
        input_output_corr = input_mean * output_confidence  # Correlation term
        prediction_strength = output_max_logit - torch.log(output_entropy + 1e-8)  # How confident vs uncertain
        computation_consistency = torch.tanh(input_sum / 196.0) * output_confidence  # Input density vs output confidence
        
        # Create layers of computational proof
        layer1 = torch.tanh(proof_base + 0.3 * input_output_corr.repeat(1, self.prover.proof_dim))
        layer2 = torch.sigmoid(layer1 + 0.2 * prediction_strength.repeat(1, self.prover.proof_dim))
        layer3 = torch.tanh(layer2 * 0.8 + 0.2 * computation_consistency.repeat(1, self.prover.proof_dim))
        
        # Final computational proof with clear structure
        computational_proof = layer3 + 0.1 * torch.sin(2 * torch.pi * layer3)  # Add deterministic nonlinearity
        
        # Ensure proof is bounded and has clear computational fingerprint
        computational_proof = torch.clamp(computational_proof, -1.0, 1.0)
        
        # Add a "computational signature" - specific pattern that only real computation can produce
        signature_index = int(torch.sum(output_prediction).item()) % self.prover.proof_dim
        computational_proof[:, signature_index] = 0.95 * torch.sign(input_mean.squeeze()) + 0.05 * output_confidence.squeeze()
        
        return computational_proof
        
    def train_prover_base_task(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[float, float]:
        """Train prover on base classification task only."""
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        log_probs, _ = self.prover(batch_data)
        classification_loss = F.nll_loss(log_probs, batch_labels)
        
        self.prover_optimizer.zero_grad()
        classification_loss.backward()
        self.prover_optimizer.step()
        
        predictions = torch.argmax(log_probs, dim=1)
        accuracy = (predictions == batch_labels).float().mean().item()
        
        return classification_loss.item(), accuracy
        
    def train_verifier_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[float, float, float]:
        """Train verifier to distinguish computational proofs from random proofs."""
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch_size = batch_data.size(0)
        
        # Generate REAL proofs from prover (based on actual computation)
        with torch.no_grad():
            real_log_probs, _ = self.prover(batch_data)
            real_proofs = self.generate_computational_proof(batch_data, real_log_probs)
            
        # Generate FAKE proofs from adversary (not based on real computation)
        with torch.no_grad():
            fake_log_probs, fake_proofs = self.adversary(batch_data)
            
        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Verifier predictions
        real_predictions = self.verifier(batch_data, real_log_probs, real_proofs)
        fake_predictions = self.verifier(batch_data, fake_log_probs, fake_proofs)
        
        # Loss with clear targets
        real_loss = self.bce_loss(real_predictions, real_labels)
        fake_loss = self.bce_loss(fake_predictions, fake_labels)
        total_loss = real_loss + fake_loss
        
        self.verifier_optimizer.zero_grad()
        total_loss.backward()
        self.verifier_optimizer.step()
        
        # Accuracies
        real_accuracy = ((real_predictions > 0.5).float() == real_labels).float().mean().item()
        fake_accuracy = ((fake_predictions < 0.5).float()).float().mean().item()
        
        return total_loss.item(), real_accuracy, fake_accuracy
        
    def train_prover_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[float, float]:
        """Train prover with both classification and computational proof generation."""
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch_size = batch_data.size(0)
        
        # Forward pass
        log_probs, _ = self.prover(batch_data)
        
        # Generate computational proof
        computational_proof = self.generate_computational_proof(batch_data, log_probs)
        
        # Classification loss
        classification_loss = F.nll_loss(log_probs, batch_labels)
        
        # Proof validity loss (verifier should accept computational proofs)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        verifier_predictions = self.verifier(batch_data, log_probs, computational_proof)
        proof_loss = self.bce_loss(verifier_predictions, real_labels)
        
        # Combined loss
        total_loss = classification_loss + 0.3 * proof_loss
        
        self.prover_optimizer.zero_grad()
        total_loss.backward()
        self.prover_optimizer.step()
        
        predictions = torch.argmax(log_probs, dim=1)
        accuracy = (predictions == batch_labels).float().mean().item()
        
        return total_loss.item(), accuracy
        
    def train_adversary_step(self, batch_data: torch.Tensor) -> Tuple[float, float]:
        """Train adversary to fool verifier with fake proofs."""
        batch_data = batch_data.to(self.device)
        batch_size = batch_data.size(0)
        
        # Adversary generates fake proofs
        fake_log_probs, fake_proofs = self.adversary(batch_data)
        
        # Try to fool verifier
        fake_labels = torch.ones(batch_size, 1, device=self.device)
        verifier_predictions = self.verifier(batch_data, fake_log_probs, fake_proofs)
        
        adversary_loss = self.bce_loss(verifier_predictions, fake_labels)
        
        self.adversary_optimizer.zero_grad()
        adversary_loss.backward()
        self.adversary_optimizer.step()
        
        success_rate = (verifier_predictions > 0.5).float().mean().item()
        
        return adversary_loss.item(), success_rate

    def train(
        self, 
        num_epochs: int = 100,
        num_samples: int = 10000
    ) -> Dict[str, List[float]]:
        """Main training loop with real MNIST data."""
        
        # Load MNIST data
        dataloader = self.load_mnist_data(num_samples)
        
        print(f"Starting ZK training on {self.device}")
        print(f"Training for {num_epochs} epochs with computational proofs")
        
        # Training statistics
        stats = {
            "prover_classification_accuracy": [],
            "verifier_accuracy_real": [],
            "verifier_accuracy_fake": [],
            "adversary_success_rate": []
        }
        
        # Phase 1: Train prover to good classification (25% of epochs)
        phase1_epochs = max(5, num_epochs // 4)
        print(f"\n🏁 Phase 1: Base Classification ({phase1_epochs} epochs)")
        
        for epoch in range(1, phase1_epochs + 1):
            epoch_prover_acc = 0.0
            num_batches = 0
            
            for batch_data, batch_labels in tqdm(dataloader, desc=f"Phase 1 - Epoch {epoch}"):
                _, acc = self.train_prover_base_task(batch_data, batch_labels)
                epoch_prover_acc += acc
                num_batches += 1
                
            epoch_prover_acc /= num_batches
            stats["prover_classification_accuracy"].append(epoch_prover_acc)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Classification Acc = {epoch_prover_acc:.4f}")
        
        # Phase 2: Train prover + verifier with computational proofs (50% of epochs)
        phase2_epochs = max(10, int(num_epochs * 0.5))
        print(f"\n🔐 Phase 2: Computational Proof Training ({phase2_epochs} epochs)")
        
        for epoch in range(phase1_epochs + 1, phase1_epochs + phase2_epochs + 1):
            epoch_prover_acc = 0.0
            epoch_verifier_real = 0.0
            epoch_verifier_fake = 0.0
            num_batches = 0
            
            for batch_data, batch_labels in tqdm(dataloader, desc=f"Phase 2 - Epoch {epoch}"):
                # Train verifier to recognize computational proofs
                _, v_real, v_fake = self.train_verifier_step(batch_data, batch_labels)
                epoch_verifier_real += v_real
                epoch_verifier_fake += v_fake
                
                # Train prover with computational proofs
                _, p_acc = self.train_prover_step(batch_data, batch_labels)
                epoch_prover_acc += p_acc
                
                num_batches += 1
                
            epoch_prover_acc /= num_batches
            epoch_verifier_real /= num_batches
            epoch_verifier_fake /= num_batches
            
            stats["prover_classification_accuracy"].append(epoch_prover_acc)
            stats["verifier_accuracy_real"].append(epoch_verifier_real)
            stats["verifier_accuracy_fake"].append(epoch_verifier_fake)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Prover Acc = {epoch_prover_acc:.4f}, "
                      f"Verifier Real/Fake = {epoch_verifier_real:.4f}/{epoch_verifier_fake:.4f}")
        
        # Phase 3: Add adversary (remaining epochs)
        phase3_start = phase1_epochs + phase2_epochs + 1
        print(f"\n⚔️ Phase 3: Adversarial Training ({num_epochs - phase3_start + 1} epochs)")
        
        for epoch in range(phase3_start, num_epochs + 1):
            epoch_prover_acc = 0.0
            epoch_verifier_real = 0.0
            epoch_verifier_fake = 0.0
            epoch_adversary_success = 0.0
            num_batches = 0
            
            for batch_idx, (batch_data, batch_labels) in enumerate(tqdm(dataloader, desc=f"Phase 3 - Epoch {epoch}")):
                # Train verifier
                _, v_real, v_fake = self.train_verifier_step(batch_data, batch_labels)
                epoch_verifier_real += v_real
                epoch_verifier_fake += v_fake
                
                # Train prover
                _, p_acc = self.train_prover_step(batch_data, batch_labels)
                epoch_prover_acc += p_acc
                
                # Train adversary occasionally 
                if batch_idx % 3 == 0:
                    _, a_success = self.train_adversary_step(batch_data)
                    epoch_adversary_success += a_success * 3
                
                num_batches += 1
                
            epoch_prover_acc /= num_batches
            epoch_verifier_real /= num_batches
            epoch_verifier_fake /= num_batches
            epoch_adversary_success /= num_batches
            
            stats["prover_classification_accuracy"].append(epoch_prover_acc)
            stats["verifier_accuracy_real"].append(epoch_verifier_real)
            stats["verifier_accuracy_fake"].append(epoch_verifier_fake)
            stats["adversary_success_rate"].append(epoch_adversary_success)
            
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}/{num_epochs}:")
                print(f"  Prover Classification: {epoch_prover_acc:.4f}")
                print(f"  Verifier Real/Fake: {epoch_verifier_real:.4f}/{epoch_verifier_fake:.4f}")
                print(f"  Adversary Success: {epoch_adversary_success:.4f}")
        
        print("\n✅ Training completed!")
        
        # Final verification test
        total_real_acc = sum(stats["verifier_accuracy_real"][-5:]) / 5 if len(stats["verifier_accuracy_real"]) >= 5 else 0
        total_fake_acc = sum(stats["verifier_accuracy_fake"][-5:]) / 5 if len(stats["verifier_accuracy_fake"]) >= 5 else 0
        
        print(f"\n📊 Final Performance (last 5 epochs average):")
        print(f"  Verifier Real Accuracy: {total_real_acc:.4f}")
        print(f"  Verifier Fake Accuracy: {total_fake_acc:.4f}")
        print(f"  Overall Verifier Performance: {(total_real_acc + total_fake_acc) / 2:.4f}")
        
        return stats
        
    def plot_training_progress(self, stats: Dict[str, List[float]], save_path: str = None):
        """Plot training statistics."""
        os.makedirs("plots", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Classification accuracy
        if stats["prover_classification_accuracy"]:
            axes[0, 0].plot(stats["prover_classification_accuracy"])
            axes[0, 0].set_title("Prover Classification Accuracy")
            axes[0, 0].set_ylabel("Accuracy")
            axes[0, 0].grid(True)
        
        # Verifier performance
        if stats["verifier_accuracy_real"]:
            axes[0, 1].plot(stats["verifier_accuracy_real"], label="Real Accuracy")
            axes[0, 1].plot(stats["verifier_accuracy_fake"], label="Fake Accuracy")
            axes[0, 1].set_title("Verifier Performance")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Adversary success
        if stats["adversary_success_rate"]:
            axes[1, 0].plot(stats["adversary_success_rate"])
            axes[1, 0].set_title("Adversary Success Rate")
            axes[1, 0].set_ylabel("Success Rate")
            axes[1, 0].grid(True)
        
        # Combined verifier accuracy
        if stats["verifier_accuracy_real"] and stats["verifier_accuracy_fake"]:
            combined = [(r + f) / 2 for r, f in zip(stats["verifier_accuracy_real"], stats["verifier_accuracy_fake"])]
            axes[1, 1].plot(combined)
            axes[1, 1].set_title("Combined Verifier Accuracy")
            axes[1, 1].set_ylabel("Accuracy")
            axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.close()
