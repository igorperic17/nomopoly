"""
Auto ZK Training pipeline for any pretrained classifier with adversarial proof verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from .networks import ZKVerifierNet, ZKAdversarialNet


class PretrainedMNISTClassifier(nn.Module):
    """Simple but effective MNIST classifier that we'll pretrain quickly."""
    
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


def train_pretrained_classifier(device='mps', epochs=10):
    """Quickly train a strong MNIST classifier."""
    print("🏃 Training pretrained MNIST classifier...")
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Model
    model = PretrainedMNISTClassifier(784, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        acc = 100. * correct / total
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
    
    print(f"✅ Pretrained classifier ready with {acc:.2f}% accuracy")
    return model


class GeneralZKTraining:
    """ZK training system that works with any pretrained classifier."""
    
    def __init__(
        self,
        pretrained_classifier: nn.Module,
        verifier: ZKVerifierNet,
        adversary: ZKAdversarialNet,
        device: str = "mps",
        learning_rates: Dict[str, float] = None
    ):
        # Use MPS if available, otherwise fallback
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"🚀 Using device: {self.device}")
        
        # Move networks to device
        self.classifier = pretrained_classifier.to(self.device)
        self.verifier = verifier.to(self.device)
        self.adversary = adversary.to(self.device)
        
        # Freeze the pretrained classifier
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        
        # Set up optimizers
        if learning_rates is None:
            learning_rates = {
                "verifier": 1e-3,
                "adversary": 5e-4
            }
            
        self.verifier_optimizer = optim.Adam(self.verifier.parameters(), lr=learning_rates["verifier"])
        self.adversary_optimizer = optim.Adam(self.adversary.parameters(), lr=learning_rates["adversary"])
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        
        # Determine input/output dimensions from the classifier
        self.input_dim = self._get_input_dim()
        self.output_dim = self._get_output_dim()
        
    def _get_input_dim(self):
        """Infer input dimension from classifier."""
        # For MNIST, we'll use 14x14 = 196 (downsampled)
        return 196
        
    def _get_output_dim(self):
        """Infer output dimension from classifier."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 28, 28).to(self.device)
            output = self.classifier(dummy_input)
            return output.shape[1]
        
    def load_mnist_data(self, num_samples: int = 10000) -> DataLoader:
        """Load MNIST data resized to match our system."""
        print("Loading MNIST dataset...")
        
        # Transform to 14x14 for our system
        transform = transforms.Compose([
            transforms.Resize((14, 14)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 196
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        
        if num_samples < len(train_dataset):
            indices = torch.randperm(len(train_dataset))[:num_samples]
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
        
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        print(f"Loaded {len(train_dataset)} MNIST samples")
        return dataloader
        
    def get_classifier_predictions(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Get predictions from pretrained classifier (resize 14x14 -> 28x28)."""
        batch_size = batch_data.shape[0]
        
        # Reshape 14x14 -> 28x28 for pretrained classifier
        resized_data = batch_data.view(batch_size, 1, 14, 14)
        resized_data = F.interpolate(resized_data, size=(28, 28), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            logits = self.classifier(resized_data)
            log_probs = F.log_softmax(logits, dim=1)
            
        return log_probs
        
    def generate_computational_proof(self, batch_data: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """Generate highly distinctive computational proofs using multiple layers of analysis."""
        batch_size = batch_data.size(0)
        
        # Layer 1: Deep input analysis
        # Reshape to 14x14 for spatial analysis
        spatial_data = batch_data.view(batch_size, 14, 14)
        
        # Spatial features
        center_mass_x = torch.sum(spatial_data * torch.arange(14, device=self.device).float().view(1, 14, 1), dim=(1,2)) / (torch.sum(spatial_data, dim=(1,2)) + 1e-8)
        center_mass_y = torch.sum(spatial_data * torch.arange(14, device=self.device).float().view(1, 1, 14), dim=(1,2)) / (torch.sum(spatial_data, dim=(1,2)) + 1e-8)
        
        # Statistical features
        input_mean = torch.mean(batch_data, dim=1)
        input_std = torch.std(batch_data, dim=1)
        input_skew = torch.mean((batch_data - input_mean.unsqueeze(1))**3, dim=1) / (input_std**3 + 1e-8)
        input_kurtosis = torch.mean((batch_data - input_mean.unsqueeze(1))**4, dim=1) / (input_std**4 + 1e-8)
        
        # Gradient features (edge detection)
        grad_x = torch.abs(spatial_data[:, :, 1:] - spatial_data[:, :, :-1])
        grad_y = torch.abs(spatial_data[:, 1:, :] - spatial_data[:, :-1, :])
        edge_density = (torch.sum(grad_x, dim=(1,2)) + torch.sum(grad_y, dim=(1,2))) / 196.0
        
        # Layer 2: Deep output analysis
        probs = torch.exp(log_probs)
        
        # Prediction confidence and uncertainty
        entropy = -torch.sum(probs * log_probs, dim=1)
        max_prob = torch.max(probs, dim=1)[0]
        prediction = torch.argmax(log_probs, dim=1).float()
        
        # Top-k analysis
        top_2_probs = torch.topk(probs, 2, dim=1)[0]
        confidence_gap = top_2_probs[:, 0] - top_2_probs[:, 1]
        
        # Decision boundary analysis
        decision_strength = torch.max(log_probs, dim=1)[0] - torch.mean(log_probs, dim=1)
        
        # Layer 3: Cross-modal correlations (input-output relationships)
        # These are hard for adversary to fake without real computation
        pixel_prediction_corr = torch.sum(batch_data, dim=1) * max_prob
        spatial_prediction_corr = (center_mass_x + center_mass_y) * prediction
        complexity_confidence_corr = edge_density * confidence_gap
        
        # Layer 4: Temporal/Sequential patterns
        # Create sequences from spatial scanning
        horizontal_scan = torch.mean(spatial_data, dim=1)  # (batch, 14)
        vertical_scan = torch.mean(spatial_data, dim=2)    # (batch, 14)
        
        # Pattern detection in scans
        h_diff = torch.abs(horizontal_scan[:, 1:] - horizontal_scan[:, :-1])
        v_diff = torch.abs(vertical_scan[:, 1:] - vertical_scan[:, :-1])
        scan_complexity = torch.sum(h_diff, dim=1) + torch.sum(v_diff, dim=1)
        
        # Combine all features into computational fingerprint
        computational_features = torch.stack([
            # Spatial features
            center_mass_x, center_mass_y, edge_density,
            # Statistical features  
            input_mean, input_std, input_skew, input_kurtosis,
            # Output features
            entropy, max_prob, prediction, confidence_gap, decision_strength,
            # Cross-modal correlations
            pixel_prediction_corr, spatial_prediction_corr, complexity_confidence_corr,
            # Sequential patterns
            scan_complexity
        ], dim=1)  # Shape: (batch_size, 16)
        
        # Layer 5: Multi-scale proof construction
        proof_dim = self.adversary.proof_dim
        
        # Scale 1: Direct feature embedding
        feature_repeat = computational_features.repeat(1, proof_dim // 16 + 1)[:, :proof_dim]
        
        # Scale 2: Nonlinear transformations
        nonlinear_features = torch.tanh(computational_features * 2.0)
        nonlinear_repeat = nonlinear_features.repeat(1, proof_dim // 16 + 1)[:, :proof_dim]
        
        # Scale 3: Interaction terms
        interaction_matrix = torch.outer(computational_features[0], computational_features[0])  # For batch processing
        interaction_features = []
        for i in range(batch_size):
            inter = torch.outer(computational_features[i], computational_features[i])
            interaction_features.append(inter.flatten()[:proof_dim])
        interaction_features = torch.stack(interaction_features)
        
        # Combine scales with different weights
        proof_layer1 = 0.4 * feature_repeat + 0.3 * nonlinear_repeat + 0.3 * interaction_features
        
        # Layer 6: Cryptographic-style transformations
        # Add deterministic but complex transformations
        proof_layer2 = torch.tanh(proof_layer1) + 0.1 * torch.sin(4 * torch.pi * proof_layer1)
        proof_layer3 = torch.sigmoid(proof_layer2 * 1.5) + 0.1 * torch.cos(2 * torch.pi * proof_layer2)
        
        # Layer 7: Computational signature
        # Add signature that depends on exact computation
        signature_positions = (prediction.long() + torch.arange(batch_size, device=self.device)) % proof_dim
        computational_proof = proof_layer3.clone()
        
        for i in range(batch_size):
            pos = signature_positions[i]
            # Strong signature that encodes computation
            computational_proof[i, pos] = 0.95 * torch.sign(center_mass_x[i] - 7.0) + 0.05 * max_prob[i]
            # Secondary signature
            pos2 = (pos + int(prediction[i].item()) + 1) % proof_dim
            computational_proof[i, pos2] = 0.9 * torch.tanh(entropy[i] - 1.0) + 0.1 * confidence_gap[i]
        
        # Final normalization and bounding
        computational_proof = torch.clamp(computational_proof, -1.0, 1.0)
        
        return computational_proof
        
    def train_verifier_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[float, float, float]:
        """Train verifier to distinguish computational proofs from adversarial proofs."""
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch_size = batch_data.size(0)
        
        # Get real predictions from pretrained classifier
        real_log_probs = self.get_classifier_predictions(batch_data)
        real_proofs = self.generate_computational_proof(batch_data, real_log_probs)
        
        # Generate fake proofs from adversary
        with torch.no_grad():
            fake_log_probs, fake_proofs = self.adversary(batch_data)
        
        # Labels for binary classification
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Verifier predictions
        real_predictions = self.verifier(batch_data, real_log_probs, real_proofs)
        fake_predictions = self.verifier(batch_data, fake_log_probs, fake_proofs)
        
        # Binary classification loss
        real_loss = self.bce_loss(real_predictions, real_labels)
        fake_loss = self.bce_loss(fake_predictions, fake_labels)
        total_loss = real_loss + fake_loss
        
        # Update verifier
        self.verifier_optimizer.zero_grad()
        total_loss.backward()
        self.verifier_optimizer.step()
        
        # Calculate accuracies
        real_accuracy = ((real_predictions > 0.5).float() == real_labels).float().mean().item()
        fake_accuracy = ((fake_predictions < 0.5).float()).float().mean().item()
        
        return total_loss.item(), real_accuracy, fake_accuracy
        
    def train_adversary_step(self, batch_data: torch.Tensor) -> Tuple[float, float]:
        """Train adversary to fool verifier."""
        batch_data = batch_data.to(self.device)
        batch_size = batch_data.size(0)
        
        # Generate fake proofs
        fake_log_probs, fake_proofs = self.adversary(batch_data)
        
        # Try to fool verifier (want verifier to output 1)
        fake_labels = torch.ones(batch_size, 1, device=self.device)
        verifier_predictions = self.verifier(batch_data, fake_log_probs, fake_proofs)
        
        adversary_loss = self.bce_loss(verifier_predictions, fake_labels)
        
        # Update adversary
        self.adversary_optimizer.zero_grad()
        adversary_loss.backward()
        self.adversary_optimizer.step()
        
        success_rate = (verifier_predictions > 0.5).float().mean().item()
        
        return adversary_loss.item(), success_rate

    def train(self, num_epochs: int = 30, num_samples: int = 5000) -> Dict[str, List[float]]:
        """Train the ZK verification system."""
        
        # Load data
        dataloader = self.load_mnist_data(num_samples)
        
        print(f"Starting ZK training on {self.device}")
        print(f"Training verifier and adversary for {num_epochs} epochs")
        
        # Training statistics
        stats = {
            "verifier_accuracy_real": [],
            "verifier_accuracy_fake": [],
            "adversary_success_rate": []
        }
        
        # Phase 1: Train verifier without adversary (50% of epochs)
        phase1_epochs = max(5, num_epochs // 2)
        print(f"\n🎯 Phase 1: Verifier Training ({phase1_epochs} epochs)")
        
        for epoch in range(1, phase1_epochs + 1):
            epoch_verifier_real = 0.0
            epoch_verifier_fake = 0.0
            num_batches = 0
            
            for batch_data, batch_labels in tqdm(dataloader, desc=f"Phase 1 - Epoch {epoch}"):
                _, v_real, v_fake = self.train_verifier_step(batch_data, batch_labels)
                epoch_verifier_real += v_real
                epoch_verifier_fake += v_fake
                num_batches += 1
                
            epoch_verifier_real /= num_batches
            epoch_verifier_fake /= num_batches
            
            stats["verifier_accuracy_real"].append(epoch_verifier_real)
            stats["verifier_accuracy_fake"].append(epoch_verifier_fake)
            stats["adversary_success_rate"].append(0.0)  # No adversary yet
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Verifier Real/Fake = {epoch_verifier_real:.4f}/{epoch_verifier_fake:.4f}")
        
        # Phase 2: Adversarial training (remaining epochs)
        phase2_start = phase1_epochs + 1
        print(f"\n⚔️ Phase 2: Adversarial Training ({num_epochs - phase1_epochs} epochs)")
        
        for epoch in range(phase2_start, num_epochs + 1):
            epoch_verifier_real = 0.0
            epoch_verifier_fake = 0.0
            epoch_adversary_success = 0.0
            num_batches = 0
            
            for batch_idx, (batch_data, batch_labels) in enumerate(tqdm(dataloader, desc=f"Phase 2 - Epoch {epoch}")):
                # Train verifier
                _, v_real, v_fake = self.train_verifier_step(batch_data, batch_labels)
                epoch_verifier_real += v_real
                epoch_verifier_fake += v_fake
                
                # Train adversary every other batch
                if batch_idx % 2 == 0:
                    _, a_success = self.train_adversary_step(batch_data)
                    epoch_adversary_success += a_success * 2
                
                num_batches += 1
                
            epoch_verifier_real /= num_batches
            epoch_verifier_fake /= num_batches
            epoch_adversary_success /= num_batches
            
            stats["verifier_accuracy_real"].append(epoch_verifier_real)
            stats["verifier_accuracy_fake"].append(epoch_verifier_fake)
            stats["adversary_success_rate"].append(epoch_adversary_success)
            
            if epoch % 5 == 0:
                binary_acc = (epoch_verifier_real + (1 - epoch_verifier_fake)) / 2
                print(f"  Epoch {epoch}: Verifier Binary Acc = {binary_acc:.4f}, Adversary Success = {epoch_adversary_success:.4f}")
        
        print("\n✅ Training completed!")
        
        # Final analysis
        final_real_acc = stats["verifier_accuracy_real"][-1]
        final_fake_acc = stats["verifier_accuracy_fake"][-1]
        final_binary_acc = (final_real_acc + (1 - final_fake_acc)) / 2
        
        print(f"\n📊 Final Performance:")
        print(f"  Verifier Real Accuracy: {final_real_acc:.4f}")
        print(f"  Verifier Fake Rejection: {(1-final_fake_acc):.4f}")
        print(f"  Verifier Binary Accuracy: {final_binary_acc:.4f}")
        
        if final_binary_acc > 0.8:
            print("🎉 SUCCESS: Verifier learned to distinguish proofs!")
        elif final_binary_acc > 0.6:
            print("⚠️ PARTIAL: Verifier shows some discrimination")
        else:
            print("❌ FAILURE: Verifier cannot distinguish proofs")
        
        return stats
        
    def plot_training_progress(self, stats: Dict[str, List[float]], save_path: str = None):
        """Plot training statistics."""
        os.makedirs("plots", exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Verifier performance
        axes[0].plot(stats["verifier_accuracy_real"], label="Real Accuracy", linewidth=2)
        axes[0].plot(stats["verifier_accuracy_fake"], label="Fake Acceptance", linewidth=2)
        axes[0].set_title("Verifier Performance")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True)
        
        # Binary classification accuracy
        if len(stats["verifier_accuracy_real"]) > 0:
            binary_acc = [(r + (1-f))/2 for r, f in zip(stats["verifier_accuracy_real"], stats["verifier_accuracy_fake"])]
            axes[1].plot(binary_acc, 'g-', linewidth=3, label="Binary Accuracy")
            axes[1].axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
            axes[1].axhline(y=0.8, color='orange', linestyle='--', label='Good Performance')
            axes[1].set_title("Verifier Binary Classification")
            axes[1].set_ylabel("Accuracy")
            axes[1].legend()
            axes[1].grid(True)
        
        # Adversary success
        if any(x > 0 for x in stats["adversary_success_rate"]):
            axes[2].plot(stats["adversary_success_rate"], 'r-', linewidth=2)
            axes[2].set_title("Adversary Success Rate")
            axes[2].set_ylabel("Success Rate")
            axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.close()
