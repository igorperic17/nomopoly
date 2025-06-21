"""
ZK Training pipeline focusing on binary proof verification.

Key principles:
- Prover network is FROZEN (original network + HRR, never trained)  
- Verifier learns binary classification: real proofs vs fake proofs
- Adversary tries to generate fake proofs that fool the verifier
- Success = Verifier achieves high binary classification accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil

from .networks import ZKProverNet, ZKVerifierNet, ZKAdversarialNet


class ZKTrainer:
    """
    ZK trainer focusing on binary proof verification.
    
    Key principles:
    - Prover (inference) network is FROZEN - never trained
    - Verifier learns binary classification: real vs fake proofs
    - Adversary generates fake proofs to fool verifier
    - Success = High verifier binary classification accuracy
    """
    
    def __init__(
        self,
        inference_net: ZKProverNet,  # FROZEN - generates authentic proofs
        verifier_net: ZKVerifierNet,  # Learns binary classification
        malicious_net: ZKAdversarialNet,  # Generates fake proofs
        device: str = "mps",
        plots_dir: str = "plots"
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
        self.inference_net = inference_net.to(self.device)
        self.verifier_net = verifier_net.to(self.device)
        self.malicious_net = malicious_net.to(self.device)
        
        # FREEZE the inference network - it should never be trained
        for param in self.inference_net.parameters():
            param.requires_grad = False
        print("🔒 Inference network FROZEN - will never be trained")
        
        self.plots_dir = plots_dir
        
        # Only create optimizers for networks that will be trained
        self.verifier_optimizer = None  # Set after verifier layers are built
        self.malicious_optimizer = None  # Set after malicious layers are built
        
        self.criterion = nn.BCELoss()
        
        # Network dimensions
        self.input_dim = self.inference_net.input_dim
        self.output_dim = self.inference_net.output_dim
        self.proof_dim = self.inference_net.proof_dim
        
    def setup_plots_directory(self) -> str:
        """Create plots directory and clear it if it exists."""
        if os.path.exists(self.plots_dir):
            shutil.rmtree(self.plots_dir)
            print(f"🧹 Cleared existing {self.plots_dir}/ directory")
        os.makedirs(self.plots_dir)
        print(f"📁 Created {self.plots_dir}/ directory for plots")
        return self.plots_dir
        
    def load_mnist_data(self, num_samples: int = 10000) -> Callable:
        """Load MNIST data and return a generator function."""
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
        
        # Create iterator
        data_iter = iter(dataloader)
        
        def data_generator():
            nonlocal data_iter
            try:
                batch = next(data_iter)
                return (batch[0].to(self.device),)  # Return only inputs, move to device
            except StopIteration:
                # Reset iterator when exhausted
                data_iter = iter(dataloader)
                batch = next(data_iter)
                return (batch[0].to(self.device),)
        
        print(f"Loaded {len(train_dataset)} MNIST samples")
        return data_generator
        
    def train_step(self, data_generator: Callable, epoch: int) -> Dict[str, float]:
        """
        Perform one training step focusing on binary proof classification.
        
        Args:
            data_generator: Function that generates training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        # Generate training data
        training_data = data_generator()
        
        # Initialize optimizers for dynamic networks if not already done
        if self.verifier_optimizer is None:
            if epoch == 0:
                print("🔧 Building verifier network layers...")
            # Run forward pass to build dynamic layers
            with torch.no_grad():  # Inference network is frozen
                real_result, real_proof = self.inference_net(*training_data)
            _ = self.verifier_net(real_proof, real_result)  # This builds the layers
            self.verifier_optimizer = optim.Adam(self.verifier_net.parameters(), lr=0.002)  # Higher LR for verifier
            if epoch == 0:
                print("✅ Verifier optimizer created")
            
        if self.malicious_optimizer is None:
            if epoch == 0:
                print("🔧 Building malicious network layers...")
            # Build malicious network layers if needed
            fake_result, fake_proof = self.malicious_net(*training_data)
            self.malicious_optimizer = optim.Adam(self.malicious_net.parameters(), lr=0.0005)  # Lower LR for adversary
            if epoch == 0:
                print("✅ Malicious optimizer created")
        
        # === STEP 1: Train Verifier for Binary Classification ===
        self.verifier_optimizer.zero_grad()
        
        # Generate real proofs (from frozen inference network)
        with torch.no_grad():  # Inference network is frozen
            real_result, real_proof = self.inference_net(*training_data)
        
        # Generate fake proofs (from trainable malicious network)
        fake_result, fake_proof = self.malicious_net(*training_data)
        
        # Verifier should output 1 for real proofs, 0 for fake proofs
        real_verification = self.verifier_net(real_proof, real_result)
        fake_verification = self.verifier_net(fake_proof, fake_result)
        
        # Binary classification loss
        verifier_loss_real = self.criterion(real_verification, torch.ones_like(real_verification))
        verifier_loss_fake = self.criterion(fake_verification, torch.zeros_like(fake_verification))
        verifier_loss = verifier_loss_real + verifier_loss_fake
        
        verifier_loss.backward()
        self.verifier_optimizer.step()
        
        # === STEP 2: Train Malicious Network to fool the verifier ===
        self.malicious_optimizer.zero_grad()
        
        # Generate new fake proofs for adversarial training
        fake_result, fake_proof = self.malicious_net(*training_data)
        fake_verification = self.verifier_net(fake_proof, fake_result)
        
        # Malicious network wants verifier to output 1 (fooled)
        malicious_loss = self.criterion(fake_verification, torch.ones_like(fake_verification))
        malicious_loss.backward()
        self.malicious_optimizer.step()
        
        # Calculate training metrics
        with torch.no_grad():
            # Binary classification accuracy
            real_correct = (real_verification > 0.5).float().mean().item()
            fake_correct = (fake_verification < 0.5).float().mean().item()  
            binary_accuracy = (real_correct + fake_correct) / 2
            
            # Adversarial metrics
            malicious_success = (fake_verification > 0.5).float().mean().item()
            
            # Score separation
            real_score_mean = real_verification.mean().item()
            fake_score_mean = fake_verification.mean().item()
            score_separation = real_score_mean - fake_score_mean
        
        return {
            'verifier_loss': verifier_loss.item(),
            'malicious_loss': malicious_loss.item(),
            'binary_accuracy': binary_accuracy,
            'real_correct': real_correct,
            'fake_correct': fake_correct,
            'malicious_success': malicious_success,
            'real_score_mean': real_score_mean,
            'fake_score_mean': fake_score_mean,
            'score_separation': score_separation
        }

    def train(self, num_epochs: int = 100, num_samples: int = 5000) -> Dict[str, List[float]]:
        """Train the ZK system focusing on binary proof verification."""
        
        # Setup
        self.setup_plots_directory()
        data_generator = self.load_mnist_data(num_samples)
        
        print(f"Starting ZK binary proof verification training on {self.device}")
        print("🎯 Training Goals:")
        print("   Inference Network: FROZEN - generates authentic proofs")
        print("   Verifier Network: Learn binary classification (real vs fake proofs)")
        print("   Malicious Network: Generate fake proofs to fool verifier")
        print("   SUCCESS = High verifier binary classification accuracy (>85%)")
        
        # Training statistics
        stats = {
            "verifier_loss": [],
            "malicious_loss": [],
            "binary_accuracy": [],
            "real_correct": [],
            "fake_correct": [],
            "malicious_success": [],
            "real_score_mean": [],
            "fake_score_mean": [],
            "score_separation": []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Set networks to training mode (except frozen inference)
            self.inference_net.eval()  # Always in eval mode since frozen
            self.verifier_net.train()
            self.malicious_net.train()
            
            # Single training step
            step_stats = self.train_step(data_generator, epoch)
            
            # Accumulate statistics
            for key, value in step_stats.items():
                stats[key].append(value)
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}:")
                print(f"  🎯 Binary Accuracy: {step_stats['binary_accuracy']:.1%}")
                print(f"  ✅ Real Detection: {step_stats['real_correct']:.1%}")
                print(f"  ❌ Fake Rejection: {step_stats['fake_correct']:.1%}")
                print(f"  🎭 Malicious Success: {step_stats['malicious_success']:.1%}")
                print(f"  📊 Score Gap: {step_stats['score_separation']:.3f}")
        
        print("\n✅ Binary proof verification training completed!")
        
        # Final analysis
        final_binary_accuracy = stats["binary_accuracy"][-1]
        final_real_correct = stats["real_correct"][-1]
        final_fake_correct = stats["fake_correct"][-1]
        final_malicious_success = stats["malicious_success"][-1]
        final_score_separation = stats["score_separation"][-1]
        
        print(f"\n📈 FINAL PERFORMANCE:")
        print(f"   🎯 Binary Classification Accuracy: {final_binary_accuracy:.1%}")
        print(f"   ✅ Real Proof Detection: {final_real_correct:.1%}")
        print(f"   ❌ Fake Proof Rejection: {final_fake_correct:.1%}")
        print(f"   🎭 Malicious Success Rate: {final_malicious_success:.1%}")
        print(f"   📊 Score Separation: {final_score_separation:.3f}")
        
        # Evaluate training success
        if final_binary_accuracy > 0.90:
            print("🎉 EXCELLENT: Verifier achieved outstanding binary classification!")
            success_level = "EXCELLENT"
        elif final_binary_accuracy > 0.85:
            print("✅ VERY GOOD: Strong binary classification performance")
            success_level = "VERY_GOOD"
        elif final_binary_accuracy > 0.75:
            print("✅ GOOD: Decent binary classification performance")
            success_level = "GOOD"
        elif final_binary_accuracy > 0.65:
            print("⚠️ MODERATE: Some learning occurred")
            success_level = "MODERATE"
        else:
            print("❌ POOR: Binary classification failed")
            success_level = "FAILED"
            
        # Check adversarial balance
        if final_score_separation > 0.3:
            print("✅ HEALTHY: Good separation between real and fake proof scores")
        elif final_score_separation > 0.1:
            print("⚠️ WEAK: Some separation between proof scores")
        else:
            print("❌ NO SEPARATION: Verifier cannot distinguish proof types")
        
        return stats
        
    def plot_training_progress(self, stats: Dict[str, List[float]], save_path: str = None):
        """Plot training statistics focusing on binary classification performance."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        axes[0, 0].plot(stats["verifier_loss"], label="Verifier Loss", linewidth=2, color='blue')
        axes[0, 0].plot(stats["malicious_loss"], label="Malicious Loss", linewidth=2, color='red')
        axes[0, 0].set_title("Training Losses")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Binary classification accuracy (main metric)
        axes[0, 1].plot(stats["binary_accuracy"], linewidth=3, color='green', label="Binary Accuracy")
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', label='Random Chance')
        axes[0, 1].axhline(y=0.85, color='orange', linestyle='--', label='Target: 85%')
        axes[0, 1].axhline(y=0.95, color='purple', linestyle='--', label='Excellent: 95%')
        axes[0, 1].set_title("Binary Classification Accuracy (MAIN METRIC)")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Binary components breakdown
        axes[0, 2].plot(stats["real_correct"], label="Real Detection", linewidth=2, color='blue')
        axes[0, 2].plot(stats["fake_correct"], label="Fake Rejection", linewidth=2, color='red')
        axes[0, 2].axhline(y=0.85, color='green', linestyle='--', label='Target: 85%')
        axes[0, 2].set_title("Binary Classification Components")
        axes[0, 2].set_ylabel("Accuracy")
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Malicious success rate (should be balanced, not too high or low)
        axes[1, 0].plot(stats["malicious_success"], linewidth=3, color='red', label="Malicious Success")
        axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', label='Balanced')
        axes[1, 0].axhline(y=0.3, color='green', linestyle='--', label='Good Defense')
        axes[1, 0].axhline(y=0.7, color='orange', linestyle='--', label='Strong Attack')
        axes[1, 0].set_title("Malicious Success Rate")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Score separation
        axes[1, 1].plot(stats["real_score_mean"], label="Real Score Mean", linewidth=2, color='blue')
        axes[1, 1].plot(stats["fake_score_mean"], label="Fake Score Mean", linewidth=2, color='red')
        axes[1, 1].set_title("Verifier Score Means")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Score gap (key indicator of learning)
        axes[1, 2].plot(stats["score_separation"], linewidth=3, color='purple', label="Score Separation")
        axes[1, 2].axhline(y=0, color='gray', linestyle='--', label='No Separation')
        axes[1, 2].axhline(y=0.3, color='green', linestyle='--', label='Good Separation')
        axes[1, 2].axhline(y=0.5, color='orange', linestyle='--', label='Excellent Separation')
        axes[1, 2].set_title("Score Separation (Real - Fake)")
        axes[1, 2].set_ylabel("Separation")
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.close()
