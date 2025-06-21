"""
Auto ZK Training pipeline for joint adversarial training of the three networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime

from .networks import ZKProverNet, ZKVerifierNet, ZKAdversarialNet


class AutoZKTraining:
    """Automated Zero Knowledge training pipeline."""
    
    def __init__(
        self,
        prover: ZKProverNet,
        verifier: ZKVerifierNet, 
        adversary: ZKAdversarialNet,
        device: str = "auto",
        learning_rates: Dict[str, float] = None,
        log_dir: str = "./logs"
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move networks to device
        self.prover = prover.to(self.device)
        self.verifier = verifier.to(self.device)
        self.adversary = adversary.to(self.device)
        
        # Set up optimizers
        if learning_rates is None:
            learning_rates = {
                "prover": 1e-4,
                "verifier": 2e-4, 
                "adversary": 1e-4
            }
            
        self.prover_optimizer = optim.Adam(self.prover.parameters(), lr=learning_rates["prover"])
        self.verifier_optimizer = optim.Adam(self.verifier.parameters(), lr=learning_rates["verifier"])
        self.adversary_optimizer = optim.Adam(self.adversary.parameters(), lr=learning_rates["adversary"])
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # Training statistics
        self.training_stats = {
            "prover_losses": [],
            "verifier_losses": [],
            "adversary_losses": [],
            "verifier_accuracy_real": [],
            "verifier_accuracy_fake": [],
            "adversary_success_rate": []
        }
        
        # Set up logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"zkml_training_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)
        
    def generate_training_data(
        self, 
        num_samples: int = 10000,
        input_range: Tuple[float, float] = (0.0, 1.0)
    ) -> DataLoader:
        """Generate training data for the MNIST classification."""
        # Generate random inputs (flattened image data)
        inputs = torch.FloatTensor(num_samples, self.prover.input_dim).uniform_(*input_range)
        
        # Create dataset and dataloader
        dataset = TensorDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        return dataloader
        
    def train_verifier_step(
        self, 
        batch: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Train the verifier to distinguish real proofs from fake ones."""
        batch = batch.to(self.device)
        batch_size = batch.size(0)
        
        # Generate real proofs from prover
        with torch.no_grad():
            real_outputs, real_proofs = self.prover(batch)
            
        # Generate fake proofs from adversary
        with torch.no_grad():
            fake_outputs, fake_proofs = self.adversary(batch)
            
        # Create labels (1 for real, 0 for fake)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Forward pass through verifier
        real_predictions = self.verifier(batch, real_outputs, real_proofs)
        fake_predictions = self.verifier(batch, fake_outputs, fake_proofs)
        
        # Compute loss
        real_loss = self.bce_loss(real_predictions, real_labels)
        fake_loss = self.bce_loss(fake_predictions, fake_labels)
        total_loss = (real_loss + fake_loss) / 2
        
        # Backward pass
        self.verifier_optimizer.zero_grad()
        total_loss.backward()
        self.verifier_optimizer.step()
        
        # Compute accuracies
        real_accuracy = ((real_predictions > 0.5).float() == real_labels).float().mean().item()
        fake_accuracy = ((fake_predictions < 0.5).float() == (1 - fake_labels)).float().mean().item()
        
        return total_loss.item(), real_accuracy, fake_accuracy
        
    def train_prover_step(self, batch: torch.Tensor) -> float:
        """Train the prover to generate proofs that fool the verifier."""
        batch = batch.to(self.device)
        batch_size = batch.size(0)
        
        # Generate outputs and proofs
        log_probs, proofs = self.prover(batch)
        
        # Generate synthetic classification labels (random for training)
        correct_labels = torch.randint(0, 10, (batch_size,), device=self.device)
        
        # Verifier should classify these as real (label = 1)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        verifier_predictions = self.verifier(batch, log_probs, proofs)
        
        # Loss components:
        # 1. Classification accuracy (negative log-likelihood)
        classification_loss = F.nll_loss(log_probs, correct_labels)
        
        # 2. Proof validity (verifier should accept the proof)
        proof_loss = self.bce_loss(verifier_predictions, real_labels)
        
        # Combined loss
        total_loss = classification_loss + proof_loss
        
        # Backward pass
        self.prover_optimizer.zero_grad()
        total_loss.backward()
        self.prover_optimizer.step()
        
        return total_loss.item()
        
    def train_adversary_step(self, batch: torch.Tensor) -> Tuple[float, float]:
        """Train the adversary to generate fake proofs that fool the verifier."""
        batch = batch.to(self.device)
        batch_size = batch.size(0)
        
        # Generate fake outputs and proofs
        fake_outputs, fake_proofs = self.adversary(batch)
        
        # We want the verifier to classify these as real (fool the verifier)
        fake_labels = torch.ones(batch_size, 1, device=self.device)
        verifier_predictions = self.verifier(batch, fake_outputs, fake_proofs)
        
        # Loss: fool the verifier
        adversary_loss = self.bce_loss(verifier_predictions, fake_labels)
        
        # Backward pass
        self.adversary_optimizer.zero_grad()
        adversary_loss.backward()
        self.adversary_optimizer.step()
        
        # Success rate (how often adversary fools verifier)
        success_rate = (verifier_predictions > 0.5).float().mean().item()
        
        return adversary_loss.item(), success_rate
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train all networks for one epoch."""
        epoch_stats = {
            "prover_loss": 0.0,
            "verifier_loss": 0.0,
            "adversary_loss": 0.0,
            "verifier_acc_real": 0.0,
            "verifier_acc_fake": 0.0,
            "adversary_success": 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, (batch,) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            
            # Train verifier every step
            v_loss, v_acc_real, v_acc_fake = self.train_verifier_step(batch)
            epoch_stats["verifier_loss"] += v_loss
            epoch_stats["verifier_acc_real"] += v_acc_real
            epoch_stats["verifier_acc_fake"] += v_acc_fake
            
            # Train prover every step  
            p_loss = self.train_prover_step(batch)
            epoch_stats["prover_loss"] += p_loss
            
            # Train adversary every other step (to balance the game)
            if batch_idx % 2 == 0:
                a_loss, a_success = self.train_adversary_step(batch)
                epoch_stats["adversary_loss"] += a_loss * 2  # Scale for averaging
                epoch_stats["adversary_success"] += a_success * 2
                
        # Average the statistics
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
            
        return epoch_stats

    def train(
        self, 
        num_epochs: int = 100,
        num_samples: int = 10000,
        save_interval: int = 10,
        save_dir: str = "./models"
    ) -> Dict[str, List[float]]:
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate training data
        dataloader = self.generate_training_data(num_samples)
        
        print(f"Starting training on {self.device}")
        print(f"Training for {num_epochs} epochs with {num_samples} samples")
        
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            epoch_stats = self.train_epoch(dataloader, epoch)
            
            # Log statistics
            for key, value in epoch_stats.items():
                # Map epoch stat keys to training stat keys
                if key == "prover_loss":
                    self.training_stats["prover_losses"].append(value)
                elif key == "verifier_loss":
                    self.training_stats["verifier_losses"].append(value)
                elif key == "adversary_loss":
                    self.training_stats["adversary_losses"].append(value)
                elif key == "verifier_acc_real":
                    self.training_stats["verifier_accuracy_real"].append(value)
                elif key == "verifier_acc_fake":
                    self.training_stats["verifier_accuracy_fake"].append(value)
                elif key == "adversary_success":
                    self.training_stats["adversary_success_rate"].append(value)
                
                self.writer.add_scalar(f"Training/{key}", value, epoch)
                
            # Print progress
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}/{num_epochs}:")
                print(f"  Prover Loss: {epoch_stats['prover_loss']:.4f}")
                print(f"  Verifier Loss: {epoch_stats['verifier_loss']:.4f}")
                print(f"  Adversary Loss: {epoch_stats['adversary_loss']:.4f}")
                print(f"  Verifier Acc (Real): {epoch_stats['verifier_acc_real']:.4f}")
                print(f"  Verifier Acc (Fake): {epoch_stats['verifier_acc_fake']:.4f}")
                print(f"  Adversary Success: {epoch_stats['adversary_success']:.4f}")
                
            # Save models periodically
            if epoch % save_interval == 0:
                self.save_models(save_dir, epoch)
                
        # Final save
        self.save_models(save_dir, "final")
        self.writer.close()
        
        return self.training_stats
        
    def save_models(self, save_dir: str, epoch: str = "final"):
        """Save all models to disk."""
        torch.save(self.prover.state_dict(), 
                  os.path.join(save_dir, f"prover_epoch_{epoch}.pth"))
        torch.save(self.verifier.state_dict(), 
                  os.path.join(save_dir, f"verifier_epoch_{epoch}.pth"))
        torch.save(self.adversary.state_dict(), 
                  os.path.join(save_dir, f"adversary_epoch_{epoch}.pth"))
                  
    def save_models_as_onnx(self, save_dir: str, input_shape: Tuple[int, ...] = (1, 196)):
        """Save prover and verifier as ONNX models for deployment."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create dummy inputs
        dummy_input = torch.randn(input_shape, device=self.device)
        dummy_output = torch.randn((input_shape[0], 10), device=self.device)  # 10 classes
        dummy_proof = torch.randn((input_shape[0], self.prover.proof_dim), device=self.device)
        
        # Export prover
        torch.onnx.export(
            self.prover,
            dummy_input,
            os.path.join(save_dir, "zk_prover.onnx"),
            input_names=["input"],
            output_names=["output", "proof"],
            dynamic_axes={"input": {0: "batch_size"}},
            opset_version=10
        )
        
        # Export verifier  
        torch.onnx.export(
            self.verifier,
            (dummy_input, dummy_output, dummy_proof),
            os.path.join(save_dir, "zk_verifier.onnx"),
            input_names=["input", "output", "proof"],
            output_names=["verification"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
                "proof": {0: "batch_size"}
            },
            opset_version=10
        )
        
        print(f"ONNX models saved to {save_dir}")
        
    def plot_training_progress(self, save_path: str = None):
        """Plot comprehensive training statistics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot losses
        epochs = range(1, len(self.training_stats["prover_losses"]) + 1)
        
        axes[0, 0].plot(epochs, self.training_stats["prover_losses"], label="Prover", color='blue')
        axes[0, 0].plot(epochs, self.training_stats["verifier_losses"], label="Verifier", color='green') 
        axes[0, 0].plot(epochs, self.training_stats["adversary_losses"], label="Adversary", color='red')
        axes[0, 0].set_title("Training Losses")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot verifier accuracies
        axes[0, 1].plot(epochs, self.training_stats["verifier_accuracy_real"], 
                       label="Real Proofs", color='blue')
        axes[0, 1].plot(epochs, self.training_stats["verifier_accuracy_fake"], 
                       label="Fake Proofs", color='red')
        axes[0, 1].set_title("Verifier Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot adversary success rate
        axes[0, 2].plot(epochs, self.training_stats["adversary_success_rate"], 
                       color='red', linewidth=2)
        axes[0, 2].set_title("Adversary Success Rate")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Success Rate")
        axes[0, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()
