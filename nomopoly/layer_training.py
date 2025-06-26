"""
Per-Layer Verifiable Training System

This module implements adversarial training for individual layers where:
1. Verifier accepts triplets (input, output, proof) and outputs binary classification
2. Generator accepts input and produces fake output + proof
3. Prover generates real examples for training the verifier
4. Training focuses on making the verifier distinguish real vs fake layer computations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from .networks import VerifiableLayer, VerifiableLinear, VerifiableReLU, VerifiableConv2d


class LayerTrainer:
    """
    Per-layer adversarial trainer for verifiable computations.
    
    For each layer:
    1. Prover generates authentic (input, output, proof) triplets
    2. Generator creates fake (input, fake_output, fake_proof) triplets  
    3. Verifier learns to distinguish real from fake triplets
    4. Success measured by verifier's binary classification accuracy
    """
    
    def __init__(
        self,
        layer: VerifiableLayer,
        device: str = "mps",
        learning_rate_verifier: float = 0.002,
        learning_rate_generator: float = 0.0005
    ):
        # Use MPS if available, otherwise fallback
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"🚀 Using device: {self.device}")
        
        self.layer = layer.to(self.device)
        
        # Initialize optimizers - will be set up after first forward pass for dynamic layers
        self.verifier_optimizer = None
        self.generator_optimizer = None
        self.learning_rate_verifier = learning_rate_verifier
        self.learning_rate_generator = learning_rate_generator
        
        # Set up optimizers for layers with pre-built verifiers/generators
        if hasattr(self.layer, 'verifier') and self.layer.verifier is not None:
            self.verifier_optimizer = optim.Adam(
                self.layer.verifier.parameters(), 
                lr=learning_rate_verifier
            )
        if hasattr(self.layer, 'generator') and self.layer.generator is not None:
            self.generator_optimizer = optim.Adam(
                self.layer.generator.parameters(), 
                lr=learning_rate_generator
            )
        
        # Don't train the actual layer computation (like the linear weights)
        # Only train the verifier and generator networks
        if hasattr(self.layer, 'linear'):
            for param in self.layer.linear.parameters():
                param.requires_grad = False
        if hasattr(self.layer, 'conv2d'):
            for param in self.layer.conv2d.parameters():
                param.requires_grad = False
            
        self.criterion = nn.BCELoss()
        
    def generate_training_data(self, batch_size: int = 64) -> torch.Tensor:
        """Generate random input data for the layer."""
        if isinstance(self.layer, VerifiableLinear):
            return torch.randn(batch_size, self.layer.in_features, device=self.device)
        elif isinstance(self.layer, VerifiableReLU):
            return torch.randn(batch_size, self.layer.num_features, device=self.device)
        elif isinstance(self.layer, VerifiableConv2d):
            # Generate random conv2d input: [batch, in_channels, height, width]
            return torch.randn(batch_size, self.layer.in_channels, 8, 8, device=self.device)
        else:
            raise ValueError(f"Unknown layer type: {type(self.layer)}")
    
    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform one training step for the layer.
        
        Tests four critical verification cases:
        ✅ Real computation + Real proof: Should accept (score = 1)
        ❌ Fake computation + Fake proof: Should reject (score = 0)  
        ❌ Real computation + Fake proof: Should reject (score = 0)
        ❌ Fake computation + Real proof: Should reject (score = 0)
        """
        # Generate input data
        input_data = self.generate_training_data(batch_size)
        
        # === STEP 1: Generate Real Examples ===
        with torch.no_grad():
            real_output = self.layer.forward(input_data)
        real_proof = self.layer.generate_proof(input_data, real_output)
        
        # === STEP 2: Generate Fake Examples ===
        fake_output, fake_proof = self.layer.generate_fake(input_data)
        
        # Create optimizers if not already created (for dynamic layers like Conv2d)
        if self.verifier_optimizer is None and hasattr(self.layer, 'verifier') and self.layer.verifier is not None:
            self.verifier_optimizer = optim.Adam(
                self.layer.verifier.parameters(), 
                lr=self.learning_rate_verifier
            )
        if self.generator_optimizer is None and hasattr(self.layer, 'generator') and self.layer.generator is not None:
            self.generator_optimizer = optim.Adam(
                self.layer.generator.parameters(), 
                lr=self.learning_rate_generator
            )
        
        # Create mixed examples (real output + fake proof, fake output + real proof)
        batch_indices = torch.randperm(batch_size, device=self.device)
        mixed_real_output = real_output[batch_indices]
        mixed_real_proof = real_proof[batch_indices]
        
        # === STEP 3: Train Verifier ===
        if self.verifier_optimizer is not None:
            self.verifier_optimizer.zero_grad()
            
            # Type 1: ✅ Real computation + Real proof → Accept (score = 1)
            real_real_score = self.layer.verify_triplet(input_data, real_output, real_proof)
            loss_real_real = self.criterion(real_real_score, torch.ones_like(real_real_score))
            
            # Type 2: ❌ Fake computation + Fake proof → Reject (score = 0)
            fake_fake_score = self.layer.verify_triplet(input_data, fake_output, fake_proof)
            loss_fake_fake = self.criterion(fake_fake_score, torch.zeros_like(fake_fake_score))
            
            # Type 3: ❌ Real computation + Fake proof → Reject (score = 0)
            real_fake_score = self.layer.verify_triplet(input_data, real_output, fake_proof)
            loss_real_fake = self.criterion(real_fake_score, torch.zeros_like(real_fake_score))
            
            # Type 4: ❌ Fake computation + Real proof → Reject (score = 0)
            fake_real_score = self.layer.verify_triplet(input_data, fake_output, mixed_real_proof)
            loss_fake_real = self.criterion(fake_real_score, torch.zeros_like(fake_real_score))
            
            # Total verifier loss
            verifier_loss = loss_real_real + loss_fake_fake + loss_real_fake + loss_fake_real
            verifier_loss.backward(retain_graph=True)
            self.verifier_optimizer.step()
        else:
            # Fallback values if verifier optimizer is not available
            verifier_loss = torch.tensor(0.0, device=self.device)
            real_real_score = torch.ones(batch_size, 1, device=self.device) * 0.5
            fake_fake_score = torch.ones(batch_size, 1, device=self.device) * 0.5
            real_fake_score = torch.ones(batch_size, 1, device=self.device) * 0.5
            fake_real_score = torch.ones(batch_size, 1, device=self.device) * 0.5
        
        # === STEP 4: Train Generator (Adversarial) ===
        if self.generator_optimizer is not None:
            self.generator_optimizer.zero_grad()
            
            # Generator tries to fool verifier: wants fake examples to be classified as real
            fake_output_adv, fake_proof_adv = self.layer.generate_fake(input_data)
            fake_score_adv = self.layer.verify_triplet(input_data, fake_output_adv, fake_proof_adv)
            
            # Generator loss: wants verifier to output 1 (mistake fake for real)
            generator_loss = self.criterion(fake_score_adv, torch.ones_like(fake_score_adv))
            generator_loss.backward()
            self.generator_optimizer.step()
        else:
            # Fallback values if generator optimizer is not available
            generator_loss = torch.tensor(0.0, device=self.device)
            fake_score_adv = torch.ones(batch_size, 1, device=self.device) * 0.5
        
        # Calculate accuracy metrics
        with torch.no_grad():
            real_real_acc = (real_real_score > 0.5).float().mean().item()
            fake_fake_acc = (fake_fake_score < 0.5).float().mean().item()
            real_fake_acc = (real_fake_score < 0.5).float().mean().item()
            fake_real_acc = (fake_real_score < 0.5).float().mean().item()
            
            overall_verifier_acc = (real_real_acc + fake_fake_acc + real_fake_acc + fake_real_acc) / 4
            generator_fool_rate = (fake_score_adv > 0.5).float().mean().item()
        
        return {
            "verifier_loss": verifier_loss.item(),
            "generator_loss": generator_loss.item(),
            "verifier_accuracy": overall_verifier_acc,
            "generator_fool_rate": generator_fool_rate,
            "real_real_accuracy": real_real_acc,
            "fake_fake_accuracy": fake_fake_acc,
            "real_fake_accuracy": real_fake_acc,
            "fake_real_accuracy": fake_real_acc
        }
    
    def train(self, num_epochs: int = 100, batch_size: int = 64) -> Dict[str, List[float]]:
        """Train the layer's verifier and generator networks."""
        print(f"🎯 Training {type(self.layer).__name__} for {num_epochs} epochs...")
        
        # Tracking metrics
        metrics = {
            "verifier_loss": [],
            "generator_loss": [],
            "verifier_accuracy": [],
            "generator_fool_rate": [],
            "real_real_accuracy": [],
            "fake_fake_accuracy": [],
            "real_fake_accuracy": [],
            "fake_real_accuracy": []
        }
        
        for epoch in tqdm(range(num_epochs), desc="Training Layer"):
            step_metrics = self.train_step(batch_size)
            
            # Store metrics
            for key, value in step_metrics.items():
                metrics[key].append(value)
            
            # Print progress more frequently for longer training
            if num_epochs <= 100:
                # Print every 20 epochs for short training
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch + 1:3d}: "
                          f"Verifier Acc: {step_metrics['verifier_accuracy']:.3f}, "
                          f"Generator Fool Rate: {step_metrics['generator_fool_rate']:.3f}")
            else:
                # Print every 50 epochs for longer training
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1:3d}: "
                          f"Verifier Acc: {step_metrics['verifier_accuracy']:.3f}, "
                          f"Generator Fool Rate: {step_metrics['generator_fool_rate']:.3f}, "
                          f"V-Loss: {step_metrics['verifier_loss']:.3f}, "
                          f"G-Loss: {step_metrics['generator_loss']:.3f}")
        
        print(f"✅ Training completed! Final verifier accuracy: {metrics['verifier_accuracy'][-1]:.3f}")
        return metrics
    
    def evaluate_layer(self, num_samples: int = 1000) -> Dict[str, float]:
        """Evaluate the trained layer on fresh data."""
        self.layer.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            # Test on multiple batches
            for _ in range(num_samples // 64 + 1):
                batch_size = min(64, num_samples - total_samples)
                if batch_size <= 0:
                    break
                    
                input_data = self.generate_training_data(batch_size)
                
                # Real examples
                real_output = self.layer.forward(input_data)
                real_proof = self.layer.generate_proof(input_data, real_output)
                real_score = self.layer.verify_triplet(input_data, real_output, real_proof)
                
                # Fake examples
                fake_output, fake_proof = self.layer.generate_fake(input_data)
                fake_score = self.layer.verify_triplet(input_data, fake_output, fake_proof)
                
                # Count correct classifications
                real_correct = (real_score > 0.5).sum().item()
                fake_correct = (fake_score < 0.5).sum().item()
                
                total_correct += real_correct + fake_correct
                total_samples += batch_size * 2  # Both real and fake
        
        accuracy = total_correct / total_samples
        self.layer.train()
        
        return {"accuracy": accuracy}


class MultiLayerTrainer:
    """
    Trainer for multiple verifiable layers in sequence.
    Can train a simple verifiable neural network layer by layer.
    """
    
    def __init__(self, layers: List[VerifiableLayer], device: str = "mps"):
        self.layers = layers
        self.device = device
        self.layer_trainers = []
        
        for layer in layers:
            trainer = LayerTrainer(layer, device)
            self.layer_trainers.append(trainer)
    
    def train_all_layers(self, num_epochs: int = 100, batch_size: int = 64) -> Dict[str, Dict[str, List[float]]]:
        """Train all layers sequentially."""
        print(f"🚀 Training {len(self.layers)} layers sequentially...")
        
        all_metrics = {}
        
        for i, trainer in enumerate(self.layer_trainers):
            print(f"\n📌 Training Layer {i+1}/{len(self.layers)}: {type(trainer.layer).__name__}")
            layer_metrics = trainer.train(num_epochs, batch_size)
            all_metrics[f"layer_{i+1}"] = layer_metrics
        
        return all_metrics
    
    def evaluate_all_layers(self, num_samples: int = 1000) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained layers."""
        print("🔍 Evaluating all layers...")
        
        all_evaluations = {}
        
        for i, trainer in enumerate(self.layer_trainers):
            layer_eval = trainer.evaluate_layer(num_samples)
            all_evaluations[f"layer_{i+1}"] = layer_eval
            print(f"Layer {i+1} ({type(trainer.layer).__name__}): {layer_eval['accuracy']:.3f}")
        
        return all_evaluations
    
    def create_simple_verifiable_network(self) -> 'SimpleVerifiableNet':
        """Create a simple network using the trained verifiable layers."""
        return SimpleVerifiableNet(self.layers)


class SimpleVerifiableNet(nn.Module):
    """
    Simple neural network composed of verifiable layers.
    Each layer can generate proofs and verify computations.
    """
    
    def __init__(self, layers: List[VerifiableLayer]):
        super(SimpleVerifiableNet, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def forward_with_proofs(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass that collects proofs from each layer."""
        proofs = []
        current_input = x
        
        for layer in self.layers:
            current_output = layer.forward(current_input)
            proof = layer.generate_proof(current_input, current_output)
            proofs.append(proof)
            current_input = current_output
        
        return current_input, proofs
    
    def verify_computation(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Verify the computation step by step and return verification scores.
        
        Returns:
            output: Final network output
            verification_scores: List of verification scores for each layer
        """
        verification_scores = []
        current_input = x
        
        for layer in self.layers:
            current_output = layer.forward(current_input)
            proof = layer.generate_proof(current_input, current_output)
            score = layer.verify_triplet(current_input, current_output, proof)
            verification_scores.append(score.mean().item())  # Average over batch
            current_input = current_output
        
        return current_input, verification_scores


def create_demo_verifiable_network(device: str = "mps") -> Tuple[MultiLayerTrainer, SimpleVerifiableNet]:
    """
    Create a demo verifiable network with Linear -> ReLU -> Linear layers.
    
    Returns:
        trainer: MultiLayerTrainer for training the layers
        network: SimpleVerifiableNet for inference
    """
    print("🏗️  Creating demo verifiable network: Linear(10->20) -> ReLU(20) -> Linear(20->5)")
    
    layers = [
        VerifiableLinear(10, 20, proof_dim=16),
        VerifiableReLU(20, proof_dim=16),
        VerifiableLinear(20, 5, proof_dim=16)
    ]
    
    trainer = MultiLayerTrainer(layers, device)
    network = SimpleVerifiableNet(layers)
    
    return trainer, network


if __name__ == "__main__":
    # Demo usage
    print("🧪 Demo: Per-Layer Verifiable Training")
    
    # Create demo network
    trainer, network = create_demo_verifiable_network()
    
    # Train all layers
    metrics = trainer.train_all_layers(num_epochs=50, batch_size=32)
    
    # Evaluate layers
    evaluations = trainer.evaluate_all_layers(num_samples=500)
    
    # Test the full network
    print("\n🔍 Testing full verifiable network...")
    test_input = torch.randn(5, 10)
    
    # Normal forward pass
    output = network(test_input)
    print(f"Network output shape: {output.shape}")
    
    # Forward pass with proofs
    output_with_proofs, proofs = network.forward_with_proofs(test_input)
    print(f"Number of proofs collected: {len(proofs)}")
    
    # Verify computation
    verified_output, verification_scores = network.verify_computation(test_input)
    print(f"Verification scores per layer: {verification_scores}")
    
    print("✅ Demo completed!") 