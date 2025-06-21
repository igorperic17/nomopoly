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
        Perform one training step with comprehensive triplet verification.
        
        Tests all four critical cases:
        ✅ Real proofs + Real results: Should be ~100% accepted
        ❌ Fake proofs + Fake results: Should be ~100% rejected  
        ❌ Fake proofs + Real results: Should be ~100% rejected
        ❌ Real proofs + Wrong results: Should be ~100% rejected
        
        Args:
            data_generator: Function that generates training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        # Generate training data
        training_data = data_generator()
        input_batch = training_data[0]
        
        # Initialize optimizers for dynamic networks if not already done
        if self.verifier_optimizer is None:
            if epoch == 0:
                print("🔧 Building verifier network layers...")
            # Run forward pass to build dynamic layers
            with torch.no_grad():  # Inference network is frozen
                real_result, real_proof = self.inference_net(input_batch)
            _ = self.verifier_net(input_batch, real_result, real_proof)  # This builds the layers
            self.verifier_optimizer = optim.Adam(self.verifier_net.parameters(), lr=0.002)  # Higher LR for verifier
            if epoch == 0:
                print("✅ Verifier optimizer created")
            
        if self.malicious_optimizer is None:
            if epoch == 0:
                print("🔧 Building malicious network layers...")
            # Build malicious network layers if needed
            fake_result, fake_proof = self.malicious_net(input_batch, mode="mixed")
            self.malicious_optimizer = optim.Adam(self.malicious_net.parameters(), lr=0.0005)  # Lower LR for adversary
            if epoch == 0:
                print("✅ Malicious optimizer created")
        
        # === STEP 1: Train Verifier with Four Types of Samples ===
        self.verifier_optimizer.zero_grad()
        
        # Generate authentic samples (from frozen inference network)
        with torch.no_grad():  # Inference network is frozen
            real_result, real_proof = self.inference_net(input_batch)
        
        # Generate fake samples (from trainable malicious network)
        fake_result, fake_proof = self.malicious_net(input_batch, mode="mixed")
        
        # Create wrong results by shuffling real results
        batch_size = real_result.shape[0]
        shuffle_indices = torch.randperm(batch_size, device=real_result.device)
        wrong_result = real_result[shuffle_indices]
        
        # === Four Types of Verification Samples ===
        
        # Type 1: ✅ Real proofs + Real results → Should accept (score = 1)
        real_real_verification = self.verifier_net(input_batch, real_result, real_proof)
        loss_real_real = self.criterion(real_real_verification, torch.ones_like(real_real_verification))
        
        # Type 2: ❌ Fake proofs + Fake results → Should reject (score = 0)
        fake_fake_verification = self.verifier_net(input_batch, fake_result, fake_proof)
        loss_fake_fake = self.criterion(fake_fake_verification, torch.zeros_like(fake_fake_verification))
        
        # Type 3: ❌ Fake proofs + Real results → Should reject (score = 0)
        fake_real_verification = self.verifier_net(input_batch, real_result, fake_proof)
        loss_fake_real = self.criterion(fake_real_verification, torch.zeros_like(fake_real_verification))
        
        # Type 4: ❌ Real proofs + Wrong results → Should reject (score = 0)
        real_wrong_verification = self.verifier_net(input_batch, wrong_result, real_proof)
        loss_real_wrong = self.criterion(real_wrong_verification, torch.zeros_like(real_wrong_verification))
        
        # Total verifier loss (all four cases)
        verifier_loss = loss_real_real + loss_fake_fake + loss_fake_real + loss_real_wrong
        verifier_loss.backward()
        self.verifier_optimizer.step()
        
        # === STEP 2: Train Malicious Network to fool the verifier ===
        self.malicious_optimizer.zero_grad()
        
        # Generate new fake samples for adversarial training
        fake_result_adv, fake_proof_adv = self.malicious_net(input_batch, mode="mixed")
        fake_fake_verification_adv = self.verifier_net(input_batch, fake_result_adv, fake_proof_adv)
        
        # Malicious network wants verifier to accept fake samples (score = 1)
        malicious_loss = self.criterion(fake_fake_verification_adv, torch.ones_like(fake_fake_verification_adv))
        malicious_loss.backward()
        self.malicious_optimizer.step()
        
        # === Calculate Comprehensive Metrics ===
        with torch.no_grad():
            # Accuracy for each verification type
            real_real_acc = (real_real_verification > 0.5).float().mean().item()  # Should be ~100%
            fake_fake_acc = (fake_fake_verification < 0.5).float().mean().item()  # Should be ~100%
            fake_real_acc = (fake_real_verification < 0.5).float().mean().item()  # Should be ~100%
            real_wrong_acc = (real_wrong_verification < 0.5).float().mean().item()  # Should be ~100%
            
            # Overall verification accuracy
            overall_accuracy = (real_real_acc + fake_fake_acc + fake_real_acc + real_wrong_acc) / 4
            
            # Adversarial metrics
            malicious_success = (fake_fake_verification_adv > 0.5).float().mean().item()
            
            # Score statistics
            real_real_mean = real_real_verification.mean().item()
            fake_fake_mean = fake_fake_verification.mean().item()
            fake_real_mean = fake_real_verification.mean().item()
            real_wrong_mean = real_wrong_verification.mean().item()
            
            # Score separation (real vs all fake types)
            fake_scores_mean = (fake_fake_mean + fake_real_mean + real_wrong_mean) / 3
            score_separation = real_real_mean - fake_scores_mean
        
        return {
            'verifier_loss': verifier_loss.item(),
            'malicious_loss': malicious_loss.item(),
            'overall_accuracy': overall_accuracy,
            'real_real_acc': real_real_acc,
            'fake_fake_acc': fake_fake_acc,
            'fake_real_acc': fake_real_acc,
            'real_wrong_acc': real_wrong_acc,
            'malicious_success': malicious_success,
            'real_real_mean': real_real_mean,
            'fake_fake_mean': fake_fake_mean,
            'fake_real_mean': fake_real_mean,
            'real_wrong_mean': real_wrong_mean,
            'score_separation': score_separation
        }

    def train(self, num_epochs: int = 100, num_samples: int = 5000) -> Dict[str, List[float]]:
        """Train the ZK system focusing on binary proof verification."""
        
        # Setup
        self.setup_plots_directory()
        data_generator = self.load_mnist_data(num_samples)
        
        print(f"Starting ZK comprehensive triplet verification training on {self.device}")
        print("🎯 Training Goals:")
        print("   Inference Network: FROZEN - generates authentic proofs")
        print("   Verifier Network: Learn to verify (input, output, proof) triplets")
        print("   Malicious Network: Generate fake outputs and fake proofs")
        print("   SUCCESS = High accuracy on all four verification types (>90%)")
        print("\n🔍 Four Verification Types:")
        print("   ✅ Real proofs + Real results: Should be ~100% accepted")
        print("   ❌ Fake proofs + Fake results: Should be ~100% rejected")
        print("   ❌ Fake proofs + Real results: Should be ~100% rejected")
        print("   ❌ Real proofs + Wrong results: Should be ~100% rejected")
        
        # Training statistics
        stats = {
            "verifier_loss": [],
            "malicious_loss": [],
            "overall_accuracy": [],
            "real_real_acc": [],
            "fake_fake_acc": [],
            "fake_real_acc": [],
            "real_wrong_acc": [],
            "malicious_success": [],
            "real_real_mean": [],
            "fake_fake_mean": [],
            "fake_real_mean": [],
            "real_wrong_mean": [],
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
                print(f"  🎯 Overall Accuracy: {step_stats['overall_accuracy']:.1%}")
                print(f"  ✅ Real+Real: {step_stats['real_real_acc']:.1%}")
                print(f"  ❌ Fake+Fake: {step_stats['fake_fake_acc']:.1%}")
                print(f"  ❌ Fake+Real: {step_stats['fake_real_acc']:.1%}")
                print(f"  ❌ Real+Wrong: {step_stats['real_wrong_acc']:.1%}")
                print(f"  🎭 Malicious Success: {step_stats['malicious_success']:.1%}")
                print(f"  📊 Score Gap: {step_stats['score_separation']:.3f}")
        
        print("\n✅ Comprehensive triplet verification training completed!")
        
        # Final analysis
        final_overall_accuracy = stats["overall_accuracy"][-1]
        final_real_real_acc = stats["real_real_acc"][-1]
        final_fake_fake_acc = stats["fake_fake_acc"][-1]
        final_fake_real_acc = stats["fake_real_acc"][-1]
        final_real_wrong_acc = stats["real_wrong_acc"][-1]
        final_malicious_success = stats["malicious_success"][-1]
        final_score_separation = stats["score_separation"][-1]
        
        print(f"\n📈 FINAL PERFORMANCE:")
        print(f"   🎯 Overall Verification Accuracy: {final_overall_accuracy:.1%}")
        print(f"   ✅ Real proofs + Real results: {final_real_real_acc:.1%}")
        print(f"   ❌ Fake proofs + Fake results: {final_fake_fake_acc:.1%}")
        print(f"   ❌ Fake proofs + Real results: {final_fake_real_acc:.1%}")
        print(f"   ❌ Real proofs + Wrong results: {final_real_wrong_acc:.1%}")
        print(f"   🎭 Malicious Success Rate: {final_malicious_success:.1%}")
        print(f"   📊 Score Separation: {final_score_separation:.3f}")
        
        # Evaluate training success
        if final_overall_accuracy > 0.95:
            print("🎉 EXCELLENT: Verifier achieved outstanding triplet verification!")
            success_level = "EXCELLENT"
        elif final_overall_accuracy > 0.90:
            print("✅ VERY GOOD: Strong triplet verification performance")
            success_level = "VERY_GOOD"
        elif final_overall_accuracy > 0.85:
            print("✅ GOOD: Decent triplet verification performance")
            success_level = "GOOD"
        elif final_overall_accuracy > 0.75:
            print("⚠️ MODERATE: Some learning occurred")
            success_level = "MODERATE"
        else:
            print("❌ POOR: Triplet verification failed")
            success_level = "FAILED"
            
        # Check adversarial balance
        if final_score_separation > 0.3:
            print("✅ HEALTHY: Good separation between real and fake proof scores")
        elif final_score_separation > 0.1:
            print("⚠️ WEAK: Some separation between proof scores")
        else:
            print("❌ NO SEPARATION: Verifier cannot distinguish proof types")
        
        # Save trained models
        self.save_trained_models()
        
        return stats
    
    def save_trained_models(self, models_dir: str = "models") -> Dict[str, str]:
        """
        Save trained model states to disk.
        
        Args:
            models_dir: Directory to save models
            
        Returns:
            Dictionary with paths to saved models
        """
        import os
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        model_paths = {}
        
        # Save inference network (prover)
        prover_path = os.path.join(models_dir, "zk_prover_net.pth")
        torch.save({
            'model_state_dict': self.inference_net.state_dict(),
            'model_config': {
                'input_dim': self.inference_net.input_dim,
                'output_dim': self.inference_net.output_dim,
                'proof_dim': self.inference_net.proof_dim
            }
        }, prover_path)
        model_paths['prover'] = prover_path
        
        # Save verifier network
        verifier_path = os.path.join(models_dir, "zk_verifier_net.pth")
        if hasattr(self.verifier_net, 'layers') and self.verifier_net.layers:
            torch.save({
                'model_state_dict': self.verifier_net.state_dict(),
                'model_config': {
                    'input_dim': self.verifier_net.input_dim,
                    'output_dim': self.verifier_net.output_dim,
                    'proof_dim': self.verifier_net.proof_dim
                }
            }, verifier_path)
            model_paths['verifier'] = verifier_path
        
        # Save adversary network
        adversary_path = os.path.join(models_dir, "zk_adversary_net.pth")
        torch.save({
            'model_state_dict': self.malicious_net.state_dict(),
            'model_config': {
                'input_dim': self.malicious_net.input_dim,
                'output_dim': self.malicious_net.output_dim,
                'proof_dim': self.malicious_net.proof_dim
            }
        }, adversary_path)
        model_paths['adversary'] = adversary_path
        
        print(f"💾 Trained models saved to {models_dir}/:")
        for model_type, path in model_paths.items():
            print(f"   {model_type.capitalize()}: {path}")
        
        return model_paths
    
    def load_trained_models(self, models_dir: str = "models") -> bool:
        """
        Load trained model states from disk.
        
        Args:
            models_dir: Directory containing saved models
            
        Returns:
            True if all models loaded successfully
        """
        import os
        
        try:
            # Load prover network
            prover_path = os.path.join(models_dir, "zk_prover_net.pth")
            if os.path.exists(prover_path):
                checkpoint = torch.load(prover_path, map_location=self.device)
                self.inference_net.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded prover from {prover_path}")
            
            # Load verifier network
            verifier_path = os.path.join(models_dir, "zk_verifier_net.pth")
            if os.path.exists(verifier_path):
                checkpoint = torch.load(verifier_path, map_location=self.device)
                # Build verifier layers if not already built
                if not hasattr(self.verifier_net, 'layers') or not self.verifier_net.layers:
                    config = checkpoint['model_config']
                    self.verifier_net._build_layers(config['input_dim'], config['output_dim'], config['proof_dim'])
                self.verifier_net.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded verifier from {verifier_path}")
            
            # Load adversary network
            adversary_path = os.path.join(models_dir, "zk_adversary_net.pth")
            if os.path.exists(adversary_path):
                checkpoint = torch.load(adversary_path, map_location=self.device)
                self.malicious_net.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded adversary from {adversary_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
        
    def create_training_plots(self, stats: Dict[str, List[float]]) -> List[str]:
        """
        Create comprehensive training plots using the plotting module.
        
        Args:
            stats: Training statistics dictionary
            
        Returns:
            List of saved plot paths
        """
        from .plotting import ZKPlotter
        
        plotter = ZKPlotter(self.plots_dir)
        return plotter.create_summary_report(stats)
