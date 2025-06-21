"""
Inference and analysis utilities for trained ZK-ML models.

This module provides functionality for:
- Loading trained models
- Running inference and verification
- Analyzing model performance
- Testing verification robustness
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm

from .networks import ZKProverNet, ZKVerifierNet, ZKAdversarialNet


class ZKInference:
    """
    Inference utilities for trained ZK-ML models.
    """
    
    def __init__(
        self,
        prover_net: ZKProverNet,
        verifier_net: ZKVerifierNet,
        adversary_net: ZKAdversarialNet,
        device: str = "mps"
    ):
        # Use MPS if available, otherwise fallback
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Move networks to device
        self.prover_net = prover_net.to(self.device)
        self.verifier_net = verifier_net.to(self.device)
        self.adversary_net = adversary_net.to(self.device)
        
        # Set to evaluation mode
        self.prover_net.eval()
        self.verifier_net.eval()
        self.adversary_net.eval()
        
        print(f"🔍 ZK Inference initialized on {self.device}")
    
    def load_test_data(self, num_samples: int = 1000) -> DataLoader:
        """Load MNIST test data for inference."""
        transform = transforms.Compose([
            transforms.Resize((14, 14)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 196
        ])
        
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        if num_samples < len(test_dataset):
            indices = torch.randperm(len(test_dataset))[:num_samples]
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
        
        dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        print(f"📊 Loaded {len(test_dataset)} test samples")
        return dataloader
    
    def run_comprehensive_verification(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Run comprehensive verification analysis on test data.
        
        Args:
            test_loader: DataLoader with test data
            
        Returns:
            Dictionary with comprehensive verification results
        """
        print("🔍 Running comprehensive verification analysis...")
        
        all_results = {
            'inputs': [],
            'true_labels': [],
            'real_outputs': [],
            'real_proofs': [],
            'fake_outputs': [],
            'fake_proofs': [],
            'real_real_scores': [],
            'fake_fake_scores': [],
            'fake_real_scores': [],
            'real_wrong_scores': [],
            'classification_correct': []
        }
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Processing batches")):
                data, targets = data.to(self.device), targets.to(self.device)
                batch_size = data.shape[0]
                
                # Generate real outputs and proofs
                real_outputs, real_proofs = self.prover_net(data)
                real_predictions = torch.argmax(real_outputs, dim=1)
                
                # Generate fake outputs and proofs
                fake_outputs, fake_proofs = self.adversary_net(data, mode="mixed")
                
                # Create wrong outputs by shuffling
                shuffle_indices = torch.randperm(batch_size, device=self.device)
                wrong_outputs = real_outputs[shuffle_indices]
                
                # Four types of verification
                real_real_scores = self.verifier_net(data, real_outputs, real_proofs)
                fake_fake_scores = self.verifier_net(data, fake_outputs, fake_proofs)
                fake_real_scores = self.verifier_net(data, real_outputs, fake_proofs)
                real_wrong_scores = self.verifier_net(data, wrong_outputs, real_proofs)
                
                # Classification accuracy
                classification_correct = (real_predictions == targets).float()
                
                # Store results
                all_results['inputs'].append(data.cpu())
                all_results['true_labels'].append(targets.cpu())
                all_results['real_outputs'].append(real_outputs.cpu())
                all_results['real_proofs'].append(real_proofs.cpu())
                all_results['fake_outputs'].append(fake_outputs.cpu())
                all_results['fake_proofs'].append(fake_proofs.cpu())
                all_results['real_real_scores'].append(real_real_scores.cpu())
                all_results['fake_fake_scores'].append(fake_fake_scores.cpu())
                all_results['fake_real_scores'].append(fake_real_scores.cpu())
                all_results['real_wrong_scores'].append(real_wrong_scores.cpu())
                all_results['classification_correct'].append(classification_correct.cpu())
        
        # Concatenate all results
        for key in all_results:
            all_results[key] = torch.cat(all_results[key], dim=0)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_verification_metrics(all_results)
        
        print("✅ Comprehensive verification analysis completed!")
        return {
            'results': all_results,
            'metrics': metrics
        }
    
    def _calculate_verification_metrics(self, results: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate comprehensive verification metrics."""
        
        # Classification metrics
        classification_accuracy = results['classification_correct'].mean().item()
        
        # Verification accuracies (should be close to 1.0)
        real_real_acc = (results['real_real_scores'] > 0.5).float().mean().item()
        fake_fake_acc = (results['fake_fake_scores'] < 0.5).float().mean().item()
        fake_real_acc = (results['fake_real_scores'] < 0.5).float().mean().item()
        real_wrong_acc = (results['real_wrong_scores'] < 0.5).float().mean().item()
        
        overall_verification_acc = (real_real_acc + fake_fake_acc + fake_real_acc + real_wrong_acc) / 4
        
        # Score statistics
        real_real_mean = results['real_real_scores'].mean().item()
        fake_fake_mean = results['fake_fake_scores'].mean().item()
        fake_real_mean = results['fake_real_scores'].mean().item()
        real_wrong_mean = results['real_wrong_scores'].mean().item()
        
        # Score separation
        fake_scores_mean = (fake_fake_mean + fake_real_mean + real_wrong_mean) / 3
        score_separation = real_real_mean - fake_scores_mean
        
        # Score standard deviations
        real_real_std = results['real_real_scores'].std().item()
        fake_fake_std = results['fake_fake_scores'].std().item()
        fake_real_std = results['fake_real_scores'].std().item()
        real_wrong_std = results['real_wrong_scores'].std().item()
        
        return {
            'classification_accuracy': classification_accuracy,
            'overall_verification_accuracy': overall_verification_acc,
            'real_real_accuracy': real_real_acc,
            'fake_fake_accuracy': fake_fake_acc,
            'fake_real_accuracy': fake_real_acc,
            'real_wrong_accuracy': real_wrong_acc,
            'real_real_score_mean': real_real_mean,
            'fake_fake_score_mean': fake_fake_mean,
            'fake_real_score_mean': fake_real_mean,
            'real_wrong_score_mean': real_wrong_mean,
            'score_separation': score_separation,
            'real_real_score_std': real_real_std,
            'fake_fake_score_std': fake_fake_std,
            'fake_real_score_std': fake_real_std,
            'real_wrong_score_std': real_wrong_std
        }
    
    def analyze_verification_robustness(self, test_loader: DataLoader, num_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze robustness of verification to different types of attacks.
        
        Args:
            test_loader: DataLoader with test data
            num_samples: Number of samples to analyze in detail
            
        Returns:
            Dictionary with robustness analysis results
        """
        print("🛡️ Analyzing verification robustness...")
        
        # Get a small sample for detailed analysis
        sample_data, sample_targets = next(iter(test_loader))
        sample_data = sample_data[:num_samples].to(self.device)
        sample_targets = sample_targets[:num_samples].to(self.device)
        
        robustness_results = {}
        
        with torch.no_grad():
            # Generate real outputs and proofs
            real_outputs, real_proofs = self.prover_net(sample_data)
            
            # Test different types of fake outputs
            fake_corrupted, proof_corrupted = self.adversary_net(sample_data, mode="corrupted")
            fake_random, proof_random = self.adversary_net(sample_data, mode="random")
            fake_mixed, proof_mixed = self.adversary_net(sample_data, mode="mixed")
            
            # Test verification against different attack types
            robustness_results['corrupted_attack'] = {
                'fake_fake_scores': self.verifier_net(sample_data, fake_corrupted, proof_corrupted),
                'fake_real_scores': self.verifier_net(sample_data, real_outputs, proof_corrupted)
            }
            
            robustness_results['random_attack'] = {
                'fake_fake_scores': self.verifier_net(sample_data, fake_random, proof_random),
                'fake_real_scores': self.verifier_net(sample_data, real_outputs, proof_random)
            }
            
            robustness_results['mixed_attack'] = {
                'fake_fake_scores': self.verifier_net(sample_data, fake_mixed, proof_mixed),
                'fake_real_scores': self.verifier_net(sample_data, real_outputs, proof_mixed)
            }
            
            # Real baseline
            robustness_results['real_baseline'] = {
                'real_real_scores': self.verifier_net(sample_data, real_outputs, real_proofs)
            }
        
        # Calculate robustness metrics
        robustness_metrics = {}
        for attack_type, scores in robustness_results.items():
            if attack_type == 'real_baseline':
                robustness_metrics[attack_type] = {
                    'acceptance_rate': (scores['real_real_scores'] > 0.5).float().mean().item(),
                    'mean_score': scores['real_real_scores'].mean().item(),
                    'std_score': scores['real_real_scores'].std().item()
                }
            else:
                robustness_metrics[attack_type] = {
                    'fake_fake_rejection_rate': (scores['fake_fake_scores'] < 0.5).float().mean().item(),
                    'fake_real_rejection_rate': (scores['fake_real_scores'] < 0.5).float().mean().item(),
                    'fake_fake_mean_score': scores['fake_fake_scores'].mean().item(),
                    'fake_real_mean_score': scores['fake_real_scores'].mean().item()
                }
        
        print("✅ Robustness analysis completed!")
        return {
            'results': robustness_results,
            'metrics': robustness_metrics
        }
    
    def run_sample_verification_demo(self, num_samples: int = 10) -> None:
        """
        Run a demonstration of verification on individual samples.
        
        Args:
            num_samples: Number of samples to demonstrate
        """
        print(f"🎭 Running verification demo on {num_samples} samples...")
        
        # Load test data
        transform = transforms.Compose([
            transforms.Resize((14, 14)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Get sample data
        sample_indices = torch.randperm(len(test_dataset))[:num_samples]
        sample_data = []
        sample_labels = []
        
        for idx in sample_indices:
            data, label = test_dataset[idx]
            sample_data.append(data)
            sample_labels.append(label)
        
        sample_data = torch.stack(sample_data).to(self.device)
        sample_labels = torch.tensor(sample_labels).to(self.device)
        
        print(f"\n{'Sample':<8} {'True':<6} {'Pred':<6} {'Conf':<8} {'Real+Real':<10} {'Fake+Fake':<10} {'Fake+Real':<10} {'Real+Wrong':<10}")
        print("-" * 80)
        
        with torch.no_grad():
            # Generate real outputs and proofs
            real_outputs, real_proofs = self.prover_net(sample_data)
            real_predictions = torch.argmax(real_outputs, dim=1)
            real_probs = torch.exp(real_outputs)  # Convert log probs to probs
            
            # Generate fake outputs and proofs
            fake_outputs, fake_proofs = self.adversary_net(sample_data, mode="mixed")
            
            # Create wrong outputs
            batch_size = sample_data.shape[0]
            shuffle_indices = torch.randperm(batch_size, device=self.device)
            wrong_outputs = real_outputs[shuffle_indices]
            
            # Run all four verification types
            real_real_scores = self.verifier_net(sample_data, real_outputs, real_proofs)
            fake_fake_scores = self.verifier_net(sample_data, fake_outputs, fake_proofs)
            fake_real_scores = self.verifier_net(sample_data, real_outputs, fake_proofs)
            real_wrong_scores = self.verifier_net(sample_data, wrong_outputs, real_proofs)
            
            for i in range(num_samples):
                true_label = sample_labels[i].item()
                pred_label = real_predictions[i].item()
                confidence = real_probs[i, pred_label].item()
                
                real_real_score = real_real_scores[i].item()
                fake_fake_score = fake_fake_scores[i].item()
                fake_real_score = fake_real_scores[i].item()
                real_wrong_score = real_wrong_scores[i].item()
                
                print(f"{i+1:<8} {true_label:<6} {pred_label:<6} {confidence:.1%} "
                      f"{real_real_score:.3f} {fake_fake_score:.3f} "
                      f"{fake_real_score:.3f} {real_wrong_score:.3f}")
        
        print("\n✅ Verification demo completed!")
        print("💡 Interpretation:")
        print("   Real+Real should be > 0.5 (accept authentic proofs)")
        print("   Fake+Fake should be < 0.5 (reject fake proofs)")
        print("   Fake+Real should be < 0.5 (reject fake proofs with real outputs)")
        print("   Real+Wrong should be < 0.5 (reject real proofs with wrong outputs)")


def load_trained_models(
    input_dim: int = 196,
    output_dim: int = 10,
    proof_dim: int = 64,
    device: str = "mps"
) -> Tuple[ZKProverNet, ZKVerifierNet, ZKAdversarialNet]:
    """
    Load trained ZK models. This is a placeholder - in practice you'd load
    from saved checkpoints.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension  
        proof_dim: Proof dimension
        device: Device to load models on
        
    Returns:
        Tuple of (prover, verifier, adversary) networks
    """
    print("🔄 Loading trained ZK models...")
    
    # Create fresh models (in practice, load from checkpoints)
    prover = ZKProverNet(input_dim, output_dim, proof_dim)
    verifier = ZKVerifierNet(input_dim, output_dim, proof_dim)
    adversary = ZKAdversarialNet(input_dim, output_dim, proof_dim)
    
    print("⚠️ Note: Using fresh models - implement checkpoint loading for production use")
    return prover, verifier, adversary


def run_inference_analysis(
    prover: ZKProverNet,
    verifier: ZKVerifierNet,
    adversary: ZKAdversarialNet,
    device: str = "mps",
    num_test_samples: int = 1000
) -> Dict[str, Any]:
    """
    Run comprehensive inference analysis on trained models.
    
    Args:
        prover: Trained prover network
        verifier: Trained verifier network
        adversary: Trained adversary network
        device: Device for inference
        num_test_samples: Number of test samples
        
    Returns:
        Complete analysis results
    """
    # Initialize inference engine
    inference_engine = ZKInference(prover, verifier, adversary, device)
    
    # Run sample demo
    inference_engine.run_sample_verification_demo(10)
    
    return {"status": "completed"} 