"""
Benchmarking suite for ZK networks performance and model evaluation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import json
import os
from typing import Dict, List, Tuple, Union
import numpy as np
from .networks import ZKProverNet, ZKVerifierNet, ZKAdversarialNet


class ZKBenchmark:
    """Comprehensive benchmarking suite for ZK networks."""
    
    def __init__(
        self,
        prover: ZKProverNet,
        verifier: ZKVerifierNet,
        adversary: ZKAdversarialNet,
        device: str = "auto"
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move networks to device
        self.prover = prover.to(self.device)
        self.verifier = verifier.to(self.device)
        self.adversary = adversary.to(self.device)
        
        # Set networks to evaluation mode
        self.prover.eval()
        self.verifier.eval()
        self.adversary.eval()
        
        # Benchmark results storage
        self.results = {}
        
    def generate_test_data(self, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate test data using real MNIST dataset."""
        from torchvision import datasets, transforms
        
        # Transform to flatten to 196 dimensions (14x14)
        transform = transforms.Compose([
            transforms.Resize((14, 14)),  # Resize to 14x14 to match our network
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 196 dims
        ])
        
        # Load MNIST test dataset
        test_dataset = datasets.MNIST(
            root='./data', 
            train=False,  # Use test set
            download=True, 
            transform=transform
        )
        
        # Create subset if requested
        if num_samples < len(test_dataset):
            indices = torch.randperm(len(test_dataset))[:num_samples]
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
        
        # Extract data and labels
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        data, labels = next(iter(test_loader))
        
        return data, labels
        
    def benchmark_classification_accuracy(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """Benchmark classification accuracy of prover network."""
        print("Benchmarking classification accuracy...")
        
        test_dataset = TensorDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                log_probs, _ = self.prover(batch_data)
                probs = torch.exp(log_probs)
                predictions = torch.argmax(log_probs, dim=1)
                confidences = torch.max(probs, dim=1)[0]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Calculate accuracy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        accuracy = np.mean(all_predictions == all_labels)
        avg_confidence = np.mean(all_confidences)
        
        # Per-class accuracy
        per_class_acc = {}
        for digit in range(10):
            mask = all_labels == digit
            if np.sum(mask) > 0:
                per_class_acc[f"digit_{digit}"] = np.mean(all_predictions[mask] == all_labels[mask])
        
        return {
            "overall_accuracy": accuracy,
            "average_confidence": avg_confidence,
            **per_class_acc
        }
        
    def benchmark_proof_verification(self, test_data: torch.Tensor, num_samples: int = 500) -> Dict[str, float]:
        """Benchmark proof generation and verification."""
        print("Benchmarking proof verification...")
        
        # Use subset for verification benchmarking
        test_subset = test_data[:num_samples].to(self.device)
        
        # Track verification performance
        real_verifications = []
        fake_verifications = []
        
        with torch.no_grad():
            # Generate real proofs from prover
            real_log_probs, real_proofs = self.prover(test_subset)
            real_predictions = self.verifier(test_subset, real_log_probs, real_proofs)
            real_verifications = (real_predictions > 0.5).float().cpu().numpy()
            
            # Generate fake proofs from adversary
            fake_log_probs, fake_proofs = self.adversary(test_subset)
            fake_predictions = self.verifier(test_subset, fake_log_probs, fake_proofs)
            fake_verifications = (fake_predictions > 0.5).float().cpu().numpy()
        
        # Calculate metrics
        real_acceptance_rate = np.mean(real_verifications)
        fake_acceptance_rate = np.mean(fake_verifications)
        
        # Overall verifier accuracy (should accept real, reject fake)
        verifier_accuracy = (real_acceptance_rate + (1 - fake_acceptance_rate)) / 2
        
        return {
            "real_proof_acceptance_rate": real_acceptance_rate,
            "fake_proof_acceptance_rate": fake_acceptance_rate,
            "verifier_discrimination_accuracy": verifier_accuracy,
            "real_avg_confidence": float(torch.mean(real_predictions).cpu()),
            "fake_avg_confidence": float(torch.mean(fake_predictions).cpu())
        }
        
    def benchmark_inference_speed(self, test_data: torch.Tensor, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference speed of all networks."""
        print("Benchmarking inference speed...")
        
        # Use single sample for speed testing
        single_input = test_data[:1].to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                self.prover(single_input)
                self.verifier(single_input, torch.randn(1, 10, device=self.device), 
                            torch.randn(1, self.prover.proof_dim, device=self.device))
                self.adversary(single_input)
        
        # Benchmark prover
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                self.prover(single_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        prover_time = (time.time() - start_time) / num_runs
        
        # Benchmark verifier
        dummy_output = torch.randn(1, 10, device=self.device)
        dummy_proof = torch.randn(1, self.prover.proof_dim, device=self.device)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                self.verifier(single_input, dummy_output, dummy_proof)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        verifier_time = (time.time() - start_time) / num_runs
        
        # Benchmark adversary
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                self.adversary(single_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        adversary_time = (time.time() - start_time) / num_runs
        
        return {
            "prover_inference_time_ms": prover_time * 1000,
            "verifier_inference_time_ms": verifier_time * 1000,
            "adversary_inference_time_ms": adversary_time * 1000,
            "total_proof_generation_time_ms": prover_time * 1000,
            "total_verification_time_ms": verifier_time * 1000
        }
        
    def benchmark_model_complexity(self) -> Dict[str, Union[int, float]]:
        """Benchmark model complexity and memory usage."""
        print("Benchmarking model complexity...")
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        def estimate_memory_mb(model):
            total_params = count_parameters(model)
            # Assume 4 bytes per parameter (float32)
            return (total_params * 4) / (1024 * 1024)
        
        prover_params = count_parameters(self.prover)
        verifier_params = count_parameters(self.verifier)
        adversary_params = count_parameters(self.adversary)
        
        return {
            "prover_parameters": prover_params,
            "verifier_parameters": verifier_params,  
            "adversary_parameters": adversary_params,
            "total_parameters": prover_params + verifier_params + adversary_params,
            "prover_memory_mb": estimate_memory_mb(self.prover),
            "verifier_memory_mb": estimate_memory_mb(self.verifier),
            "adversary_memory_mb": estimate_memory_mb(self.adversary),
            "total_memory_mb": estimate_memory_mb(self.prover) + estimate_memory_mb(self.verifier) + estimate_memory_mb(self.adversary)
        }
        
    def benchmark_adversarial_robustness(self, test_data: torch.Tensor, num_samples: int = 200) -> Dict[str, float]:
        """Benchmark robustness against adversarial examples."""
        print("Benchmarking adversarial robustness...")
        
        test_subset = test_data[:num_samples].to(self.device)
        
        with torch.no_grad():
            # Original predictions
            original_log_probs, _ = self.prover(test_subset)
            original_preds = torch.argmax(original_log_probs, dim=1)
            
            # Add small perturbations
            epsilon = 0.1
            perturbed_data = test_subset + epsilon * torch.randn_like(test_subset)
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            
            # Perturbed predictions
            perturbed_log_probs, _ = self.prover(perturbed_data)
            perturbed_preds = torch.argmax(perturbed_log_probs, dim=1)
            
            # Calculate robustness
            robustness = (original_preds == perturbed_preds).float().mean().item()
            
            # Average confidence change
            original_conf = torch.max(torch.exp(original_log_probs), dim=1)[0]
            perturbed_conf = torch.max(torch.exp(perturbed_log_probs), dim=1)[0]
            conf_change = torch.abs(original_conf - perturbed_conf).mean().item()
        
        return {
            "adversarial_robustness": robustness,
            "average_confidence_change": conf_change,
            "perturbation_epsilon": epsilon
        }
        
    def run_comprehensive_benchmark(self, num_test_samples: int = 2000) -> Dict:
        """Run all benchmarks and return comprehensive results."""
        print(f"🚀 Starting comprehensive benchmark on {self.device}")
        print(f"Using {num_test_samples} test samples")
        
        # Auto-clear results directory
        os.makedirs("benchmark_results", exist_ok=True)
        
        # Generate test data
        test_data, test_labels = self.generate_test_data(num_test_samples)
        
        # Run all benchmarks
        results = {
            "device": self.device,
            "test_samples": num_test_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Classification accuracy
        results["classification"] = self.benchmark_classification_accuracy(test_data, test_labels)
        
        # Proof verification
        results["verification"] = self.benchmark_proof_verification(test_data)
        
        # Inference speed
        results["performance"] = self.benchmark_inference_speed(test_data)
        
        # Model complexity
        results["complexity"] = self.benchmark_model_complexity()
        
        # Adversarial robustness  
        results["robustness"] = self.benchmark_adversarial_robustness(test_data)
        
        # Store results
        self.results = results
        
        # Save to file
        results_file = "benchmark_results/benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"📊 Benchmark results saved to {results_file}")
        
        return results
        
    def print_benchmark_summary(self):
        """Print a formatted summary of benchmark results."""
        if not self.results:
            print("No benchmark results available. Run benchmark first.")
            return
            
        print("\n" + "="*80)
        print(" 📈 ZK NEURAL NETWORK BENCHMARK SUMMARY")
        print("="*80)
        
        # Classification Performance
        if "classification" in self.results:
            class_results = self.results["classification"]
            print(f"\n🎯 CLASSIFICATION PERFORMANCE:")
            print(f"   Overall Accuracy: {class_results['overall_accuracy']:.4f}")
            print(f"   Average Confidence: {class_results['average_confidence']:.4f}")
            
            # Show per-digit accuracy
            print(f"   Per-Digit Accuracy:")
            for digit in range(10):
                key = f"digit_{digit}"
                if key in class_results:
                    print(f"     Digit {digit}: {class_results[key]:.4f}")
        
        # Verification Performance
        if "verification" in self.results:
            verif_results = self.results["verification"]
            print(f"\n🔐 PROOF VERIFICATION PERFORMANCE:")
            print(f"   Real Proof Acceptance: {verif_results['real_proof_acceptance_rate']:.4f}")
            print(f"   Fake Proof Acceptance: {verif_results['fake_proof_acceptance_rate']:.4f}")
            print(f"   Verifier Accuracy: {verif_results['verifier_discrimination_accuracy']:.4f}")
            print(f"   Real Proof Confidence: {verif_results['real_avg_confidence']:.4f}")
            print(f"   Fake Proof Confidence: {verif_results['fake_avg_confidence']:.4f}")
        
        # Speed Performance
        if "performance" in self.results:
            perf_results = self.results["performance"]
            print(f"\n⚡ INFERENCE SPEED:")
            print(f"   Prover: {perf_results['prover_inference_time_ms']:.2f} ms")
            print(f"   Verifier: {perf_results['verifier_inference_time_ms']:.2f} ms")
            print(f"   Adversary: {perf_results['adversary_inference_time_ms']:.2f} ms")
        
        # Model Complexity
        if "complexity" in self.results:
            complex_results = self.results["complexity"]
            print(f"\n🧠 MODEL COMPLEXITY:")
            print(f"   Total Parameters: {complex_results['total_parameters']:,}")
            print(f"   Prover Parameters: {complex_results['prover_parameters']:,}")
            print(f"   Verifier Parameters: {complex_results['verifier_parameters']:,}")
            print(f"   Total Memory: {complex_results['total_memory_mb']:.2f} MB")
        
        # Robustness
        if "robustness" in self.results:
            robust_results = self.results["robustness"]
            print(f"\n🛡️ ADVERSARIAL ROBUSTNESS:")
            print(f"   Robustness Score: {robust_results['adversarial_robustness']:.4f}")
            print(f"   Confidence Change: {robust_results['average_confidence_change']:.4f}")
        
        print("\n" + "="*80)
